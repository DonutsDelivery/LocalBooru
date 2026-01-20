"""
NVIDIA Optical Flow Worker - subprocess for Python 3.14 opencv-cuda.

This module provides a worker process that runs NVIDIA Optical Flow
using opencv-cuda compiled with Python 3.14. The main application
(Python 3.11) communicates with this worker via shared memory for
efficient frame transfer without copying.

Architecture:
    Main Process (Py 3.11)          Worker Process (Py 3.14)
    ─────────────────────           ────────────────────────
    [Decode frame to CPU]
           │
           ├── Shared Memory ──────► [Read from shared mem]
           │                                 │
           │                         [Upload to GPU]
           │                                 │
           │                         [Run NVIDIA OptFlow]
           │                                 │
           │                         [Download flow]
           │                                 │
           ◄── Shared Memory ──────── [Write flow to shared mem]
           │
    [Warp frames on GPU]

Performance:
- Shared memory transfer: ~1ms (vs ~17ms for pickling)
- NVIDIA Optical Flow: ~10.8ms @ 1440p
- Total worker round-trip: ~13ms (enables 60+ fps)
"""
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Worker Script (executed by Python 3.14)
# =============================================================================

# This script is embedded and executed by the subprocess.
# It uses stdin/stdout for commands and shared memory for frame data.
WORKER_SCRIPT = '''
"""NVOF Worker - runs in Python 3.14 with opencv-cuda"""
import json
import sys
import signal
import time
from multiprocessing import shared_memory

def main():
    import numpy as np

    # Signal handling for graceful shutdown
    running = True
    def signal_handler(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Try to import opencv-cuda
    nvof = None
    cuda_available = False

    try:
        import cv2
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        cuda_available = cuda_count > 0
    except Exception as e:
        send_response({"success": False, "error": f"OpenCV import failed: {e}"})
        return

    if not cuda_available:
        send_response({"success": False, "error": "OpenCV CUDA not available"})
        return

    # State
    width = 0
    height = 0
    preset = "fast"
    frame1_shm = None
    frame2_shm = None
    flow_shm = None

    def send_response(data):
        """Send JSON response to parent process"""
        sys.stdout.write(json.dumps(data) + "\\n")
        sys.stdout.flush()

    def init_nvof(w, h, p):
        """Initialize NVIDIA Optical Flow"""
        nonlocal nvof, width, height, preset

        width = w
        height = h
        preset = p

        preset_map = {
            'slow': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
            'medium': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_MEDIUM,
            'fast': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST,
        }
        perf_level = preset_map.get(preset, cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST)

        try:
            nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
                imageSize=(width, height),
                perfPreset=perf_level,
                enableCostBuffer=False,
                gpuId=0
            )
            return True
        except Exception as e:
            send_response({"success": False, "error": f"NVOF init failed: {e}"})
            return False

    def attach_shared_memory(name):
        """Attach to existing shared memory block"""
        try:
            return shared_memory.SharedMemory(name=name)
        except Exception as e:
            send_response({"success": False, "error": f"Failed to attach to shared memory {name}: {e}"})
            return None

    def compute_flow(frame1_shm_name, frame2_shm_name, flow_shm_name,
                     frame_shape, flow_shape):
        """Compute optical flow using NVOF"""
        nonlocal frame1_shm, frame2_shm, flow_shm

        if nvof is None:
            return {"success": False, "error": "NVOF not initialized"}

        try:
            # Attach to shared memory (reuse if same names)
            if frame1_shm is None or frame1_shm.name != frame1_shm_name:
                if frame1_shm:
                    frame1_shm.close()
                frame1_shm = attach_shared_memory(frame1_shm_name)
            if frame2_shm is None or frame2_shm.name != frame2_shm_name:
                if frame2_shm:
                    frame2_shm.close()
                frame2_shm = attach_shared_memory(frame2_shm_name)
            if flow_shm is None or flow_shm.name != flow_shm_name:
                if flow_shm:
                    flow_shm.close()
                flow_shm = attach_shared_memory(flow_shm_name)

            if not all([frame1_shm, frame2_shm, flow_shm]):
                return {"success": False, "error": "Failed to attach shared memory"}

            # Read frames from shared memory
            frame1 = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame1_shm.buf)
            frame2 = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame2_shm.buf)

            # Convert to grayscale if needed
            if len(frame_shape) == 3 and frame_shape[2] == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frame1, frame2

            # Upload to GPU
            gpu_gray1 = cv2.cuda_GpuMat()
            gpu_gray2 = cv2.cuda_GpuMat()
            gpu_gray1.upload(gray1)
            gpu_gray2.upload(gray2)

            # Compute optical flow - returns tuple (flow, cost)
            start_time = time.perf_counter()
            gpu_flow, _ = nvof.calc(gpu_gray1, gpu_gray2, None)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Download flow result (int16, 1/32 pixel precision)
            flow_raw = gpu_flow.download()

            # Convert to float32 and write to shared memory
            flow_float = flow_raw.astype(np.float32) / 32.0
            flow_out = np.ndarray(flow_shape, dtype=np.float32, buffer=flow_shm.buf)
            np.copyto(flow_out, flow_float)

            return {
                "success": True,
                "elapsed_ms": elapsed_ms,
                "flow_shape": list(flow_shape)
            }

        except Exception as e:
            return {"success": False, "error": f"Flow computation error: {e}"}

    # Send ready signal
    send_response({
        "success": True,
        "ready": True,
        "cuda_available": cuda_available,
        "cuda_devices": cv2.cuda.getCudaEnabledDeviceCount()
    })

    # Main command loop
    while running:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF

            cmd = json.loads(line.strip())
            command = cmd.get("command")

            if command == "init":
                w = cmd.get("width", 1920)
                h = cmd.get("height", 1080)
                p = cmd.get("preset", "fast")
                if init_nvof(w, h, p):
                    send_response({
                        "success": True,
                        "width": w,
                        "height": h,
                        "preset": p
                    })

            elif command == "compute_flow":
                result = compute_flow(
                    cmd["frame1_shm"],
                    cmd["frame2_shm"],
                    cmd["flow_shm"],
                    tuple(cmd["frame_shape"]),
                    tuple(cmd["flow_shape"])
                )
                send_response(result)

            elif command == "ping":
                send_response({"success": True, "pong": True})

            elif command == "shutdown":
                send_response({"success": True, "shutdown": True})
                break

            else:
                send_response({"success": False, "error": f"Unknown command: {command}"})

        except json.JSONDecodeError as e:
            send_response({"success": False, "error": f"JSON decode error: {e}"})
        except Exception as e:
            send_response({"success": False, "error": f"Error: {e}"})

    # Cleanup
    if nvof:
        try:
            nvof.collectGarbage()
        except:
            pass
    for shm in [frame1_shm, frame2_shm, flow_shm]:
        if shm:
            try:
                shm.close()
            except:
                pass

if __name__ == "__main__":
    main()
'''


class NvidiaOpticalFlowWorker:
    """
    Manager for the NVIDIA Optical Flow worker subprocess.

    Spawns a Python 3.14 subprocess with opencv-cuda for hardware-accelerated
    optical flow computation. Communication uses stdin/stdout for commands
    and shared memory for efficient frame data transfer.

    Usage:
        worker = NvidiaOpticalFlowWorker(1920, 1080)
        if await worker.start():
            flow = await worker.compute_flow(frame1, frame2)
            # flow is numpy array (H, W, 2) of motion vectors
        worker.stop()
    """

    # Python 3.14 paths to try (system Python with opencv-cuda)
    PYTHON_CANDIDATES = [
        "/usr/bin/python3",       # Default system Python (often 3.14 on Arch)
        "/usr/bin/python3.14",
        "/usr/local/bin/python3.14",
        "/opt/python3.14/bin/python",
    ]

    def __init__(self, width: int, height: int,
                 preset: str = "fast",
                 python_path: Optional[str] = None,
                 timeout: float = 5.0):
        """
        Initialize the worker manager.

        Args:
            width: Frame width
            height: Frame height
            preset: Performance preset ('slow', 'medium', 'fast')
            python_path: Path to Python 3.14 interpreter (auto-detected if None)
            timeout: Timeout for worker responses in seconds
        """
        self.width = width
        self.height = height
        self.preset = preset
        self.python_path = python_path or self._find_python314()
        self.timeout = timeout

        self._process: Optional[subprocess.Popen] = None
        self._initialized = False

        # Shared memory blocks
        self._frame1_shm: Optional[shared_memory.SharedMemory] = None
        self._frame2_shm: Optional[shared_memory.SharedMemory] = None
        self._flow_shm: Optional[shared_memory.SharedMemory] = None

        # Performance tracking
        self._total_requests = 0
        self._total_time_ms = 0.0

    def _find_python314(self) -> str:
        """Find Python 3.14 interpreter with opencv-cuda."""
        for path in self.PYTHON_CANDIDATES:
            if os.path.exists(path):
                # Verify it has opencv-cuda
                try:
                    result = subprocess.run(
                        [path, "-c",
                         "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and int(result.stdout.strip()) > 0:
                        logger.info(f"Found Python with opencv-cuda: {path}")
                        return path
                except (subprocess.TimeoutExpired, ValueError):
                    continue
                except Exception as e:
                    logger.debug(f"Python check failed for {path}: {e}")

        # No Python with opencv-cuda found
        logger.warning("No Python with opencv-cuda found, using system Python")
        return self.PYTHON_CANDIDATES[0] if os.path.exists(self.PYTHON_CANDIDATES[0]) else sys.executable

    def _allocate_shared_memory(self, frame_shape: Tuple[int, ...]):
        """Allocate shared memory blocks for frame and flow data."""
        h, w = frame_shape[:2]
        channels = frame_shape[2] if len(frame_shape) == 3 else 1

        # Frame buffers (uint8)
        frame_size = h * w * channels

        # Flow buffer (float32, 2 channels for x,y)
        # NVOF outputs at same resolution as input
        flow_size = h * w * 2 * 4  # float32 = 4 bytes

        # Clean up old shared memory
        self._cleanup_shared_memory()

        # Create new shared memory blocks
        try:
            self._frame1_shm = shared_memory.SharedMemory(create=True, size=frame_size)
            self._frame2_shm = shared_memory.SharedMemory(create=True, size=frame_size)
            self._flow_shm = shared_memory.SharedMemory(create=True, size=flow_size)
            logger.debug(f"Allocated shared memory: frame={frame_size}B, flow={flow_size}B")
        except Exception as e:
            logger.error(f"Failed to allocate shared memory: {e}")
            self._cleanup_shared_memory()
            raise

    def _cleanup_shared_memory(self):
        """Release shared memory blocks."""
        for shm in [self._frame1_shm, self._frame2_shm, self._flow_shm]:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        self._frame1_shm = None
        self._frame2_shm = None
        self._flow_shm = None

    def _send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send command to worker and wait for response."""
        if self._process is None or self._process.poll() is not None:
            logger.error("Worker process not running")
            return None

        try:
            # Send command
            cmd_json = json.dumps(command) + "\n"
            self._process.stdin.write(cmd_json)
            self._process.stdin.flush()

            # Read response (with timeout via select)
            import select
            ready, _, _ = select.select([self._process.stdout], [], [], self.timeout)
            if not ready:
                logger.error("Worker response timeout")
                return None

            response_line = self._process.stdout.readline()
            if not response_line:
                logger.error("Worker closed connection")
                return None

            return json.loads(response_line.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse worker response: {e}")
            return None
        except Exception as e:
            logger.error(f"Communication error with worker: {e}")
            return None

    async def start(self) -> bool:
        """
        Start the worker subprocess.

        Returns:
            True if worker started and initialized successfully
        """
        if self._initialized:
            return True

        try:
            # Start subprocess with Python 3.14
            self._process = subprocess.Popen(
                [self.python_path, "-c", WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            # Wait for ready signal
            import select
            ready, _, _ = select.select([self._process.stdout], [], [], self.timeout)
            if not ready:
                logger.error("Worker startup timeout")
                self.stop()
                return False

            response_line = self._process.stdout.readline()
            if not response_line:
                # Check stderr for error
                stderr = self._process.stderr.read()
                logger.error(f"Worker failed to start: {stderr}")
                self.stop()
                return False

            response = json.loads(response_line.strip())
            if not response.get("success"):
                logger.error(f"Worker initialization failed: {response.get('error')}")
                self.stop()
                return False

            logger.info(f"NVOF worker started (CUDA devices: {response.get('cuda_devices')})")

            # Initialize NVOF with dimensions
            init_response = self._send_command({
                "command": "init",
                "width": self.width,
                "height": self.height,
                "preset": self.preset
            })

            if not init_response or not init_response.get("success"):
                logger.error(f"NVOF init failed: {init_response}")
                self.stop()
                return False

            self._initialized = True
            logger.info(f"NVOF initialized: {self.width}x{self.height}, preset={self.preset}")
            return True

        except Exception as e:
            logger.error(f"Failed to start NVOF worker: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop the worker subprocess."""
        if self._process is not None:
            try:
                # Send shutdown command
                self._send_command({"command": "shutdown"})

                # Wait for graceful shutdown
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            except Exception as e:
                logger.warning(f"Error stopping worker: {e}")
                try:
                    self._process.kill()
                except:
                    pass

            self._process = None

        self._cleanup_shared_memory()
        self._initialized = False

    async def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame (numpy array, H x W x 3 BGR or H x W grayscale)
            frame2: Second frame (same format as frame1)

        Returns:
            Flow field as numpy array (H, W, 2) with x,y motion vectors,
            or None on error
        """
        if not self._initialized:
            if not await self.start():
                return None

        try:
            h, w = frame1.shape[:2]
            frame_shape = frame1.shape
            flow_shape = (h, w, 2)

            # Allocate/reallocate shared memory if needed
            if (self._frame1_shm is None or
                self._frame1_shm.size < frame1.nbytes):
                self._allocate_shared_memory(frame_shape)

                # Re-initialize NVOF if dimensions changed
                if h != self.height or w != self.width:
                    self.width = w
                    self.height = h
                    init_response = self._send_command({
                        "command": "init",
                        "width": w,
                        "height": h,
                        "preset": self.preset
                    })
                    if not init_response or not init_response.get("success"):
                        logger.error(f"NVOF reinit failed: {init_response}")
                        return None

            # Copy frames to shared memory
            frame1_arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=self._frame1_shm.buf)
            frame2_arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=self._frame2_shm.buf)
            np.copyto(frame1_arr, frame1)
            np.copyto(frame2_arr, frame2)

            # Send compute command
            start_time = time.perf_counter()
            response = self._send_command({
                "command": "compute_flow",
                "frame1_shm": self._frame1_shm.name,
                "frame2_shm": self._frame2_shm.name,
                "flow_shm": self._flow_shm.name,
                "frame_shape": list(frame_shape),
                "flow_shape": list(flow_shape)
            })
            total_time = (time.perf_counter() - start_time) * 1000

            if not response or not response.get("success"):
                logger.error(f"Flow computation failed: {response}")
                return None

            # Read flow from shared memory
            flow = np.ndarray(flow_shape, dtype=np.float32, buffer=self._flow_shm.buf).copy()

            # Track performance
            self._total_requests += 1
            nvof_time = response.get("elapsed_ms", 0)
            self._total_time_ms += nvof_time
            logger.debug(f"Flow computed: NVOF={nvof_time:.1f}ms, total={total_time:.1f}ms")

            return flow

        except Exception as e:
            logger.error(f"Error computing flow: {e}")
            return None

    async def ping(self) -> bool:
        """Check if worker is responsive."""
        if not self._initialized:
            return False

        response = self._send_command({"command": "ping"})
        return response is not None and response.get("pong", False)

    @property
    def is_running(self) -> bool:
        """Check if worker process is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def avg_compute_time_ms(self) -> float:
        """Average flow computation time in milliseconds."""
        if self._total_requests == 0:
            return 0.0
        return self._total_time_ms / self._total_requests

    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            'initialized': self._initialized,
            'running': self.is_running,
            'total_requests': self._total_requests,
            'avg_compute_time_ms': self.avg_compute_time_ms,
            'width': self.width,
            'height': self.height,
            'preset': self.preset,
            'python_path': self.python_path,
        }


class DirectNvidiaOpticalFlow:
    """
    Direct NVIDIA Optical Flow using opencv-cuda in the current process.

    Use this if the main application already has opencv-cuda available
    (e.g., running on Python 3.14). This avoids the subprocess overhead.

    For Python 3.11 main process with opencv-cuda on 3.14, use
    NvidiaOpticalFlowWorker instead.
    """

    def __init__(self, width: int, height: int, preset: str = "fast", gpu_id: int = 0):
        """
        Initialize direct optical flow.

        Args:
            width: Frame width
            height: Frame height
            preset: Performance preset ('slow', 'medium', 'fast')
            gpu_id: CUDA device ID
        """
        self.width = width
        self.height = height
        self.preset = preset
        self.gpu_id = gpu_id

        self._nvof = None
        self._initialized = False

        # Pre-allocated GPU mats
        self._gpu_gray1 = None
        self._gpu_gray2 = None
        self._gpu_flow = None

    def initialize(self) -> bool:
        """Initialize NVIDIA Optical Flow."""
        if self._initialized:
            return True

        try:
            import cv2

            # Check CUDA availability
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                logger.error("OpenCV CUDA not available")
                return False

            # Map preset
            preset_map = {
                'slow': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
                'medium': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_MEDIUM,
                'fast': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST,
            }
            perf_level = preset_map.get(self.preset, cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST)

            # Create NVIDIA Optical Flow
            self._nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
                imageSize=(self.width, self.height),
                perfPreset=perf_level,
                enableCostBuffer=False,
                gpuId=self.gpu_id
            )

            # Pre-allocate GPU mats
            self._gpu_gray1 = cv2.cuda_GpuMat()
            self._gpu_gray2 = cv2.cuda_GpuMat()

            self._initialized = True
            logger.info(f"Direct NVIDIA Optical Flow initialized: {self.width}x{self.height}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA Optical Flow: {e}")
            return False

    def compute_flow(self, frame1, frame2) -> Optional[np.ndarray]:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame (numpy BGR or grayscale)
            frame2: Second frame (same format)

        Returns:
            Flow field (H, W, 2) or None on error
        """
        if not self._initialized:
            if not self.initialize():
                return None

        import cv2

        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frame1, frame2

            # Upload to GPU
            self._gpu_gray1.upload(gray1)
            self._gpu_gray2.upload(gray2)

            # Compute flow - returns tuple (flow, cost)
            self._gpu_flow, _ = self._nvof.calc(self._gpu_gray1, self._gpu_gray2, None)

            # Download result
            flow = self._gpu_flow.download()

            # Convert from int16 (1/32 pixel) to float32
            return flow.astype(np.float32) / 32.0

        except Exception as e:
            logger.error(f"Flow computation failed: {e}")
            return None

    def compute_flow_gpu(self, gpu_gray1, gpu_gray2):
        """
        Compute optical flow with input/output on GPU.

        This is the fastest path - no CPU transfers at all.

        Args:
            gpu_gray1: First frame as GPU mat (grayscale)
            gpu_gray2: Second frame as GPU mat (grayscale)

        Returns:
            Flow field as GPU mat, or None on error
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            flow, _ = self._nvof.calc(gpu_gray1, gpu_gray2, None)
            return flow
        except Exception as e:
            logger.error(f"GPU flow computation failed: {e}")
            return None

    def cleanup(self):
        """Release resources."""
        if self._nvof is not None:
            try:
                self._nvof.collectGarbage()
            except:
                pass
        self._nvof = None
        self._gpu_gray1 = None
        self._gpu_gray2 = None
        self._gpu_flow = None
        self._initialized = False


def check_nvidia_of_available() -> dict:
    """
    Check NVIDIA Optical Flow availability.

    Checks both:
    1. Direct availability (current Python has opencv-cuda)
    2. Worker availability (system Python 3.14 has opencv-cuda)
    """
    result = {
        'available': False,
        'direct_available': False,
        'worker_available': False,
        'opencv_cuda': False,
        'nvof_version': None,
        'python_path': None,
        'error': None
    }

    # Check direct availability (current process)
    try:
        import cv2
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        result['opencv_cuda'] = cuda_count > 0

        if cuda_count > 0:
            try:
                nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
                    imageSize=(64, 64),
                    perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST,
                    gpuId=0
                )
                result['direct_available'] = True
                result['available'] = True
                result['nvof_version'] = '2.0'
                nvof.collectGarbage()
            except Exception as e:
                result['error'] = str(e)
    except ImportError:
        pass
    except Exception as e:
        result['error'] = str(e)

    # Check worker availability (system Python with opencv-cuda)
    if not result['direct_available']:
        worker = NvidiaOpticalFlowWorker(64, 64)
        result['python_path'] = worker.python_path

        try:
            # Quick check if Python has opencv-cuda
            check_result = subprocess.run(
                [worker.python_path, "-c",
                 "import cv2; "
                 "n=cv2.cuda.getCudaEnabledDeviceCount(); "
                 "print('OK' if n > 0 else 'NO')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if check_result.returncode == 0 and 'OK' in check_result.stdout:
                result['worker_available'] = True
                result['available'] = True
                result['nvof_version'] = '2.0 (worker)'
        except Exception as e:
            if not result['error']:
                result['error'] = f"Worker check failed: {e}"

    return result
