"""
GPU Video Pipeline - Full GPU decode/encode using PyNvVideoCodec.

This module provides hardware-accelerated video decode and encode that keeps
frames on the GPU, avoiding costly CPU<->GPU transfers. Combined with GPU
optical flow and warping, this enables real-time 60fps interpolation at 1440p.

Performance targets:
- NVDEC decode: ~2-3ms per frame at 1440p
- NVENC encode: ~1-2ms per frame at 1440p
- Total decode+encode overhead: <5ms (leaving ~12ms for optical flow + warp)

Dependencies:
- PyNvVideoCodec: pip install PyNvVideoCodec
- PyTorch with CUDA (for tensor interop via DLPack)
"""
import logging
from typing import Optional, Tuple, List, Union, Generator
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Feature detection
HAS_PYNVVIDEOCODEC = False
HAS_TORCH = False
CUDA_AVAILABLE = False

try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    logger.info("PyTorch not available - GPU tensor ops disabled")

try:
    import PyNvVideoCodec as nvc
    HAS_PYNVVIDEOCODEC = True
    logger.info("PyNvVideoCodec available for GPU decode/encode")
except ImportError:
    logger.info("PyNvVideoCodec not installed - using fallback decode/encode")

# Fallback: try cv2 with CUDA
HAS_CV2 = False
HAS_CV2_CUDA = False
try:
    import cv2
    HAS_CV2 = True
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            HAS_CV2_CUDA = True
    except Exception:
        pass
except ImportError:
    pass


class GPUFrame:
    """
    Represents a video frame residing on the GPU.

    Can be backed by either:
    - PyTorch CUDA tensor (most flexible, good interop)
    - PyNvVideoCodec NvSurface (zero-copy from NVDEC)
    - OpenCV cuda_GpuMat (for OpenCV CUDA workflows)

    Provides conversion methods between formats with minimal copies.
    """

    def __init__(self, data: Union['torch.Tensor', 'cv2.cuda.GpuMat', 'nvc.NvSurface', None] = None,
                 width: int = 0, height: int = 0, format: str = 'bgr'):
        """
        Initialize GPU frame.

        Args:
            data: GPU-resident frame data (tensor, GpuMat, or NvSurface)
            width: Frame width
            height: Frame height
            format: Pixel format ('bgr', 'rgb', 'nv12', 'yuv420')
        """
        self._data = data
        self.width = width
        self.height = height
        self.format = format
        self._tensor_cache = None
        self._gpumat_cache = None

    @classmethod
    def from_tensor(cls, tensor: 'torch.Tensor', format: str = 'bgr') -> 'GPUFrame':
        """Create GPUFrame from PyTorch CUDA tensor."""
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")

        # Expect (H, W, C) or (C, H, W) format
        if tensor.dim() == 3:
            if tensor.shape[0] in (1, 3, 4):  # CHW format
                h, w = tensor.shape[1], tensor.shape[2]
            else:  # HWC format
                h, w = tensor.shape[0], tensor.shape[1]
        else:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")

        frame = cls(tensor, width=w, height=h, format=format)
        frame._tensor_cache = tensor
        return frame

    @classmethod
    def from_numpy(cls, array: np.ndarray, device: int = 0, format: str = 'bgr') -> 'GPUFrame':
        """Upload numpy array to GPU as GPUFrame."""
        if not HAS_TORCH or not CUDA_AVAILABLE:
            raise RuntimeError("PyTorch CUDA required for from_numpy")

        import torch
        h, w = array.shape[:2]
        tensor = torch.from_numpy(array).to(f'cuda:{device}')
        return cls.from_tensor(tensor, format=format)

    def to_tensor(self, format: str = 'hwc') -> 'torch.Tensor':
        """
        Get frame as PyTorch CUDA tensor.

        Args:
            format: Output format - 'hwc' (H,W,C) or 'chw' (C,H,W)

        Returns:
            CUDA tensor with frame data
        """
        if self._tensor_cache is not None:
            tensor = self._tensor_cache
        elif HAS_TORCH and isinstance(self._data, torch.Tensor):
            tensor = self._data
            self._tensor_cache = tensor
        elif HAS_PYNVVIDEOCODEC and hasattr(self._data, '__dlpack__'):
            # Convert NvSurface to tensor via DLPack (zero-copy)
            import torch
            tensor = torch.from_dlpack(self._data)
            self._tensor_cache = tensor
        elif HAS_CV2_CUDA and hasattr(self._data, 'download'):
            # Convert GpuMat to tensor (requires download+upload)
            import torch
            cpu_array = self._data.download()
            tensor = torch.from_numpy(cpu_array).cuda()
            self._tensor_cache = tensor
        else:
            raise RuntimeError("Cannot convert frame data to tensor")

        # Handle format conversion
        if format == 'chw' and tensor.dim() == 3 and tensor.shape[2] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        elif format == 'hwc' and tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)

        return tensor

    def to_gpumat(self) -> 'cv2.cuda.GpuMat':
        """Get frame as OpenCV cuda_GpuMat."""
        if not HAS_CV2_CUDA:
            raise RuntimeError("OpenCV CUDA required for to_gpumat")

        if self._gpumat_cache is not None:
            return self._gpumat_cache

        # Convert from tensor
        tensor = self.to_tensor(format='hwc')
        cpu_array = tensor.cpu().numpy()

        gpumat = cv2.cuda_GpuMat()
        gpumat.upload(cpu_array)
        self._gpumat_cache = gpumat
        return gpumat

    def to_numpy(self) -> np.ndarray:
        """Download frame to CPU as numpy array."""
        tensor = self.to_tensor(format='hwc')
        return tensor.cpu().numpy()

    def clone(self) -> 'GPUFrame':
        """Create a copy of this frame on GPU."""
        if self._tensor_cache is not None:
            return GPUFrame.from_tensor(self._tensor_cache.clone(), format=self.format)
        tensor = self.to_tensor()
        return GPUFrame.from_tensor(tensor.clone(), format=self.format)


class GPUVideoDecoder:
    """
    Hardware-accelerated video decoder using NVDEC.

    Decodes video frames directly to GPU memory, avoiding CPU copies.
    Falls back to OpenCV with GPU upload if PyNvVideoCodec unavailable.

    Usage:
        decoder = GPUVideoDecoder(video_path, gpu_id=0)
        if decoder.initialize():
            for frame in decoder.decode_frames():
                # frame is GPUFrame, data stays on GPU
                tensor = frame.to_tensor()
                # ... process tensor ...
    """

    def __init__(self, video_path: str, gpu_id: int = 0,
                 target_width: int = 0, target_height: int = 0):
        """
        Initialize GPU video decoder.

        Args:
            video_path: Path to video file
            gpu_id: GPU device ID for decoding
            target_width: Target width (0 = source width)
            target_height: Target height (0 = source height)
        """
        self.video_path = str(video_path)
        self.gpu_id = gpu_id
        self.target_width = target_width
        self.target_height = target_height

        self._decoder = None
        self._cap = None  # Fallback cv2.VideoCapture
        self._initialized = False

        # Video properties (populated on initialize)
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0
        self.duration = 0.0

        # Backend info
        self.backend = None

    def initialize(self) -> bool:
        """Initialize the decoder and get video properties."""
        if self._initialized:
            return True

        try:
            # Try PyNvVideoCodec first (best performance)
            if HAS_PYNVVIDEOCODEC:
                return self._init_pynvvideocodec()

            # Fallback to OpenCV
            if HAS_CV2:
                return self._init_opencv()

            logger.error("No video decode backend available")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize decoder: {e}")
            return False

    def _init_pynvvideocodec(self) -> bool:
        """Initialize using PyNvVideoCodec."""
        try:
            import PyNvVideoCodec as nvc

            # Create decoder
            self._decoder = nvc.CreateDecoder(
                input_file=self.video_path,
                gpu_id=self.gpu_id
            )

            # Get video info
            self.width = self._decoder.Width()
            self.height = self._decoder.Height()
            self.fps = self._decoder.FrameRate()
            self.frame_count = self._decoder.NumFrames()
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0

            # Apply target dimensions
            if self.target_width > 0:
                self.width = self.target_width
            if self.target_height > 0:
                self.height = self.target_height

            self.backend = "pynvvideocodec"
            self._initialized = True
            logger.info(f"PyNvVideoCodec decoder initialized: {self.width}x{self.height} @ {self.fps:.2f}fps")
            return True

        except Exception as e:
            logger.warning(f"PyNvVideoCodec init failed: {e}, trying fallback")
            self._decoder = None

            # Try OpenCV fallback
            if HAS_CV2:
                return self._init_opencv()
            return False

    def _init_opencv(self) -> bool:
        """Initialize using OpenCV (with GPU upload)."""
        try:
            import cv2

            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                logger.error(f"Failed to open video: {self.video_path}")
                return False

            # Get video properties
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0

            # Apply target dimensions
            if self.target_width > 0:
                self.width = self.target_width
            if self.target_height > 0:
                self.height = self.target_height

            self.backend = "opencv_cuda" if HAS_CV2_CUDA else "opencv_cpu"
            self._initialized = True
            logger.info(f"OpenCV decoder initialized: {self.width}x{self.height} @ {self.fps:.2f}fps")
            return True

        except Exception as e:
            logger.error(f"OpenCV init failed: {e}")
            return False

    def decode_frame(self, frame_idx: Optional[int] = None) -> Optional[GPUFrame]:
        """
        Decode a single frame.

        Args:
            frame_idx: Frame index to decode (None = next frame)

        Returns:
            GPUFrame with decoded data, or None if no more frames
        """
        if not self._initialized:
            if not self.initialize():
                return None

        if HAS_PYNVVIDEOCODEC and self._decoder is not None:
            return self._decode_frame_nvc(frame_idx)
        else:
            return self._decode_frame_opencv(frame_idx)

    def _decode_frame_nvc(self, frame_idx: Optional[int]) -> Optional[GPUFrame]:
        """Decode frame using PyNvVideoCodec."""
        try:
            import PyNvVideoCodec as nvc

            # Seek if needed
            if frame_idx is not None:
                self._decoder.Seek(frame_idx)

            # Decode frame
            surface = self._decoder.DecodeSingleFrame()
            if surface is None:
                return None

            # Convert NV12 to BGR on GPU if needed
            # PyNvVideoCodec returns NV12, we need BGR for most processing
            if HAS_TORCH:
                import torch
                # Use DLPack for zero-copy tensor creation
                tensor = torch.from_dlpack(surface)

                # Handle NV12->BGR conversion on GPU
                # For now, we'll accept NV12 and convert in the processing step
                return GPUFrame(tensor, self.width, self.height, format='nv12')
            else:
                return GPUFrame(surface, self.width, self.height, format='nv12')

        except Exception as e:
            logger.warning(f"NVC decode failed: {e}")
            return None

    def _decode_frame_opencv(self, frame_idx: Optional[int]) -> Optional[GPUFrame]:
        """Decode frame using OpenCV (with GPU upload)."""
        try:
            import cv2

            # Seek if needed
            if frame_idx is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = self._cap.read()
            if not ret:
                return None

            # Resize if needed
            if (self.target_width > 0 or self.target_height > 0):
                target_w = self.target_width or int(frame.shape[1])
                target_h = self.target_height or int(frame.shape[0])
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    frame = cv2.resize(frame, (target_w, target_h))

            # Upload to GPU
            if HAS_TORCH and CUDA_AVAILABLE:
                import torch
                tensor = torch.from_numpy(frame).to(f'cuda:{self.gpu_id}')
                return GPUFrame.from_tensor(tensor, format='bgr')
            elif HAS_CV2_CUDA:
                gpumat = cv2.cuda_GpuMat()
                gpumat.upload(frame)
                return GPUFrame(gpumat, frame.shape[1], frame.shape[0], format='bgr')
            else:
                # CPU fallback (not ideal but works)
                return GPUFrame.from_numpy(frame, format='bgr')

        except Exception as e:
            logger.warning(f"OpenCV decode failed: {e}")
            return None

    def decode_frames(self, start_frame: int = 0, end_frame: int = -1) -> Generator[GPUFrame, None, None]:
        """
        Generator that yields decoded frames.

        Args:
            start_frame: First frame to decode
            end_frame: Last frame to decode (-1 = all remaining)

        Yields:
            GPUFrame instances
        """
        if not self._initialized:
            if not self.initialize():
                return

        if end_frame < 0:
            end_frame = self.frame_count

        # Seek to start
        if start_frame > 0:
            if self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            elif self._decoder is not None:
                self._decoder.Seek(start_frame)

        for i in range(start_frame, end_frame):
            frame = self.decode_frame()
            if frame is None:
                break
            yield frame

    def seek(self, frame_idx: int) -> bool:
        """Seek to a specific frame."""
        if not self._initialized:
            return False

        try:
            if self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            elif self._decoder is not None:
                self._decoder.Seek(frame_idx)
            return True
        except Exception as e:
            logger.warning(f"Seek failed: {e}")
            return False

    def cleanup(self):
        """Release decoder resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._decoder = None
        self._initialized = False


class GPUVideoEncoder:
    """
    Hardware-accelerated video encoder using NVENC.

    Encodes frames directly from GPU memory to H.264/HEVC bitstream,
    avoiding CPU copies. Falls back to FFmpeg subprocess if PyNvVideoCodec
    unavailable.

    Usage:
        encoder = GPUVideoEncoder(width, height, fps=60, codec='h264')
        if encoder.initialize():
            for frame in frames:
                data = encoder.encode_frame(frame)
                output_file.write(data)
            encoder.finalize()
    """

    def __init__(self, width: int, height: int, fps: float = 60.0,
                 codec: str = 'h264', preset: str = 'p4',
                 crf: int = 23, gpu_id: int = 0):
        """
        Initialize GPU video encoder.

        Args:
            width: Output width
            height: Output height
            fps: Output frame rate
            codec: Codec ('h264' or 'hevc')
            preset: NVENC preset (p1=fastest to p7=slowest/best)
            crf: Quality level (lower = better, 18-28 typical)
            gpu_id: GPU device ID
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.gpu_id = gpu_id

        self._encoder = None
        self._ffmpeg_proc = None  # Fallback
        self._initialized = False
        self.backend = None

    def initialize(self, output_path: Optional[str] = None) -> bool:
        """
        Initialize the encoder.

        Args:
            output_path: Output file path (for FFmpeg fallback)

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Try PyNvVideoCodec first
            if HAS_PYNVVIDEOCODEC:
                return self._init_pynvvideocodec()

            # Fallback to FFmpeg subprocess
            if output_path:
                return self._init_ffmpeg(output_path)

            logger.error("No video encode backend available")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            return False

    def _init_pynvvideocodec(self) -> bool:
        """Initialize using PyNvVideoCodec."""
        try:
            import PyNvVideoCodec as nvc

            # Map codec name
            codec_map = {
                'h264': nvc.CudaVideoCodec.H264,
                'hevc': nvc.CudaVideoCodec.HEVC,
                'h265': nvc.CudaVideoCodec.HEVC,
            }
            codec_id = codec_map.get(self.codec.lower(), nvc.CudaVideoCodec.H264)

            # Create encoder
            self._encoder = nvc.CreateEncoder(
                width=self.width,
                height=self.height,
                codec=codec_id,
                preset=self.preset,
                gpu_id=self.gpu_id
            )

            self.backend = "pynvvideocodec"
            self._initialized = True
            logger.info(f"PyNvVideoCodec encoder initialized: {self.width}x{self.height} {self.codec}")
            return True

        except Exception as e:
            logger.warning(f"PyNvVideoCodec encoder init failed: {e}")
            self._encoder = None
            return False

    def _init_ffmpeg(self, output_path: str) -> bool:
        """Initialize using FFmpeg subprocess with NVENC."""
        import subprocess

        try:
            # Check for NVENC availability
            check = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                 capture_output=True, text=True, timeout=5)
            has_nvenc = 'h264_nvenc' in check.stdout

            # Build FFmpeg command
            encoder = 'h264_nvenc' if has_nvenc else 'libx264'

            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',  # stdin
            ]

            if has_nvenc:
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-tune', 'll',
                    '-rc', 'vbr',
                    '-cq', str(self.crf),
                    '-b:v', '0',
                ])
            else:
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', str(self.crf),
                ])

            cmd.extend([
                '-f', 'mp4' if output_path.endswith('.mp4') else 'matroska',
                output_path
            ])

            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            self.backend = "ffmpeg_nvenc" if has_nvenc else "ffmpeg_x264"
            self._initialized = True
            logger.info(f"FFmpeg encoder initialized: {self.width}x{self.height} ({self.backend})")
            return True

        except Exception as e:
            logger.error(f"FFmpeg encoder init failed: {e}")
            return False

    def encode_frame(self, frame: Union[GPUFrame, np.ndarray, 'torch.Tensor']) -> Optional[bytes]:
        """
        Encode a single frame.

        Args:
            frame: Frame to encode (GPUFrame, numpy array, or CUDA tensor)

        Returns:
            Encoded bitstream data, or None on error
        """
        if not self._initialized:
            logger.error("Encoder not initialized")
            return None

        try:
            # Get frame data in appropriate format
            if isinstance(frame, GPUFrame):
                if HAS_PYNVVIDEOCODEC and self._encoder is not None:
                    tensor = frame.to_tensor(format='hwc')
                    return self._encode_frame_nvc(tensor)
                else:
                    array = frame.to_numpy()
                    return self._encode_frame_ffmpeg(array)
            elif HAS_TORCH and isinstance(frame, torch.Tensor):
                if HAS_PYNVVIDEOCODEC and self._encoder is not None:
                    return self._encode_frame_nvc(frame)
                else:
                    array = frame.cpu().numpy() if frame.is_cuda else frame.numpy()
                    return self._encode_frame_ffmpeg(array)
            elif isinstance(frame, np.ndarray):
                return self._encode_frame_ffmpeg(frame)
            else:
                logger.error(f"Unsupported frame type: {type(frame)}")
                return None

        except Exception as e:
            logger.error(f"Encode frame failed: {e}")
            return None

    def _encode_frame_nvc(self, tensor: 'torch.Tensor') -> Optional[bytes]:
        """Encode frame using PyNvVideoCodec."""
        try:
            # PyNvVideoCodec accepts CUDA tensors via DLPack
            data = self._encoder.EncodeFrame(tensor)
            return bytes(data)
        except Exception as e:
            logger.warning(f"NVC encode failed: {e}")
            return None

    def _encode_frame_ffmpeg(self, array: np.ndarray) -> Optional[bytes]:
        """Encode frame using FFmpeg subprocess."""
        if self._ffmpeg_proc is None:
            return None

        try:
            # Ensure BGR uint8
            if array.dtype != np.uint8:
                array = (array.clip(0, 255)).astype(np.uint8)

            self._ffmpeg_proc.stdin.write(array.tobytes())
            return b''  # FFmpeg writes to file directly

        except BrokenPipeError:
            logger.warning("FFmpeg pipe broken")
            return None

    def encode_frames(self, frames: List[Union[GPUFrame, np.ndarray]]) -> List[bytes]:
        """Encode multiple frames."""
        results = []
        for frame in frames:
            data = self.encode_frame(frame)
            if data is not None:
                results.append(data)
        return results

    def finalize(self) -> Optional[bytes]:
        """
        Finalize encoding and return any remaining data.

        Call this after encoding all frames to flush the encoder
        and get final bitstream data.
        """
        try:
            if HAS_PYNVVIDEOCODEC and self._encoder is not None:
                data = self._encoder.Flush()
                return bytes(data) if data else None
            elif self._ffmpeg_proc is not None:
                self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=10)
                return None
        except Exception as e:
            logger.error(f"Finalize failed: {e}")
        return None

    def cleanup(self):
        """Release encoder resources."""
        if self._ffmpeg_proc is not None:
            try:
                if self._ffmpeg_proc.stdin:
                    self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=5)
            except:
                self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None
        self._encoder = None
        self._initialized = False


class GPUColorConverter:
    """
    GPU-accelerated color space conversion.

    Handles NV12 <-> BGR/RGB conversion on GPU to avoid CPU copies
    when decoding with NVDEC (which outputs NV12).
    """

    def __init__(self, gpu_id: int = 0):
        """Initialize color converter."""
        self.gpu_id = gpu_id
        self._device = None

        if HAS_TORCH and CUDA_AVAILABLE:
            import torch
            self._device = torch.device(f'cuda:{gpu_id}')

    def nv12_to_bgr(self, nv12_tensor: 'torch.Tensor', width: int, height: int) -> 'torch.Tensor':
        """
        Convert NV12 to BGR on GPU.

        NV12 format: Y plane (H*W) followed by interleaved UV plane (H/2 * W)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU color conversion")

        import torch
        import torch.nn.functional as F

        # NV12: Y plane is first H rows, UV plane is next H/2 rows (interleaved)
        y_size = height * width
        uv_size = (height // 2) * width

        # Reshape to planes
        y = nv12_tensor[:y_size].view(height, width).float()
        uv = nv12_tensor[y_size:y_size + uv_size].view(height // 2, width)

        # Separate U and V (interleaved)
        u = uv[:, 0::2].float()
        v = uv[:, 1::2].float()

        # Upsample UV to full resolution
        u = F.interpolate(u.unsqueeze(0).unsqueeze(0), size=(height, width),
                         mode='bilinear', align_corners=False).squeeze()
        v = F.interpolate(v.unsqueeze(0).unsqueeze(0), size=(height, width),
                         mode='bilinear', align_corners=False).squeeze()

        # YUV to BGR conversion (BT.601)
        y = y - 16
        u = u - 128
        v = v - 128

        r = 1.164 * y + 1.596 * v
        g = 1.164 * y - 0.392 * u - 0.813 * v
        b = 1.164 * y + 2.017 * u

        # Stack to BGR and clamp
        bgr = torch.stack([b, g, r], dim=-1).clamp(0, 255).byte()
        return bgr

    def bgr_to_nv12(self, bgr_tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Convert BGR to NV12 on GPU."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU color conversion")

        import torch
        import torch.nn.functional as F

        h, w = bgr_tensor.shape[:2]
        bgr = bgr_tensor.float()

        # BGR to YUV (BT.601)
        b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]

        y = 0.257 * r + 0.504 * g + 0.098 * b + 16
        u = -0.148 * r - 0.291 * g + 0.439 * b + 128
        v = 0.439 * r - 0.368 * g - 0.071 * b + 128

        # Downsample UV
        u_down = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), 2).squeeze()
        v_down = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 2).squeeze()

        # Interleave UV
        uv = torch.stack([u_down, v_down], dim=-1).view(-1)

        # Concatenate Y and UV planes
        nv12 = torch.cat([y.view(-1), uv]).byte()
        return nv12


def get_gpu_pipeline_status() -> dict:
    """Get status of GPU video pipeline components."""
    return {
        "pynvvideocodec_available": HAS_PYNVVIDEOCODEC,
        "torch_available": HAS_TORCH,
        "cuda_available": CUDA_AVAILABLE,
        "cv2_available": HAS_CV2,
        "cv2_cuda_available": HAS_CV2_CUDA,
        "recommended_backend": (
            "pynvvideocodec" if HAS_PYNVVIDEOCODEC else
            "opencv_cuda" if HAS_CV2_CUDA else
            "opencv_cpu" if HAS_CV2 else
            None
        )
    }
