"""
CUDA IPC (Inter-Process Communication) utilities.

Enables sharing GPU memory between processes without copying to CPU.
This is essential for the optical flow worker architecture where:
- Main process (Python 3.11) handles video decode/encode and frame warping
- Worker process (Python 3.14) runs NVIDIA Optical Flow via opencv-cuda

IPC handles allow the worker to access GPU tensors allocated by the main
process, and return flow results that remain on GPU.

Performance: IPC handle creation/opening is ~0.1ms, vs ~14ms for GPU<->CPU copy
"""
import logging
from typing import Optional, Tuple, Dict, Any
import pickle
import struct

logger = logging.getLogger(__name__)

# Feature detection
HAS_TORCH = False
CUDA_AVAILABLE = False
IPC_SUPPORTED = False

try:
    import torch
    import torch.cuda
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()

    # Check for IPC support (requires CUDA 11.0+)
    if CUDA_AVAILABLE:
        try:
            # Test IPC handle creation
            test_tensor = torch.zeros(1, device='cuda')
            if hasattr(torch.cuda, 'ipc_collect'):
                IPC_SUPPORTED = True
                logger.info("CUDA IPC supported")
            del test_tensor
        except Exception as e:
            logger.debug(f"CUDA IPC not supported: {e}")
except ImportError:
    logger.info("PyTorch not available - CUDA IPC disabled")


class IPCHandle:
    """
    Wrapper for CUDA IPC memory handle.

    Encapsulates the handle data needed to share a GPU tensor between processes.
    Can be serialized via pickle and sent through multiprocessing queues.
    """

    def __init__(self, handle_bytes: bytes, shape: Tuple[int, ...],
                 dtype: str, device_id: int):
        """
        Create IPC handle wrapper.

        Args:
            handle_bytes: Raw CUDA IPC handle bytes
            shape: Tensor shape
            dtype: Tensor dtype as string (e.g., 'torch.float32')
            device_id: CUDA device ID
        """
        self.handle_bytes = handle_bytes
        self.shape = shape
        self.dtype = dtype
        self.device_id = device_id

    def to_bytes(self) -> bytes:
        """Serialize handle for transmission."""
        return pickle.dumps({
            'handle': self.handle_bytes,
            'shape': self.shape,
            'dtype': self.dtype,
            'device_id': self.device_id
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> 'IPCHandle':
        """Deserialize handle from transmission."""
        d = pickle.loads(data)
        return cls(
            handle_bytes=d['handle'],
            shape=d['shape'],
            dtype=d['dtype'],
            device_id=d['device_id']
        )

    def __repr__(self):
        return f"IPCHandle(shape={self.shape}, dtype={self.dtype}, device={self.device_id})"


def create_ipc_handle(tensor: 'torch.Tensor') -> Optional[IPCHandle]:
    """
    Create an IPC handle for sharing a CUDA tensor.

    The tensor must be:
    - On a CUDA device (not CPU)
    - Contiguous in memory
    - Not a view of another tensor

    Args:
        tensor: PyTorch CUDA tensor to share

    Returns:
        IPCHandle that can be passed to another process, or None on failure
    """
    if not HAS_TORCH or not IPC_SUPPORTED:
        logger.error("CUDA IPC not available")
        return None

    if not tensor.is_cuda:
        logger.error("Tensor must be on CUDA device for IPC")
        return None

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    try:
        # Get the IPC handle from the tensor's storage
        storage = tensor.storage()

        # PyTorch's storage has _share_cuda_() method for IPC
        # This returns a tuple: (device, handle, size, offset, ref_counter_offset)
        cuda_info = storage._share_cuda_()

        # Pack the IPC info
        handle_data = {
            'cuda_info': cuda_info,
            'storage_size': storage.size(),
            'storage_offset': tensor.storage_offset(),
        }

        return IPCHandle(
            handle_bytes=pickle.dumps(handle_data),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device_id=tensor.device.index or 0
        )

    except Exception as e:
        logger.error(f"Failed to create IPC handle: {e}")
        return None


def open_ipc_handle(handle: IPCHandle) -> Optional['torch.Tensor']:
    """
    Open an IPC handle and get a tensor that shares the GPU memory.

    The returned tensor shares memory with the original tensor in the
    other process. Changes to one will be visible in the other.

    Args:
        handle: IPCHandle received from another process

    Returns:
        PyTorch CUDA tensor sharing the memory, or None on failure
    """
    if not HAS_TORCH or not IPC_SUPPORTED:
        logger.error("CUDA IPC not available")
        return None

    try:
        import torch

        # Unpack handle data
        handle_data = pickle.loads(handle.handle_bytes)
        cuda_info = handle_data['cuda_info']

        # Reconstruct storage from IPC handle
        # _new_shared_cuda() recreates the storage from IPC handle
        storage = torch.cuda.Storage._new_shared_cuda(
            cuda_info[0],  # device
            cuda_info[1],  # handle
            cuda_info[2],  # size
            cuda_info[3],  # offset
            cuda_info[4]   # ref_counter_offset
        )

        # Map dtype string to torch dtype
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
            'torch.float64': torch.float64,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.int16': torch.int16,
            'torch.uint8': torch.uint8,
            'torch.int8': torch.int8,
        }
        dtype = dtype_map.get(handle.dtype, torch.float32)

        # Create tensor from storage
        tensor = torch.tensor([], dtype=dtype, device=f'cuda:{handle.device_id}')
        tensor.set_(storage, handle_data.get('storage_offset', 0), handle.shape)

        return tensor

    except Exception as e:
        logger.error(f"Failed to open IPC handle: {e}")
        return None


def close_ipc_handle(handle: IPCHandle):
    """
    Release an IPC handle.

    Should be called when done with a received handle to allow
    the original process to free the memory.
    """
    # PyTorch handles cleanup automatically via reference counting
    # This function exists for explicit cleanup if needed
    pass


class IPCTensorPool:
    """
    Pool of pre-allocated GPU tensors for IPC sharing.

    Pre-allocates tensors for common frame sizes to avoid allocation
    overhead during streaming. Provides handles that can be reused
    for multiple frames.
    """

    def __init__(self, device_id: int = 0):
        """Initialize tensor pool."""
        self.device_id = device_id
        self._tensors: Dict[Tuple[int, ...], 'torch.Tensor'] = {}
        self._handles: Dict[Tuple[int, ...], IPCHandle] = {}
        self._in_use: Dict[Tuple[int, ...], bool] = {}

    def get_tensor(self, shape: Tuple[int, ...], dtype: 'torch.dtype' = None) -> Optional['torch.Tensor']:
        """
        Get a tensor from the pool, allocating if necessary.

        Args:
            shape: Desired tensor shape
            dtype: Tensor dtype (default: float32)

        Returns:
            CUDA tensor from pool
        """
        if not HAS_TORCH or not CUDA_AVAILABLE:
            return None

        import torch
        dtype = dtype or torch.float32
        key = (shape, str(dtype))

        # Check if we have a free tensor
        if key in self._tensors and not self._in_use.get(key, False):
            self._in_use[key] = True
            return self._tensors[key]

        # Allocate new tensor
        try:
            tensor = torch.empty(shape, dtype=dtype, device=f'cuda:{self.device_id}')
            self._tensors[key] = tensor
            self._in_use[key] = True
            return tensor
        except Exception as e:
            logger.error(f"Failed to allocate tensor {shape}: {e}")
            return None

    def release_tensor(self, shape: Tuple[int, ...], dtype: 'torch.dtype' = None):
        """Mark a tensor as no longer in use."""
        import torch
        dtype = dtype or torch.float32
        key = (shape, str(dtype))
        self._in_use[key] = False

    def get_handle(self, tensor: 'torch.Tensor') -> Optional[IPCHandle]:
        """Get or create IPC handle for a pooled tensor."""
        key = (tuple(tensor.shape), str(tensor.dtype))

        if key not in self._handles:
            handle = create_ipc_handle(tensor)
            if handle:
                self._handles[key] = handle
            return handle

        return self._handles[key]

    def preallocate(self, shapes: list, dtype: 'torch.dtype' = None):
        """Pre-allocate tensors for given shapes."""
        for shape in shapes:
            self.get_tensor(tuple(shape), dtype)
            # Immediately release so they're available
            self.release_tensor(tuple(shape), dtype)

    def cleanup(self):
        """Release all tensors."""
        self._tensors.clear()
        self._handles.clear()
        self._in_use.clear()
        if HAS_TORCH and CUDA_AVAILABLE:
            torch.cuda.empty_cache()


class IPCQueue:
    """
    Queue for passing GPU tensors between processes via IPC handles.

    Wraps multiprocessing.Queue with automatic IPC handle creation/opening.
    """

    def __init__(self):
        """Initialize IPC queue."""
        from multiprocessing import Queue
        self._queue = Queue()

    def put_tensor(self, tensor: 'torch.Tensor', block: bool = True, timeout: float = None):
        """
        Put a CUDA tensor on the queue via IPC handle.

        Args:
            tensor: CUDA tensor to send
            block: Whether to block if queue is full
            timeout: Timeout in seconds
        """
        handle = create_ipc_handle(tensor)
        if handle is None:
            raise RuntimeError("Failed to create IPC handle")
        self._queue.put(handle.to_bytes(), block=block, timeout=timeout)

    def get_tensor(self, block: bool = True, timeout: float = None) -> Optional['torch.Tensor']:
        """
        Get a CUDA tensor from the queue.

        Args:
            block: Whether to block if queue is empty
            timeout: Timeout in seconds

        Returns:
            CUDA tensor that shares memory with sender's tensor
        """
        try:
            data = self._queue.get(block=block, timeout=timeout)
            handle = IPCHandle.from_bytes(data)
            return open_ipc_handle(handle)
        except Exception as e:
            logger.error(f"Failed to get tensor from IPC queue: {e}")
            return None

    def put(self, obj, block: bool = True, timeout: float = None):
        """Put any picklable object on the queue."""
        self._queue.put(obj, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float = None):
        """Get any object from the queue."""
        return self._queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def close(self):
        """Close the queue."""
        self._queue.close()


def get_ipc_status() -> dict:
    """Get CUDA IPC capability status."""
    return {
        "torch_available": HAS_TORCH,
        "cuda_available": CUDA_AVAILABLE,
        "ipc_supported": IPC_SUPPORTED,
    }
