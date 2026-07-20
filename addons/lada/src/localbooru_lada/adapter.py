from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .constants import DEFAULT_MAX_CLIP_LENGTH


@dataclass(frozen=True)
class RestoredFrame:
    data: bytes
    width: int
    height: int
    stride: int
    pts_ns: int
    duration_ns: int


class LadaFrameSource:
    def __init__(
        self,
        *,
        source_path: str,
        start_ns: int,
        device: str,
        detection_model_path: str,
        restoration_model_path: str,
        fp16: bool = True,
        max_clip_length: int = DEFAULT_MAX_CLIP_LENGTH,
    ):
        source = Path(source_path)
        if not source.is_absolute() or not source.is_file():
            raise ValueError("source_path must be an existing absolute file")
        self._source_path = str(source)
        self._start_ns = start_ns
        self._device_name = device
        self._detection_model_path = detection_model_path
        self._restoration_model_path = restoration_model_path
        self._fp16 = fp16
        self._max_clip_length = max_clip_length
        self._restorer = None
        self._metadata = None

    def start(self) -> None:
        import torch
        from lada.restorationpipeline import FrameRestorer, load_models

        detection, restoration, pad_mode = load_models(
            torch.device(self._device_name),
            "basicvsrpp-v1.2",
            self._restoration_model_path,
            None,
            self._detection_model_path,
            self._fp16,
            False,
        )
        self._restorer = FrameRestorer(
            torch.device(self._device_name),
            self._source_path,
            self._max_clip_length,
            "basicvsrpp-v1.2",
            detection,
            restoration,
            pad_mode,
        )
        self._metadata = self._restorer.video_meta_data
        self._restorer.start(start_ns=self._start_ns)

    def __iter__(self):
        return self

    def __next__(self) -> RestoredFrame:
        if self._restorer is None or self._metadata is None:
            raise RuntimeError("frame source has not started")
        value = next(self._restorer)
        if isinstance(value, Exception):
            detail = getattr(value, "stack_trace", None) or str(value)
            raise RuntimeError(detail or type(value).__name__) from value
        if not isinstance(value, tuple) or len(value) != 2:
            raise StopIteration
        frame, pts = value
        array = frame.detach().to("cpu").contiguous().numpy()
        height, width, channels = array.shape
        if channels != 3:
            raise RuntimeError(f"expected BGR frame with three channels, got {array.shape}")
        pts_ns = int(Fraction(pts - self._metadata.start_pts) * self._metadata.time_base * 1_000_000_000)
        duration_ns = int(Fraction(1, 1) / self._metadata.video_fps_exact * 1_000_000_000)
        return RestoredFrame(
            data=array.tobytes(),
            width=width,
            height=height,
            stride=width * 3,
            pts_ns=pts_ns,
            duration_ns=duration_ns,
        )

    def stop(self) -> None:
        if self._restorer is not None:
            self._restorer.stop()
            self._restorer = None
