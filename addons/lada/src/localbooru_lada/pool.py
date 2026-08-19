import mmap
import os
import tempfile
import threading
from dataclasses import dataclass


class PoolExhausted(RuntimeError):
    pass


class StaleLease(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameLease:
    buffer_id: int
    sequence: int
    generation: int
    capacity: int
    fd: int


@dataclass
class _Slot:
    fd: int
    mapping: mmap.mmap
    sequence: int | None = None
    generation: int | None = None


class FramePool:
    def __init__(self, *, buffer_count: int, buffer_capacity: int):
        if buffer_count < 1 or buffer_capacity < 1:
            raise ValueError("buffer count and capacity must be positive")
        self._capacity = buffer_capacity
        self._condition = threading.Condition()
        self._generation = 0
        self._next_sequence = 1
        self._closed = False
        self._slots = [self._create_slot(index) for index in range(buffer_count)]

    def _create_slot(self, index: int) -> _Slot:
        if hasattr(os, "memfd_create"):
            fd = os.memfd_create(
                f"localbooru-lada-frame-{index}",
                getattr(os, "MFD_CLOEXEC", 0),
            )
        else:
            fd, path = tempfile.mkstemp(prefix=f"localbooru-lada-frame-{index}-")
            os.unlink(path)
        os.ftruncate(fd, self._capacity)
        return _Slot(fd=fd, mapping=mmap.mmap(fd, self._capacity))

    @property
    def in_flight(self) -> int:
        with self._condition:
            return sum(slot.sequence is not None for slot in self._slots)

    @property
    def descriptors(self) -> list[dict]:
        return [
            {"buffer_id": index, "capacity": self._capacity, "fd": slot.fd}
            for index, slot in enumerate(self._slots)
        ]

    def acquire(self, *, generation: int, block: bool = True) -> FrameLease:
        with self._condition:
            if self._closed:
                raise RuntimeError("frame pool is closed")
            if self._generation not in (0, generation):
                raise StaleLease(f"generation {generation} is not active")
            if self._generation == 0:
                self._generation = generation
            while True:
                if generation != self._generation:
                    raise StaleLease(f"generation {generation} is stale")
                for buffer_id, slot in enumerate(self._slots):
                    if slot.sequence is None:
                        sequence = self._next_sequence
                        self._next_sequence += 1
                        slot.sequence = sequence
                        slot.generation = generation
                        return FrameLease(
                            buffer_id=buffer_id,
                            sequence=sequence,
                            generation=generation,
                            capacity=self._capacity,
                            fd=slot.fd,
                        )
                if not block:
                    raise PoolExhausted("all frame buffers are leased")
                self._condition.wait()
                if self._closed:
                    raise RuntimeError("frame pool is closed")

    def write(self, lease: FrameLease, data: bytes | memoryview) -> int:
        size = len(data)
        if size > self._capacity:
            raise ValueError(f"frame needs {size} bytes but buffer capacity is {self._capacity}")
        with self._condition:
            slot = self._validate(lease.buffer_id, lease.sequence, lease.generation)
            slot.mapping.seek(0)
            slot.mapping.write(data)
            return size

    def release(self, buffer_id: int, sequence: int, *, generation: int) -> None:
        with self._condition:
            slot = self._validate(buffer_id, sequence, generation)
            slot.sequence = None
            slot.generation = None
            self._condition.notify()

    def _validate(self, buffer_id: int, sequence: int, generation: int) -> _Slot:
        if self._closed:
            raise RuntimeError("frame pool is closed")
        if generation != self._generation:
            raise StaleLease(f"generation {generation} is stale")
        try:
            slot = self._slots[buffer_id]
        except IndexError as error:
            raise StaleLease(f"unknown buffer {buffer_id}") from error
        if slot.sequence != sequence or slot.generation != generation:
            raise StaleLease(f"buffer {buffer_id} lease is stale or already released")
        return slot

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        for slot in self._slots:
            slot.mapping.close()
            os.close(slot.fd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
