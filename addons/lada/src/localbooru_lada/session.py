from collections.abc import Callable

from .protocol import ProtocolError


class SessionController:
    def __init__(self, restorer_factory: Callable):
        self._restorer_factory = restorer_factory
        self._restorer = None
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def restorer(self):
        return self._restorer

    def open(self, *, start_ns: int) -> dict:
        if self._restorer is not None:
            raise ProtocolError("a restoration session is already open")
        return self._replace(start_ns=start_ns, request_id=0)

    def seek(self, *, start_ns: int, request_id: int) -> dict:
        if self._restorer is None:
            raise ProtocolError("cannot seek without an open session")
        return self._replace(start_ns=start_ns, request_id=request_id)

    def _replace(self, *, start_ns: int, request_id: int) -> dict:
        if start_ns < 0:
            raise ProtocolError("start_ns must not be negative")
        if self._restorer is not None:
            self._restorer.stop()
        self._generation += 1
        self._restorer = self._restorer_factory(start_ns)
        start = getattr(self._restorer, "start", None)
        if callable(start):
            start()
        return {
            "type": "generation_started",
            "request_id": request_id,
            "generation": self._generation,
            "start_ns": start_ns,
        }

    def accept_generation(self, generation: int) -> None:
        if generation != self._generation:
            raise ProtocolError(
                f"stale generation {generation}; active generation is {self._generation}"
            )

    def stop(self) -> None:
        if self._restorer is not None:
            self._restorer.stop()
            self._restorer = None
        self._generation += 1
