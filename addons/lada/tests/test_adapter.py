from types import SimpleNamespace

import pytest

from localbooru_lada.adapter import LadaFrameSource


class ErrorMarker(Exception):
    def __init__(self, stack_trace):
        self.stack_trace = stack_trace


class FailingRestorer:
    def __next__(self):
        return ErrorMarker("worker traceback")


def test_lada_error_marker_is_reported_as_a_restoration_failure():
    source = object.__new__(LadaFrameSource)
    source._restorer = FailingRestorer()
    source._metadata = SimpleNamespace()

    with pytest.raises(RuntimeError, match="worker traceback"):
        next(source)
