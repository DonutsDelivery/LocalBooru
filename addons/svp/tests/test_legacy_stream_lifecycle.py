import asyncio
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as svp_app  # noqa: E402


class JsonRequest:
    def __init__(self, payload=None):
        self.payload = payload or {}

    async def json(self):
        return self.payload


class FakeStream:
    def __init__(self, start_gate=None, ready=True):
        self.stream_id = f"fake-{id(self)}"
        self.start_gate = start_gate
        self.ready = ready
        self.started = asyncio.Event()
        self.stopped = False
        self.error = None
        self._duration = 120.0
        self._width = 1920
        self._height = 1080

    async def start(self):
        self.started.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        return True

    async def wait_for_ready(self, timeout=45):
        return self.ready

    async def stop_async(self):
        self.stopped = True

    def stop(self):
        self.stopped = True


class LegacyStreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        svp_app._active_streams.clear()
        svp_app._legacy_stream_epoch = 0
        self.request = JsonRequest(
            svp_app.PlayRequest(file_path="/media/video.mp4").model_dump()
        )
        self.stop_request = JsonRequest()
        self.common_patches = (
            patch.object(svp_app.os.path, "exists", return_value=True),
            patch.object(svp_app, "check_vspipe", return_value=True),
            patch.object(svp_app, "get_ffmpeg_path", return_value="/usr/bin/ffmpeg"),
            patch.object(svp_app, "check_svp_plugins", return_value=True),
        )
        for active_patch in self.common_patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    async def test_stop_during_blocking_prepare_never_starts_pipeline(self):
        # AC: @svp-single-player ac-disable-during-start
        prepare_started = threading.Event()
        release_prepare = threading.Event()
        stream = svp_app.SVPStream(video_path="/media/video.mp4")

        def delayed_prepare():
            prepare_started.set()
            release_prepare.wait(timeout=2)
            return Path("/tmp/unused-svp-script.vpy")

        with patch.object(stream, "_prepare", side_effect=delayed_prepare):
            starting = asyncio.create_task(stream.start())
            self.assertTrue(await asyncio.to_thread(prepare_started.wait, 1))
            await stream.stop_async()
            release_prepare.set()
            self.assertFalse(await starting)

        self.assertFalse(stream._running)
        self.assertIsNone(stream._task)

    async def test_stop_during_startup_prevents_late_stream_publication(self):
        # AC: @reliable-stream-transitions ac-stop-superseded-producer
        # AC: @svp-single-player ac-disable-during-start
        gate = asyncio.Event()
        stream = FakeStream(start_gate=gate)

        with patch.object(svp_app, "SVPStream", return_value=stream):
            play_task = asyncio.create_task(svp_app.play(self.request))
            await stream.started.wait()
            await svp_app.stop(self.stop_request)
            gate.set()

            with self.assertRaises(HTTPException) as raised:
                await play_task

        self.assertEqual(raised.exception.status_code, 409)
        self.assertTrue(stream.stopped)

    async def test_concurrent_starts_publish_only_the_latest_stream(self):
        # AC: @reliable-stream-transitions ac-final-source-owner
        # AC: @svp-single-player ac-final-transition-owner
        first_gate = asyncio.Event()
        first = FakeStream(start_gate=first_gate)
        second = FakeStream()
        streams = iter((first, second))

        with patch.object(svp_app, "SVPStream", side_effect=lambda **_kwargs: next(streams)):
            first_task = asyncio.create_task(svp_app.play(self.request))
            await first.started.wait()
            second_result = await svp_app.play(self.request)
            first_gate.set()
            with self.assertRaises(HTTPException) as raised:
                await first_task

        self.assertEqual(raised.exception.status_code, 409)
        self.assertTrue(first.stopped)
        self.assertEqual(second_result["stream_id"], second.stream_id)
        self.assertFalse(second.stopped)

    async def test_stale_external_start_does_not_stop_current_producers(self):
        # AC: @reliable-stream-transitions ac-stop-superseded-producer
        svp_app._legacy_stream_epoch = 5
        current = FakeStream()
        svp_app._active_streams[current.stream_id] = current
        stale_request = JsonRequest({
            **svp_app.PlayRequest(file_path="/media/video.mp4").model_dump(),
            "transition_id": 4,
        })

        with patch.object(svp_app.processing_session_service, "stop_all") as stop_sessions:
            with self.assertRaises(HTTPException) as raised:
                await svp_app.play(stale_request)

        self.assertEqual(raised.exception.status_code, 409)
        self.assertFalse(current.stopped)
        stop_sessions.assert_not_called()

    async def test_stale_external_stop_does_not_stop_current_producers(self):
        # AC: @reliable-stream-transitions ac-stop-superseded-producer
        svp_app._legacy_stream_epoch = 5
        current = FakeStream()
        svp_app._active_streams[current.stream_id] = current
        stale_request = JsonRequest({"transition_id": 4})

        with patch.object(svp_app.processing_session_service, "stop_all") as stop_sessions:
            with self.assertRaises(HTTPException) as raised:
                await svp_app.stop(stale_request)

        self.assertEqual(raised.exception.status_code, 409)
        self.assertFalse(current.stopped)
        stop_sessions.assert_not_called()

    async def test_unready_stream_is_stopped_instead_of_returned(self):
        # AC: @reliable-stream-transitions ac-stop-superseded-producer
        stream = FakeStream(ready=False)

        with patch.object(svp_app, "SVPStream", return_value=stream):
            with self.assertRaises(HTTPException) as raised:
                await svp_app.play(self.request)

        self.assertEqual(raised.exception.status_code, 504)
        self.assertTrue(stream.stopped)


if __name__ == "__main__":
    unittest.main()
