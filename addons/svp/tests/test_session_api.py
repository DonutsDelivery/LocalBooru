import sys
import tempfile
import threading
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing_session import SyntheticSegmentProducer  # noqa: E402
from session_api import (  # noqa: E402
    ProcessingSessionService,
    ProcessorUnavailable,
    create_session_router,
)
from session_protocol import OpenSessionRequest, SessionMetadata  # noqa: E402


class TrackingProcessor:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def client(self, service):
        app = FastAPI()
        app.include_router(create_session_router(service))
        return TestClient(app)

    @staticmethod
    def metadata(request):
        return SessionMetadata(
            source_duration=120.0,
            width=1920,
            height=1080,
            source_fps=24.0,
            output_fps=48.0,
            mime_type='video/mp4; codecs="avc1.640028, mp4a.40.2"',
            initial_source_position=request.start_position,
            max_av_drift_ms=50.0,
        )

    @staticmethod
    def start_fake(session):
        producer = SyntheticSegmentProducer(session, segment_duration=1.0, byte_length=4)
        producer.produce_init()
        producer.produce_one()

    def open_body(self):
        return {
            "protocol_version": 1,
            "generation": 1,
            "file_path": "/media/fixture.mp4",
            "start_position": 10.0,
            "graph": {
                "kind": "manager_snapshot",
                "revision": 1,
                "snapshot_path": "/private/localbooru-svp-runtime.py",
                "snapshot_sha256": "a" * 64,
            },
            "max_buffer_bytes": 64,
            "max_buffer_duration": 3.0,
        }

    def test_production_router_refuses_sessions_without_a_real_processor(self):
        service = ProcessingSessionService(Path(self.temp_dir.name))
        with self.client(service) as client:
            response = client.post("/svp/sessions", json=self.open_body())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "processor_unavailable")

    # AC: @svp-bounded-stream ac-bounded-buffer
    # AC: @svp-bounded-stream ac-transactional-seek
    def test_session_lifecycle_exposes_owned_segments_and_rejects_stale_commands(self):
        service = ProcessingSessionService(
            Path(self.temp_dir.name),
            metadata_provider=self.metadata,
            session_started=self.start_fake,
        )
        with self.client(service) as client:
            opened = client.post("/svp/sessions", json=self.open_body())
            self.assertEqual(opened.status_code, 200)
            session_id = opened.json()["session_id"]

            event_response = client.get(f"/svp/sessions/{session_id}/events", params={"after": -1})
            self.assertEqual(event_response.status_code, 200)
            events = event_response.json()["events"]
            init = next(event["init_segment"] for event in events if event["type"] == "init_ready")
            media = next(event["segment"] for event in events if event["type"] == "segment_ready")

            init_response = client.get(
                f"/svp/sessions/{session_id}/segments/1/{init['filename']}"
            )
            media_response = client.get(
                f"/svp/sessions/{session_id}/segments/1/{media['filename']}"
            )
            self.assertEqual(init_response.content, b"synthetic-init")
            self.assertEqual(media_response.content, bytes(4))

            stale = client.post(
                f"/svp/sessions/{session_id}/pause",
                json={"protocol_version": 1, "generation": 2},
            )
            self.assertEqual(stale.status_code, 409)
            self.assertEqual(stale.json()["detail"]["code"], "generation_conflict")

            seek = client.post(
                f"/svp/sessions/{session_id}/seek",
                json={
                    "protocol_version": 1,
                    "expected_generation": 1,
                    "generation": 2,
                    "position": 40.0,
                },
            )
            self.assertEqual(seek.status_code, 200)
            self.assertEqual(seek.json()["generation"], 2)

            revoked = client.get(
                f"/svp/sessions/{session_id}/segments/1/{media['filename']}"
            )
            self.assertEqual(revoked.status_code, 409)

            stopped = client.delete(
                f"/svp/sessions/{session_id}",
                params={"expected_generation": 2, "protocol_version": 1},
            )
            self.assertEqual(stopped.status_code, 200)
            self.assertEqual(stopped.json()["state"], "stopped")
            self.assertEqual(client.get(f"/svp/sessions/{session_id}").status_code, 404)

    # AC: @reliable-stream-transitions ac-stop-superseded-producer
    def test_stop_all_invalidates_an_open_still_loading_metadata(self):
        metadata_started = threading.Event()
        release_metadata = threading.Event()
        result = {}

        def delayed_metadata(request):
            metadata_started.set()
            release_metadata.wait(timeout=2)
            return self.metadata(request)

        service = ProcessingSessionService(
            Path(self.temp_dir.name),
            metadata_provider=delayed_metadata,
            session_started=lambda _session: TrackingProcessor(),
        )
        request = OpenSessionRequest.model_validate(self.open_body())

        def open_session():
            try:
                result["session"] = service.open(request)
            except Exception as error:
                result["error"] = error

        worker = threading.Thread(target=open_session)
        worker.start()
        self.assertTrue(metadata_started.wait(timeout=1))
        service.stop_all()
        release_metadata.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertIsInstance(result.get("error"), ProcessorUnavailable)
        self.assertNotIn("session", result)

    # AC: @svp-bounded-stream ac-transactional-seek
    # AC: @reliable-stream-transitions ac-stop-superseded-producer
    def test_stale_seek_and_stop_preserve_the_current_processor(self):
        processors = []

        def start_processor(_session):
            processor = TrackingProcessor()
            processors.append(processor)
            return processor

        service = ProcessingSessionService(
            Path(self.temp_dir.name),
            metadata_provider=self.metadata,
            session_started=start_processor,
        )
        with self.client(service) as client:
            session_id = client.post("/svp/sessions", json=self.open_body()).json()["session_id"]
            seek = client.post(
                f"/svp/sessions/{session_id}/seek",
                json={
                    "protocol_version": 1,
                    "expected_generation": 1,
                    "generation": 2,
                    "position": 40.0,
                },
            )
            self.assertEqual(seek.status_code, 200)
            self.assertTrue(processors[0].stopped)
            self.assertFalse(processors[1].stopped)

            stale_seek = client.post(
                f"/svp/sessions/{session_id}/seek",
                json={
                    "protocol_version": 1,
                    "expected_generation": 1,
                    "generation": 3,
                    "position": 60.0,
                },
            )
            self.assertEqual(stale_seek.status_code, 409)
            stale_stop = client.delete(
                f"/svp/sessions/{session_id}",
                params={"expected_generation": 1, "protocol_version": 1},
            )
            self.assertEqual(stale_stop.status_code, 409)
            self.assertFalse(processors[1].stopped)
            self.assertEqual(client.get(f"/svp/sessions/{session_id}").status_code, 200)

        service.stop_all()
        self.assertTrue(processors[1].stopped)
        self.assertIsNone(service.registry.get(session_id))

    def test_segment_acknowledgement_releases_retained_capacity(self):
        service = ProcessingSessionService(
            Path(self.temp_dir.name),
            metadata_provider=self.metadata,
            session_started=self.start_fake,
        )
        with self.client(service) as client:
            opened = client.post("/svp/sessions", json=self.open_body()).json()
            session_id = opened["session_id"]
            events = client.get(
                f"/svp/sessions/{session_id}/events", params={"after": -1}
            ).json()["events"]
            media = next(event["segment"] for event in events if event["type"] == "segment_ready")

            acknowledged = client.request(
                "DELETE",
                f"/svp/sessions/{session_id}/segments",
                json={"protocol_version": 1, "generation": 1, "sequence": media["sequence"]},
            )
            self.assertEqual(acknowledged.status_code, 200)
            self.assertLess(acknowledged.json()["buffered_bytes"], 4 + len(b"synthetic-init"))
            self.assertEqual(
                client.get(
                    f"/svp/sessions/{session_id}/segments/1/{media['filename']}"
                ).status_code,
                409,
            )
        service.registry.stop_all()


if __name__ == "__main__":
    unittest.main()
