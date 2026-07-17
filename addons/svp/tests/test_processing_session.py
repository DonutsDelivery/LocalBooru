import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing_session import (  # noqa: E402
    SyntheticSegmentProducer,
    EventHistoryExpired,
    GenerationConflict,
    ProcessingSessionRegistry,
    SegmentRejected,
)
from session_protocol import (  # noqa: E402
    GenerationCommand,
    InitSegmentDescriptor,
    OpenSessionRequest,
    OpenSessionResponse,
    SeekSessionRequest,
    SegmentDescriptor,
    SessionEvent,
    SessionGraph,
    SessionMetadata,
)


def manager_graph():
    return SessionGraph(
        kind="manager_snapshot",
        revision=1,
        snapshot_path="/private/localbooru-svp-runtime.py",
        snapshot_sha256="a" * 64,
    )


class ProcessingSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = ProcessingSessionRegistry(Path(self.temp_dir.name))

    def tearDown(self):
        self.registry.stop_all()
        self.temp_dir.cleanup()

    def open_session(self, *, max_bytes=30, max_duration=3.0, with_init=True):
        request = OpenSessionRequest(
            protocol_version=1,
            generation=1,
            file_path="/media/fixture.mp4",
            start_position=10.0,
            graph=manager_graph(),
            max_buffer_bytes=max_bytes,
            max_buffer_duration=max_duration,
        )
        metadata = SessionMetadata(
            source_duration=120.0,
            width=1920,
            height=1080,
            source_fps=24.0,
            output_fps=48.0,
            mime_type='video/mp4; codecs="avc1.640028, mp4a.40.2"',
            initial_source_position=10.0,
            max_av_drift_ms=50.0,
        )
        session = self.registry.open(request, metadata)
        if with_init:
            init_path = session.output_dir / "initialization.mp4"
            init_path.write_bytes(b"i")
            session.publish_init(
                InitSegmentDescriptor(
                    generation=session.generation,
                    byte_length=1,
                    filename=init_path.name,
                )
            )
            session.acknowledge_init(session.generation)
        return session

    def publish_init(self, session, *, generation=None, size=1):
        generation = generation or session.generation
        path = session.output_dir / f"{generation}-init.mp4"
        path.write_bytes(b"i" * size)
        descriptor = InitSegmentDescriptor(
            generation=generation,
            byte_length=size,
            filename=path.name,
        )
        self.assertTrue(session.publish_init(descriptor))
        return descriptor

    def write_segment(self, session, *, generation, sequence, start, duration=1.0, size=10, drift=0.0):
        path = session.output_dir / f"{generation}-{sequence}.m4s"
        path.write_bytes(b"x" * size)
        return SegmentDescriptor(
            generation=generation,
            sequence=sequence,
            source_start=start,
            duration=duration,
            byte_length=size,
            filename=path.name,
            independent=True,
            av_drift_ms=drift,
        )

    # AC: @svp-bounded-stream ac-bounded-buffer
    def test_buffer_backpressure_and_acknowledgement_keep_output_bounded(self):
        session = self.open_session(max_bytes=20, max_duration=2.0)
        first = self.write_segment(session, generation=1, sequence=0, start=10.0)
        second = self.write_segment(session, generation=1, sequence=1, start=11.0)
        blocked = self.write_segment(session, generation=1, sequence=2, start=12.0)

        self.assertTrue(session.publish_segment(first))
        self.assertTrue(session.publish_segment(second))
        self.assertFalse(session.publish_segment(blocked))
        self.assertFalse((session.output_dir / blocked.filename).exists())
        self.assertEqual(session.buffered_bytes, 20)
        self.assertEqual(session.buffered_duration, 2.0)

        session.acknowledge(1, 0)
        replacement = self.write_segment(session, generation=1, sequence=2, start=12.0)
        self.assertTrue(session.publish_segment(replacement))
        self.assertLessEqual(session.buffered_bytes, 20)
        self.assertLessEqual(session.buffered_duration, 2.0)

    # AC: @svp-bounded-stream ac-bounded-buffer
    def test_synthetic_producer_stops_at_capacity_and_resumes_after_ack(self):
        session = self.open_session(max_bytes=8, max_duration=2.0)
        producer = SyntheticSegmentProducer(session, segment_duration=1.0, byte_length=4)

        self.assertTrue(producer.produce_one())
        self.assertTrue(producer.produce_one())
        self.assertFalse(producer.produce_one())
        self.assertEqual(session.buffered_bytes, 8)

        session.acknowledge(1, 0)
        self.assertTrue(producer.produce_one())
        self.assertEqual(session.buffered_bytes, 8)

    # AC: @svp-bounded-stream ac-transactional-seek
    def test_rapid_seeks_revoke_old_generations_and_accept_only_the_final_one(self):
        session = self.open_session()
        old = self.write_segment(session, generation=1, sequence=0, start=10.0)
        self.assertTrue(session.publish_segment(old))

        session.seek(expected_generation=1, generation=2, position=40.0)
        session.seek(expected_generation=2, generation=3, position=70.0)

        self.assertFalse((session.output_dir / old.filename).exists())
        stale = self.write_segment(session, generation=2, sequence=0, start=40.0)
        with self.assertRaises(GenerationConflict):
            session.publish_segment(stale)
        self.assertFalse((session.output_dir / stale.filename).exists())

        self.publish_init(session, generation=3)
        final = self.write_segment(session, generation=3, sequence=0, start=70.0)
        self.assertTrue(session.publish_segment(final))
        self.assertEqual(session.generation, 3)
        self.assertEqual(session.start_position, 70.0)

    # AC: @svp-bounded-stream ac-av-timeline
    def test_muxed_timeline_preserves_duration_and_rejects_timing_drift(self):
        session = self.open_session()
        self.assertEqual(session.metadata.source_duration, 120.0)

        first = self.write_segment(session, generation=1, sequence=0, start=10.0, drift=25.0)
        self.assertTrue(session.publish_segment(first))

        gap = self.write_segment(session, generation=1, sequence=1, start=20.0)
        with self.assertRaises(SegmentRejected):
            session.publish_segment(gap)
        self.assertFalse((session.output_dir / gap.filename).exists())

        regressed = self.write_segment(session, generation=1, sequence=1, start=9.5)
        with self.assertRaises(SegmentRejected):
            session.publish_segment(regressed)
        self.assertFalse((session.output_dir / regressed.filename).exists())

        excessive_drift = self.write_segment(
            session,
            generation=1,
            sequence=1,
            start=11.0,
            drift=51.0,
        )
        with self.assertRaises(SegmentRejected):
            session.publish_segment(excessive_drift)
        self.assertFalse((session.output_dir / excessive_drift.filename).exists())

    def test_protocol_rejects_unknown_versions_and_invalid_limits(self):
        command = GenerationCommand(protocol_version=1, generation=4)
        response = OpenSessionResponse(
            session_id="session-1",
            generation=command.generation,
            state="running",
        )
        seek = SeekSessionRequest(
            protocol_version=1,
            expected_generation=4,
            generation=5,
            position=60.0,
        )
        self.assertEqual(response.protocol_version, 1)
        self.assertEqual(seek.generation, 5)

        with self.assertRaises(ValidationError):
            SessionGraph(kind="deterministic_double")
        with self.assertRaises(ValidationError):
            SessionGraph(
                kind="manager_snapshot",
                revision=1,
                snapshot_path="/private/snapshot.vpy",
            )

        with self.assertRaises(ValidationError):
            OpenSessionRequest(
                protocol_version=2,
                generation=1,
                file_path="/media/fixture.mp4",
                graph=manager_graph(),
            )
        with self.assertRaises(ValidationError):
            SeekSessionRequest(
                protocol_version=1,
                expected_generation=4,
                generation=4,
                position=60.0,
            )
        with self.assertRaises(ValidationError):
            OpenSessionRequest(
                protocol_version=1,
                generation=1,
                file_path="/media/fixture.mp4",
                graph=manager_graph(),
                max_buffer_bytes=0,
            )

        with self.assertRaises(ValidationError):
            OpenSessionRequest(
                protocol_version=1,
                generation=1,
                file_path="/media/fixture.mp4",
                graph=manager_graph(),
                max_buffer_duration=float("inf"),
            )
        with self.assertRaises(ValidationError):
            SegmentDescriptor(
                generation=1,
                sequence=0,
                source_start=0.0,
                duration=1.0,
                byte_length=1,
                filename="0.m4s",
                independent=True,
                av_drift_ms=float("nan"),
            )
        with self.assertRaises(ValidationError):
            SessionEvent(
                event_sequence=0,
                session_id="session-1",
                generation=1,
                type="segment_ready",
            )
        with self.assertRaises(ValidationError):
            SessionEvent(
                event_sequence=0,
                session_id="session-1",
                generation=1,
                type="paused",
                message="unexpected",
            )

    def test_commands_are_generation_checked_and_stop_is_idempotent(self):
        session = self.open_session()
        with self.assertRaises(GenerationConflict):
            session.pause(expected_generation=2)
        session.pause(expected_generation=1)
        session.pause(expected_generation=1)
        session.resume(expected_generation=1)
        session.stop(expected_generation=1)
        session.stop(expected_generation=1)
        self.assertEqual(session.state.value, "stopped")
        self.assertEqual(session.buffered_bytes, 0)

    def test_init_segment_is_generation_owned_and_counted_toward_byte_limit(self):
        session = self.open_session(max_bytes=12, with_init=False)
        path = session.output_dir / "1-init.mp4"
        path.write_bytes(b"init")
        descriptor = InitSegmentDescriptor(
            generation=1,
            byte_length=4,
            filename=path.name,
        )

        self.assertTrue(session.publish_init(descriptor))
        self.assertEqual(session.buffered_bytes, 4)
        self.assertEqual(session.events_after(-1)[-2].type, "init_ready")

        session.seek(expected_generation=1, generation=2, position=20.0)
        self.assertFalse(path.exists())
        self.assertEqual(session.buffered_bytes, 0)

    def test_media_requires_exactly_one_init_segment_per_generation(self):
        session = self.open_session(with_init=False)
        media = self.write_segment(session, generation=1, sequence=0, start=10.0)
        with self.assertRaises(SegmentRejected):
            session.publish_segment(media)

        self.publish_init(session)
        duplicate_path = session.output_dir / "duplicate-init.mp4"
        duplicate_path.write_bytes(b"i")
        duplicate = InitSegmentDescriptor(
            generation=1,
            byte_length=1,
            filename=duplicate_path.name,
        )
        with self.assertRaises(SegmentRejected):
            session.publish_init(duplicate)
        self.assertFalse(duplicate_path.exists())

    def test_terminal_error_reports_last_acknowledged_position(self):
        session = self.open_session()
        first = self.write_segment(session, generation=1, sequence=0, start=10.0)
        second = self.write_segment(session, generation=1, sequence=1, start=11.0)
        self.assertTrue(session.publish_segment(first))
        self.assertTrue(session.publish_segment(second))
        self.assertTrue(session.acknowledge(1, 0))

        session.fail(1, code="processor", message="failed")
        terminal = session.events_after(-1)[-1]
        self.assertEqual(terminal.type, "terminal_error")
        self.assertEqual(terminal.last_safe_position, 11.0)

    def test_nonseekable_session_rejects_seek(self):
        request = OpenSessionRequest(
            protocol_version=1,
            generation=1,
            file_path="/media/fixture.mp4",
            graph=manager_graph(),
        )
        metadata = SessionMetadata(
            source_duration=120.0,
            width=1920,
            height=1080,
            source_fps=24.0,
            output_fps=48.0,
            mime_type='video/mp4; codecs="avc1.640028, mp4a.40.2"',
            initial_source_position=0.0,
            seekable=False,
        )
        session = self.registry.open(request, metadata)
        with self.assertRaises(SegmentRejected):
            session.seek(expected_generation=1, generation=2, position=5.0)

    def test_publish_validates_file_size_without_deleting_owned_segment(self):
        session = self.open_session()
        accepted = self.write_segment(session, generation=1, sequence=0, start=10.0)
        self.assertTrue(session.publish_segment(accepted))

        collision = SegmentDescriptor(
            generation=2,
            sequence=0,
            source_start=20.0,
            duration=1.0,
            byte_length=accepted.byte_length,
            filename=accepted.filename,
            independent=True,
        )
        with self.assertRaises(GenerationConflict):
            session.publish_segment(collision)
        self.assertTrue((session.output_dir / accepted.filename).exists())

        wrong_size = self.write_segment(session, generation=1, sequence=1, start=11.0, size=10)
        wrong_size = wrong_size.model_copy(update={"byte_length": 1})
        with self.assertRaises(SegmentRejected):
            session.publish_segment(wrong_size)
        self.assertFalse((session.output_dir / wrong_size.filename).exists())

    def test_terminal_states_cannot_be_rewritten_or_resurrected(self):
        ended = self.open_session()
        ended.end(expected_generation=1)
        with self.assertRaises(SegmentRejected):
            ended.seek(expected_generation=1, generation=2, position=20.0)
        with self.assertRaises(SegmentRejected):
            ended.fail(expected_generation=1, code="late", message="late failure")

        failed = self.open_session()
        failed.fail(expected_generation=1, code="processor", message="failed")
        with self.assertRaises(SegmentRejected):
            failed.end(expected_generation=1)
        with self.assertRaises(SegmentRejected):
            failed.seek(expected_generation=1, generation=2, position=20.0)

        stopped = self.open_session()
        stopped.stop(expected_generation=1)
        with self.assertRaises(SegmentRejected):
            stopped.seek(expected_generation=1, generation=2, position=20.0)

    def test_event_replay_is_bounded_and_reports_expired_cursors(self):
        registry = ProcessingSessionRegistry(Path(self.temp_dir.name) / "events", event_history_limit=4)
        request = OpenSessionRequest(
            protocol_version=1,
            generation=1,
            file_path="/media/fixture.mp4",
            graph=manager_graph(),
        )
        metadata = SessionMetadata(
            source_duration=120.0,
            width=1920,
            height=1080,
            source_fps=24.0,
            output_fps=48.0,
            mime_type='video/mp4; codecs="avc1.640028, mp4a.40.2"',
            initial_source_position=0.0,
        )
        session = registry.open(request, metadata)
        session.pause(1)
        session.resume(1)
        session.pause(1)
        session.resume(1)

        self.assertLessEqual(len(session.events_after(0)), 4)
        with self.assertRaises(EventHistoryExpired):
            session.events_after(-1)
        registry.stop_all()


if __name__ == "__main__":
    unittest.main()
