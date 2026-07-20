import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manager_graph import (  # noqa: E402
    ManagerGraphUnavailable,
    generate_manager_snapshot_stdin_script,
    load_manager_snapshot,
    trusted_snapshot_root,
)
from session_protocol import SessionGraph  # noqa: E402


class ManagerGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "snapshots"
        self.root.mkdir(mode=0o700)
        self.previous_root = os.environ.get("LOCALBOORU_SVP_SNAPSHOT_ROOT")
        os.environ["LOCALBOORU_SVP_SNAPSHOT_ROOT"] = str(self.root)

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop("LOCALBOORU_SVP_SNAPSHOT_ROOT", None)
        else:
            os.environ["LOCALBOORU_SVP_SNAPSHOT_ROOT"] = self.previous_root
        self.temp_dir.cleanup()

    def graph(self, path, contents):
        return SessionGraph(
            kind="manager_snapshot",
            revision=3,
            snapshot_path=str(path),
            snapshot_sha256=hashlib.sha256(contents).hexdigest(),
        )

    # AC: @desktop-svp-video ac-manager-owned-graph
    def test_loads_exact_bytes_from_private_trusted_root(self):
        contents = b"# coding: latin-1\nvideo_in.set_output()\n# \xff\n"
        path = self.root / "graph.vpy"
        path.write_bytes(contents)

        captured = load_manager_snapshot(self.graph(path, contents), trusted_snapshot_root())

        self.assertEqual(captured.contents, contents)
        self.assertEqual(captured.revision, 3)

    # AC: @desktop-svp-video ac-unavailable-fallback
    def test_rejects_hash_mismatch_outside_paths_and_symlinks(self):
        contents = b"video_in.set_output()\n"
        path = self.root / "graph.vpy"
        path.write_bytes(contents)
        wrong_hash = self.graph(path, b"different")
        with self.assertRaises(ManagerGraphUnavailable):
            load_manager_snapshot(wrong_hash, trusted_snapshot_root())

        outside = Path(self.temp_dir.name) / "outside.vpy"
        outside.write_bytes(contents)
        with self.assertRaises(ManagerGraphUnavailable):
            load_manager_snapshot(self.graph(outside, contents), trusted_snapshot_root())

        link = self.root / "link.vpy"
        link.symlink_to(outside)
        with self.assertRaises(ManagerGraphUnavailable):
            load_manager_snapshot(self.graph(link, contents), trusted_snapshot_root())

    @unittest.skipIf(os.name == "nt", "Unix ownership and mode checks")
    def test_rejects_non_private_snapshot_root(self):
        self.root.chmod(0o755)
        with self.assertRaises(ManagerGraphUnavailable):
            trusted_snapshot_root()

    # AC: @desktop-svp-video ac-manager-owned-graph
    def test_bootstrap_embeds_captured_bytes_and_manager_video_input_contract(self):
        original = b"assert video_in_dw == 160\nvideo_in.set_output()\n"
        path = self.root / "graph.vpy"
        path.write_bytes(original)
        captured = load_manager_snapshot(self.graph(path, original), trusted_snapshot_root())
        path.write_bytes(b"raise RuntimeError('mutated')\n")

        script = generate_manager_snapshot_stdin_script(captured, 160, 90, 24000, 1001, 36)

        self.assertIn("video_in = core.std.ModifyFrame", script)
        self.assertIn("video_in_dw = WIDTH", script)
        self.assertIn("video_in_dh = HEIGHT", script)
        self.assertIn("container_fps = FPS_NUM / FPS_DEN", script)
        self.assertIn("display_fps = 0.0", script)
        self.assertIn("display_res = [WIDTH, HEIGHT]", script)
        self.assertIn('user_data = ""', script)
        self.assertNotIn("smooth.set_output()", script)
        self.assertNotIn("mutated", script)


if __name__ == "__main__":
    unittest.main()
