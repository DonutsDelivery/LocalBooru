#!/usr/bin/env python3
"""Exercise normalization and, when supplied, the real linuxdeploy parser."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "prepare_desktop", Path(__file__).with_name("prepare-appimage-desktop.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = argparse.ArgumentParser()
parser.add_argument("--wrapped-launcher", type=Path)
options, remaining = parser.parse_known_args()


class DesktopTests(unittest.TestCase):
    def test_main_and_action_preserve_arguments_newlines_and_mode(self):
        with tempfile.TemporaryDirectory(prefix="dmc-desktop-") as folder:
            path = Path(folder) / "fixture.desktop"
            original = (b'[Desktop Entry]\r\nExec="donutmediacenter" %U\r\n'
                        b'[Desktop Action NewWindow]\r\n'
                        b'Exec="donutmediacenter" --label="two words"\r\n')
            path.write_bytes(original)
            path.chmod(0o640)
            module.normalize(path)
            expected = original.replace(b'Exec="donutmediacenter"', b'Exec=donutmediacenter')
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            module.normalize(path)
            self.assertEqual(path.read_bytes(), expected)

    def test_check_rejects_quotes_without_mutation(self):
        with tempfile.TemporaryDirectory(prefix="dmc-desktop-") as folder:
            path = Path(folder) / "fixture.desktop"
            original = b'Exec="donutmediacenter" %U\n'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                module.normalize(path, check=True)
            self.assertEqual(path.read_bytes(), original)

    def test_unquoted_check_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="dmc-desktop-") as folder:
            path = Path(folder) / "fixture.desktop"
            original = b'Exec=donutmediacenter %U\n'
            path.write_bytes(original)
            module.normalize(path, check=True)
            self.assertEqual(path.read_bytes(), original)

    def test_unknown_executable_refuses_entire_update(self):
        for value in ('other', 'donutmediacenter-other', '"donutmediacenter"suffix'):
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix="dmc-desktop-") as folder:
                path = Path(folder) / "fixture.desktop"
                original = ('Exec="donutmediacenter" %U\n[Desktop Action NewWindow]\nExec=' + value + '\n').encode()
                path.write_bytes(original)
                with self.assertRaises(ValueError):
                    module.normalize(path)
                self.assertEqual(path.read_bytes(), original)

    def test_missing_exec_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory(prefix="dmc-desktop-") as folder:
            path = Path(folder) / "fixture.desktop"
            path.write_text("[Desktop Entry]\nName=Fixture\n")
            with self.assertRaises(ValueError):
                module.normalize(path)
            link = Path(folder) / "link.desktop"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                module.normalize(link)

    @unittest.skipUnless(options.wrapped_launcher, "Pass the packaged AppRun.wrapped for the real parser regression")
    def test_real_wrapped_launcher_fails_before_and_runs_stub_after(self):
        with tempfile.TemporaryDirectory(prefix="dmc-desktop-launcher-") as folder:
            root = Path(folder)
            launcher = root / "AppRun"
            shutil.copyfile(options.wrapped_launcher, launcher)
            launcher.chmod(0o755)
            binary = root / "usr/bin/donutmediacenter"
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\nprintf "owned-fixture\\n"\nprintf "<%s>\\n" "$@"\n')
            binary.chmod(0o755)
            desktop = root / "fixture.desktop"
            desktop.write_text('[Desktop Entry]\nType=Application\nName=Fixture\nExec="donutmediacenter" %U\n')
            env = dict(os.environ, APPDIR=str(root), HOME=str(root / "home"),
                       PATH=str(binary.parent) + ":/usr/bin:/bin")
            for key in ("home", "config", "data", "state", "cache", "tmp", "library"):
                (root / key).mkdir(mode=0o700)
            env.update(XDG_CONFIG_HOME=str(root / "config"), XDG_DATA_HOME=str(root / "data"),
                       XDG_STATE_HOME=str(root / "state"), XDG_CACHE_HOME=str(root / "cache"),
                       TMPDIR=str(root / "tmp"), LOCALBOORU_PORTABLE_DATA=str(root / "library"),
                       LOCALBOORU_DISABLE_SINGLE_INSTANCE="1", LOCALBOORU_ISOLATED_INSTANCE_ID="synthetic-parser-fixture",
                       LOCALBOORU_ISOLATED_TEMP_DIR=str(root / "tmp"))
            before = subprocess.run([str(launcher), "argument with spaces"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(before.returncode, 0, before.stdout)
            self.assertNotIn("owned-fixture", before.stdout)
            module.normalize(desktop)
            after = subprocess.run([str(launcher), "argument with spaces"], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=10)
            self.assertEqual(after.returncode, 0, after.stderr)
            self.assertIn("owned-fixture\n", after.stdout)
            self.assertIn("<argument with spaces>", after.stdout)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
