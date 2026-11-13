#!/usr/bin/env python3
"""Exercise lock ownership with synthetic commands; no desktop app is launched."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('macos-build-lock.py')


class LockTests(unittest.TestCase):
    def command(self, lock, code, *options):
        return [sys.executable, str(SCRIPT), '--lock', str(lock), *options, '--', sys.executable, '-c', code]

    def test_foreign_owner_is_preserved_without_starting_work(self):
        with tempfile.TemporaryDirectory(prefix='dmc mac lock fixture ') as tmp:
            lock = Path(tmp) / 'build.lock'; marker = Path(tmp) / 'started'
            with lock.open('a+') as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                result = subprocess.run(self.command(lock, 'from pathlib import Path; Path('+repr(str(marker))+').touch()'), capture_output=True, text=True)
                self.assertEqual(result.returncode, 75)
                self.assertFalse(marker.exists())

    def test_child_status_and_one_job_environment_propagate_then_unlock(self):
        with tempfile.TemporaryDirectory(prefix='dmc mac lock fixture ') as tmp:
            lock = Path(tmp) / 'build.lock'
            result = subprocess.run(self.command(lock, 'import os,sys; print(os.environ["DMC_MAC_BUILD_LOCK_HELD"],os.environ["CARGO_BUILD_JOBS"]); sys.exit(7)'), capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(result.stdout.strip(), '1 1')
            with lock.open('a+') as held: fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_owned_timeout_cleans_up_and_releases_lock(self):
        with tempfile.TemporaryDirectory(prefix='dmc mac lock fixture ') as tmp:
            lock = Path(tmp) / 'build.lock'
            result = subprocess.run(self.command(lock, 'import time; time.sleep(20)', '--timeout', '1'), timeout=15)
            self.assertEqual(result.returncode, 124)
            with lock.open('a+') as held: fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_symlink_lock_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dmc mac lock fixture ') as tmp:
            original = Path(tmp) / 'original'; original.touch()
            lock = Path(tmp) / 'alias'; lock.symlink_to(original)
            result = subprocess.run(self.command(lock, 'raise Exception("must not run")'), capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__': unittest.main()
