#!/usr/bin/env python3
"""Native child-loader boundary tests using only owned, synthetic bookmark stores."""
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parent.parent / 'src-tauri/macos/media-folder-grants.c'
VARIABLE = 'DMC_MEDIA_BOOKMARK_DIRECTORY'


@unittest.skipUnless(platform.system() == 'Darwin', 'Requires the native Apple SDK')
class ChildGrantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('DMC_MAC_BUILD_LOCK_HELD') != '1':
            raise RuntimeError('Run these compiler tests under scripts/macos-build-lock.py')
        cls.temporary = tempfile.TemporaryDirectory(prefix='dmc child grant fixture ')
        cls.root = Path(cls.temporary.name)
        main = cls.root / 'main.c'
        main.write_text('#include <stdlib.h>\nint main(void) { return getenv("' + VARIABLE + '") ? 99 : 0; }\n')
        for arch in ('arm64', 'x86_64'):
            subprocess.run(['clang', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
                            '-arch', arch, '-mmacosx-version-min=11.0', SOURCE, main,
                            '-framework', 'CoreFoundation', '-o', cls.root / arch], check=True)
        cls.binary = cls.root / platform.machine()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self, directory=None):
        env = os.environ.copy(); env.pop(VARIABLE, None)
        if directory is not None: env[VARIABLE] = str(directory)
        result = subprocess.run([self.binary], env=env, text=True, capture_output=True, timeout=20)
        if result.returncode == 78:
            self.assertEqual(result.stderr, 'Media folder grant initialization failed.\n')
        return result

    def store(self):
        temporary = tempfile.TemporaryDirectory(prefix='dmc child bookmark store ')
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_standalone_and_empty_transfer(self):
        self.assertEqual(self.execute().returncode, 0)
        self.assertEqual(self.execute(self.store()).returncode, 0)

    def test_unresolvable_bookmark_is_skipped_without_being_modified(self):
        root = self.store(); bookmark = root / 'opaque-1.bookmark'
        bookmark.write_bytes(b'deliberately invalid synthetic bookmark')
        self.assertEqual(self.execute(root).returncode, 0)
        self.assertEqual(bookmark.read_bytes(), b'deliberately invalid synthetic bookmark')

    def test_missing_and_symlink_directory_fail_closed(self):
        root = self.store()
        self.assertEqual(self.execute(root / 'missing').returncode, 78)
        alias = root / 'alias'; alias.symlink_to(root, target_is_directory=True)
        self.assertEqual(self.execute(str(alias) + '/').returncode, 78)

    def test_symlink_and_fifo_entries_fail_without_blocking(self):
        root = self.store(); original = root / 'original'; original.write_bytes(b'keep')
        bookmark = root / 'opaque.bookmark'; bookmark.symlink_to(original)
        self.assertEqual(self.execute(root).returncode, 78)
        self.assertEqual(original.read_bytes(), b'keep')
        bookmark.unlink(); os.mkfifo(bookmark)
        self.assertEqual(self.execute(root).returncode, 78)

    def test_entry_and_store_limits_fail_closed(self):
        root = self.store(); bookmark = root / 'opaque.bookmark'
        bookmark.touch()
        self.assertEqual(self.execute(root).returncode, 78)
        with bookmark.open('wb') as output: output.truncate(1024 * 1024 + 1)
        self.assertEqual(self.execute(root).returncode, 78)
        bookmark.unlink()
        for number in range(1025): (root / f'{number}.bookmark').write_bytes(b'invalid')
        self.assertEqual(self.execute(root).returncode, 78)


if __name__ == '__main__': unittest.main()
