#!/usr/bin/env python3
"""Synthetic ownership and cleanup checks; no real app container is used."""
import importlib.util
from pathlib import Path
import plistlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('roots', Path(__file__).with_name('isolated-macos-root.py'))
roots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roots)


class RootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dmc-container-root-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.home = self.base / 'home'
        self.app = self.base / 'Lite.app'
        (self.app / 'Contents').mkdir(parents=True)
        with (self.app / 'Contents/Info.plist').open('wb') as stream:
            plistlib.dump({'CFBundleIdentifier': 'com.localbooru.app.lite'}, stream)
        self.parent = self.home / 'Library/Containers/com.localbooru.app.lite/Data/Library/Application Support'
        self.parent.mkdir(parents=True)
        self.parent.chmod(0o700)

    def test_owned_cleanup_preserves_siblings(self):
        sibling = self.parent / 'synthetic-existing-asset'
        sibling.write_text('preserve')
        root, identity = roots.create(self.app, self.parent, self.home)
        (root / 'synthetic-library').mkdir()
        (root / 'synthetic-library/fixture').write_text('owned')
        roots.cleanup(self.app, self.parent, self.home, root, identity)
        self.assertFalse(root.exists())
        self.assertEqual(sibling.read_text(), 'preserve')

    def test_identity_change_refuses_cleanup(self):
        root, _ = roots.create(self.app, self.parent, self.home)
        with self.assertRaises(ValueError):
            roots.cleanup(self.app, self.parent, self.home, root, '0:0')
        self.assertTrue(root.exists())

    def test_shared_permissions_refuse_cleanup(self):
        root, identity = roots.create(self.app, self.parent, self.home)
        root.chmod(0o755)
        with self.assertRaises(ValueError):
            roots.cleanup(self.app, self.parent, self.home, root, identity)

    def test_foreign_parent_and_wrong_edition_are_rejected(self):
        with self.assertRaises(ValueError):
            roots.create(self.app, self.base, self.home)
        with (self.app / 'Contents/Info.plist').open('wb') as stream:
            plistlib.dump({'CFBundleIdentifier': 'com.localbooru.app'}, stream)
        with self.assertRaises(ValueError):
            roots.create(self.app, self.parent, self.home)

    def test_parent_symlink_and_shared_write_are_rejected(self):
        target = self.base / 'owned-target'
        target.mkdir()
        self.parent.rmdir()
        self.parent.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            roots.create(self.app, self.parent, self.home)
        self.parent.unlink()
        self.parent.mkdir()
        self.parent.chmod(0o777)
        with self.assertRaises(ValueError):
            roots.create(self.app, self.parent, self.home)

    def test_root_symlink_and_unrecognized_name_are_preserved(self):
        target = self.base / 'owned-target'
        target.mkdir()
        linked = self.parent / 'localbooru-isolated.link'
        linked.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            roots.cleanup(self.app, self.parent, self.home, linked, '0:0')
        with self.assertRaises(ValueError):
            roots.cleanup(self.app, self.parent, self.home, self.parent, '0:0')
        self.assertTrue(target.exists())


if __name__ == '__main__':
    unittest.main()
