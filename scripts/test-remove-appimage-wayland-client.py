#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('wayland', Path(__file__).with_name('remove-appimage-wayland-client.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestPolicy(unittest.TestCase):
    def test_removal_and_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'AppDir'
            lib = root / 'usr/lib'
            lib.mkdir(parents=True)
            (root / 'usr/share/doc/localbooru').mkdir(parents=True)
            (root / 'AppRun').touch()
            (lib / 'libwayland-client.so.0.3').write_text('client')
            (lib / 'libwayland-client.so.0').symlink_to('libwayland-client.so.0.3')
            (lib / 'libwayland-client.so').symlink_to('libwayland-client.so.0')
            other = lib / 'libwayland-server.so.0'
            other.write_text('keep')
            self.assertEqual(len(module.apply(root)['removed']), 3)
            self.assertEqual(other.read_text(), 'keep')
            module.apply(root, True)
            (lib / 'libwayland-client.so.0').touch()
            with self.assertRaises(ValueError): module.apply(root, True)

    def test_preflight_refuses_redirect_and_invalid_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'AppDir'
            lib = root / 'usr/lib'
            lib.mkdir(parents=True)
            (root / 'usr/share/doc/localbooru').mkdir(parents=True)
            (root / 'AppRun').touch()
            good = lib / 'libwayland-client.so.0'
            good.touch()
            bad = lib / 'libwayland-client.so.1'
            bad.symlink_to('/etc/passwd')
            with self.assertRaises(ValueError): module.apply(root)
            self.assertTrue(good.exists())
            bad.unlink()
            bad.mkdir()
            with self.assertRaises(ValueError): module.apply(root)
            self.assertTrue(good.exists())
            with self.assertRaises(ValueError): module.apply(Path('relative'))
            bad.rmdir()
            doc = root / 'usr/share/doc/localbooru'
            doc.rmdir()
            doc.symlink_to(Path(tmp))
            with self.assertRaises(ValueError): module.apply(root)
            self.assertTrue(good.exists())


if __name__ == '__main__': unittest.main()
