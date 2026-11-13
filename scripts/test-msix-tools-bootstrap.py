#!/usr/bin/env python3
"""Validate native-tool bootstrap boundaries without compiling or downloading."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bootstrap', Path(__file__).with_name('build-msix-tools.py'))
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_wrong_archive_cannot_reach_extraction_or_compilation(self):
        with tempfile.TemporaryDirectory(prefix='dmc msix pin fixture ') as temporary:
            root = Path(temporary); archive = root / 'tampered.tar.gz'
            archive.write_bytes(b'synthetic altered source')
            with patch.object(bootstrap.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    bootstrap.build(root / 'new owned cache', archive)
                run.assert_not_called()
            self.assertFalse((root / 'new owned cache' / ('msix-packaging-' + bootstrap.COMMIT)).exists())

    def test_unknown_nonempty_cache_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix='dmc msix unrelated fixture ') as temporary:
            root = Path(temporary); existing = root / 'keep.txt'
            existing.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'unrecognized'):
                bootstrap.prepare_cache(root)
            self.assertEqual(existing.read_bytes(), b'preserve')

    def test_symlink_component_is_rejected_and_recognized_cache_can_reopen(self):
        with tempfile.TemporaryDirectory(prefix='dmc msix cache fixture ') as temporary:
            root = Path(temporary); cache = root / 'cache with spaces'
            bootstrap.prepare_cache(cache)
            self.assertEqual(bootstrap.prepare_cache(cache), cache)
            link = root / 'alias'; link.symlink_to(cache, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                bootstrap.prepare_cache(link / 'child')


if __name__ == '__main__': unittest.main()
