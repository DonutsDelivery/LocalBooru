#!/usr/bin/env python3
"""Synthetic build-boundary and payload validation tests; no compiler or app starts."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('media_build', 'build-macos-media.py')
packager = load('media_package', 'package-macos-media.py')


class BuildBoundaryTests(unittest.TestCase):
    def test_unknown_cache_and_symlink_preserved(self):
        with tempfile.TemporaryDirectory(prefix='dmc media fixture ') as tmp:
            root = Path(tmp); original = root / 'original'; original.mkdir()
            keep = original / 'keep'; keep.write_text('owned by someone else')
            with self.assertRaises(ValueError): builder.owned_directory(original, {'kind': 'test'})
            alias = root / 'alias'; alias.symlink_to(original, target_is_directory=True)
            with self.assertRaises(ValueError): builder.owned_directory(alias, {'kind': 'test'})
            self.assertEqual(keep.read_text(), 'owned by someone else')

    def test_owned_cache_can_reopen_only_same_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'cache'
            builder.owned_directory(root, {'recipe': 'one'})
            builder.owned_directory(root, {'recipe': 'one'})
            with self.assertRaises(ValueError): builder.owned_directory(root, {'recipe': 'two'})

    def test_altered_download_is_not_accepted_or_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'source.tar.xz'; archive.write_bytes(b'changed')
            with patch.object(builder.urllib.request, 'urlopen') as fetch:
                with self.assertRaises(ValueError):
                    builder.download({'url': 'https://example.invalid/source.tar.xz', 'sha256': '0' * 64}, root)
                fetch.assert_not_called()
            self.assertEqual(archive.read_bytes(), b'changed')

    def test_unsafe_archives_cannot_escape(self):
        for member_name, kind in (('../escape', tarfile.REGTYPE), ('source/link', tarfile.SYMTYPE)):
            with self.subTest(member=member_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); archive = root / 'input.tar'; sources = root / 'sources'; sources.mkdir()
                with tarfile.open(archive, 'w') as target:
                    member = tarfile.TarInfo(member_name); member.type = kind; member.linkname = '/escape'
                    target.addfile(member, io.BytesIO())
                with self.assertRaises(ValueError):
                    builder.extract(archive, {'root': 'source', 'sha256': builder.digest(archive)}, sources)
                self.assertFalse((root / 'escape').exists())

    def test_source_extraction_is_pinned_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'input.tar'; sources = root / 'sources'; sources.mkdir()
            with tarfile.open(archive, 'w') as target:
                member = tarfile.TarInfo('source/COPYING'); member.size = 4
                target.addfile(member, io.BytesIO(b'test'))
            pin = {'root': 'source', 'sha256': builder.digest(archive)}
            path = builder.extract(archive, pin, sources)
            self.assertEqual(builder.extract(archive, pin, sources), path)
            self.assertEqual((path / 'COPYING').read_text(), 'test')

    def test_no_build_without_native_mac_lock(self):
        with patch.object(builder.platform, 'system', return_value='Darwin'), patch.dict(os.environ, {'DMC_MAC_BUILD_LOCK_HELD': '0'}):
            with patch.object(builder, 'run') as run:
                with self.assertRaises(ValueError): builder.build(Path('/not-read'), Path('/not-written'))
                run.assert_not_called()

    def test_source_marker_cannot_hide_modified_added_or_missing_files(self):
        for change in ('modify', 'add', 'remove', 'symlink', 'executable-mode'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); archive = root / 'input.tar'; sources = root / 'sources'; sources.mkdir()
                with tarfile.open(archive, 'w') as target:
                    member = tarfile.TarInfo('source/COPYING'); member.size = 4
                    target.addfile(member, io.BytesIO(b'test'))
                pin = {'root': 'source', 'sha256': builder.digest(archive)}
                tree = builder.extract(archive, pin, sources)
                if change == 'modify': (tree / 'COPYING').write_bytes(b'evil')
                elif change == 'add': (tree / 'added.c').write_bytes(b'added')
                elif change == 'remove': (tree / 'COPYING').unlink()
                elif change == 'symlink':
                    (tree / 'COPYING').unlink(); (tree / 'COPYING').symlink_to(archive)
                else: (tree / 'COPYING').chmod(0o755)
                with self.assertRaises(ValueError): builder.extract(archive, pin, sources)


class PayloadTests(unittest.TestCase):
    def test_wrong_architecture_or_homebrew_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'ffmpeg'; binary.write_bytes(b'DMC_MEDIA_BOOKMARK_DIRECTORY'); binary.chmod(0o755)
            with patch.object(builder, 'run', return_value='arm64'):
                with self.assertRaises(ValueError): builder.validate_tool(binary)
            def fake(command, **_kwargs):
                if command[0] == 'lipo': return 'arm64 x86_64'
                if '-L' in command: return 'ffmpeg:\n\t/opt/homebrew/lib/libx264.dylib (compatibility version 1)'
                return ''
            with patch.object(builder, 'run', side_effect=fake):
                with self.assertRaises(ValueError): builder.validate_tool(binary)

    def test_every_slice_is_checked_and_system_dependencies_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'ffprobe'; binary.write_bytes(b'DMC_MEDIA_BOOKMARK_DIRECTORY'); binary.chmod(0o755)
            inspected = []
            def fake(command, **_kwargs):
                if command[0] == 'lipo': return 'x86_64 arm64'
                inspected.append(command[2])
                if '-L' in command: return 'ffprobe:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1)'
                return 'Load command 10\ncmd LC_BUILD_VERSION\nminos 11.0\nsdk 27.2\nntools 1\ntool LD\nversion 1222.1\nLoad command 11\ncmd LC_SOURCE_VERSION\nversion 999.0'
            with patch.object(builder, 'run', side_effect=fake): builder.validate_tool(binary)
            self.assertEqual(inspected, ['arm64', 'arm64', 'x86_64', 'x86_64'])

    def test_changed_payload_is_rejected_before_signing(self):
        with tempfile.TemporaryDirectory(prefix='dmc payload fixture ') as tmp:
            root = Path(tmp); (root / 'bin').mkdir()
            for name in builder.TOOLS: (root / 'bin' / name).write_bytes(b'fixture')
            receipt = {'architectures': builder.ARCHS, 'sources': json.loads(builder.PINS.read_text()),
                       'binaries': {name: builder.digest(root / 'bin' / name) for name in builder.TOOLS}}
            with patch.object(packager.builder, 'validate_tool'):
                packager.validate_payload(root, receipt)
                (root / 'bin/ffprobe').write_bytes(b'changed')
                with self.assertRaises(ValueError): packager.validate_payload(root, receipt)

    def test_newer_macos_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'ffmpeg'; binary.write_bytes(b'DMC_MEDIA_BOOKMARK_DIRECTORY'); binary.chmod(0o755)
            def fake(command, **_kwargs):
                if command[0] == 'lipo': return 'arm64 x86_64'
                if '-L' in command: return 'ffmpeg:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1)'
                return 'cmd LC_BUILD_VERSION\nminos 14.0'
            with patch.object(builder, 'run', side_effect=fake):
                with self.assertRaises(ValueError): builder.validate_tool(binary)


if __name__ == '__main__': unittest.main()
