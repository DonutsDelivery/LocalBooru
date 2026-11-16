#!/usr/bin/env python3
"""Synthetic AVIF packaging checks; no compiler or application execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('avif', Path(__file__).with_name('stage-appimage-avif.py'))
avif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avif)


def inspect_fixture(path):
    return json.loads(path.read_text())


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dmc-avif-fixture-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'AppDir'
        self.libs = self.root / 'usr/lib'
        self.libs.mkdir(parents=True)
        (self.root / 'AppRun').write_text('synthetic launcher\n')
        self.source = self.base / 'image-libs'
        self.source.mkdir()
        self.write(avif.ROOT, ['libgav1.so.1', 'libsharpyuv.so.0'], imports=['SharpYuvConvert'])
        self.write('libgav1.so.1', ['libabsl_test.so.1', 'libc.so.6'])
        self.write('libabsl_test.so.1', ['libgav1.so.1', 'libstdc++.so.6'])
        self.write('libsharpyuv.so.0', ['libc.so.6'], exports=['SharpYuvConvert'])

    def write(self, name, needed, imports=(), exports=()):
        (self.source / name).write_text(json.dumps(dict(soname=name, needed=sorted(needed), sharp_imports=sorted(imports), sharp_exports=sorted(exports))))

    def stage(self):
        return avif.stage(self.root, [self.source], inspect_fixture)

    def verify(self):
        return avif.verify(self.root, inspect_fixture)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_recursive_chain_cycle_and_platform_boundary(self):
        manifest = self.stage()
        self.assertEqual(len(manifest['libraries']), 4)
        self.assertNotIn('libc.so.6', manifest['libraries'])
        self.assertNotIn('libstdc++.so.6', manifest['libraries'])
        self.assertEqual(self.verify(), manifest)
        self.assertEqual((self.libs / avif.ROOT).read_bytes(), (self.source / avif.ROOT).read_bytes())

    def test_replaces_existing_codec_with_matching_image_bytes(self):
        (self.libs / 'libsharpyuv.so.0').write_text('old mismatched codec')
        self.stage()
        self.assertEqual((self.libs / 'libsharpyuv.so.0').read_bytes(), (self.source / 'libsharpyuv.so.0').read_bytes())

    def test_missing_transitive_dependency_refuses_before_mutation(self):
        (self.source / 'libabsl_test.so.1').unlink()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'Missing image codec dependency'):
            self.stage()
        self.assertEqual(self.snapshot(), before)

    def test_incompatible_sharpyuv_refuses_before_overwriting_existing_file(self):
        self.write(avif.ROOT, ['libsharpyuv.so.0'], imports=['SharpYuvConvertWithOptions'])
        (self.libs / 'libsharpyuv.so.0').write_text('prior retained bytes')
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'SharpYuvConvertWithOptions'):
            self.stage()
        self.assertEqual(self.snapshot(), before)

    def test_wrong_source_soname_refuses_before_mutation(self):
        data = inspect_fixture(self.source / avif.ROOT)
        data['soname'] = 'libavif.so.99'
        (self.source / avif.ROOT).write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'SONAME'):
            self.stage()
        self.assertEqual(list(self.libs.iterdir()), [])

    def test_dependency_traversal_refuses_before_mutation(self):
        self.write(avif.ROOT, ['../outside'])
        with self.assertRaisesRegex(ValueError, 'Unsafe library name'):
            self.stage()
        self.assertEqual(list(self.libs.iterdir()), [])

    def test_source_symlink_escape_is_refused(self):
        original = self.source / avif.ROOT
        original.rename(self.base / 'outside')
        original.symlink_to(self.base / 'outside')
        with self.assertRaisesRegex(ValueError, 'Missing image codec dependency'):
            self.stage()

    def test_redirected_output_directory_is_refused_before_mutation(self):
        self.libs.rmdir()
        self.libs.symlink_to(self.source, target_is_directory=True)
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with self.assertRaisesRegex(ValueError, 'Redirected package path'):
            self.stage()
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)

    def test_manifest_symlink_is_refused_before_copy(self):
        path = self.root / avif.MANIFEST
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root / 'AppRun')
        with self.assertRaises(ValueError):
            self.stage()
        self.assertEqual(list(self.libs.iterdir()), [])
        self.assertEqual((self.root / 'AppRun').read_text(), 'synthetic launcher\n')

    def test_verify_rejects_missing_transitive_file(self):
        self.stage()
        (self.libs / 'libabsl_test.so.1').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing regular packaged codec'):
            self.verify()

    def test_verify_rejects_changed_bytes(self):
        self.stage()
        with (self.libs / 'libabsl_test.so.1').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'differs from manifest'):
            self.verify()

    def test_verify_rejects_extra_manifest_entry(self):
        self.stage()
        path = self.root / avif.MANIFEST
        manifest = json.loads(path.read_text())
        manifest['libraries']['unrelated.so.1'] = {}
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'does not match closure'):
            self.verify()

    def test_readelf_parser_extracts_real_tag_and_symbol_shapes(self):
        dynamic = ' 0x0000000000000001 (NEEDED) Shared library: [libsharpyuv.so.0]\n 0x000000000000000e (SONAME) Library soname: [libavif.so.16]\n'
        symbols = ' 1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND SharpYuvConvert\n 2: 0000000000001234 5 FUNC GLOBAL DEFAULT 12 avifDecoderCreate\n 3: 0000000000000000 0 NOTYPE WEAK DEFAULT UND weak_optional\n'
        with patch.object(avif.subprocess, 'check_output', side_effect=[dynamic, symbols]):
            result = avif.inspect(Path('synthetic.so'))
        self.assertEqual(result['soname'], avif.ROOT)
        self.assertEqual(result['sharp_imports'], ['SharpYuvConvert'])
        self.assertEqual(result['needed'], ['libsharpyuv.so.0'])

    def test_readelf_parser_rejects_codec_runtime_path(self):
        dynamic = ' (SONAME) Library soname: [libavif.so.16]\n (RUNPATH) Library runpath: [/build/private]\n'
        with patch.object(avif.subprocess, 'check_output', side_effect=[dynamic, '']):
            with self.assertRaisesRegex(ValueError, 'unexpected runtime path'):
                avif.inspect(Path('synthetic.so'))


if __name__ == '__main__':
    unittest.main()
