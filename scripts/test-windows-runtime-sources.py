#!/usr/bin/env python3
"""Runtime source/notice export regressions using disposable synthetic archives."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runtime_sources', Path(__file__).with_name('export-windows-runtime-sources.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
FIXTURES = Path(os.environ['DMC_MEDIA_FIXTURE_ROOT']).resolve()
if FIXTURES.is_relative_to(Path(__file__).resolve().parent.parent):
    raise SystemExit('Use fixtures outside the repository')


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='dmc-runtime-source-test-', dir=FIXTURES)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.downloads = self.root / 'downloads'; self.downloads.mkdir()
        self.pins = {'schema': 1, 'toolchain_sha256': 'd' * 64, 'sources': {}}
        for number, name in enumerate(['llvm-project', 'mingw-w64', 'llvm-mingw']):
            path = self.downloads / (name + '.tar.gz')
            self.make_archive(path, name)
            self.pins['sources'][name] = {'url': 'https://fixtures.invalid/' + path.name,
                'filename': path.name, 'root': name, 'sha256': runtime.digest(path),
                'commit': str(number + 1) * 40, 'licenses': ['COPYING']}
        self.pins_path = self.root / 'pins.json'; self.pins_path.write_text(json.dumps(self.pins))
        self.receipt = self.root / 'media.json'
        self.receipt.write_text(json.dumps({'toolchain': {'sha256': 'd' * 64},
            'target': runtime.media.TARGET, 'tool_versions': {'target_clang': 'clang (' + '1' * 40 + ')'},
            'binaries': {'ffmpeg.exe': 'e' * 64, 'ffprobe.exe': 'f' * 64}}))
        self.patch = patch.object(runtime, 'PINS', self.pins_path); self.patch.start(); self.addCleanup(self.patch.stop)
        self.output = self.root / 'output'

    def make_archive(self, path, root, symlink=False, missing=False):
        with tarfile.open(path, 'w:gz') as archive:
            item = tarfile.TarInfo(root + ('/other' if missing else '/COPYING'))
            if symlink:
                item.type = tarfile.SYMTYPE; item.linkname = '../unrelated'; archive.addfile(item)
            else:
                value = b'Synthetic component license'; item.size = len(value); archive.addfile(item, io.BytesIO(value))

    def export(self):
        return runtime.export_offer(self.receipt, self.downloads, self.output)

    def test_export_keeps_full_sources_external_and_stages_small_notices(self):
        receipt = self.export(); destination = self.root / 'stage'
        runtime.stage_offer(self.output, self.receipt, destination)
        self.assertFalse((destination / receipt['source_archive']).exists())
        self.assertTrue((destination / 'runtime-source-receipt.json').is_file())
        self.assertEqual((destination / 'llvm-project/COPYING').read_bytes(), b'Synthetic component license')
        runtime.verify_offer(self.output, self.receipt)
        with tarfile.open(self.output / receipt['source_archive']) as archive:
            for item in archive.getmembers():
                self.assertEqual((item.uid, item.gid, item.uname, item.gname, item.mtime), (0, 0, '', '', 0))

    def test_wrong_toolchain_or_compiler_refuses_before_output_mutation(self):
        original = json.loads(self.receipt.read_text())
        for change in [{'toolchain': {'sha256': '0' * 64}}, {'tool_versions': {'target_clang': 'unrelated compiler'}}]:
            self.receipt.write_text(json.dumps(original | change))
            with self.assertRaises(ValueError): self.export()
            self.assertFalse(self.output.exists())

    def test_wrong_archive_bytes_preserve_cache_and_output(self):
        path = self.downloads / 'llvm-project.tar.gz'; path.write_bytes(b'Wrong source archive')
        with self.assertRaises(ValueError): self.export()
        self.assertEqual(path.read_bytes(), b'Wrong source archive'); self.assertFalse(self.output.exists())

    def test_symlink_or_missing_license_refuses_before_output_creation(self):
        for symlink in [True, False]:
            path = self.downloads / 'llvm-project.tar.gz'
            self.make_archive(path, 'llvm-project', symlink=symlink, missing=not symlink)
            self.pins['sources']['llvm-project']['sha256'] = runtime.digest(path)
            self.pins_path.write_text(json.dumps(self.pins))
            with self.assertRaises(ValueError): self.export()
            self.assertFalse(self.output.exists())

    def test_existing_destination_is_preserved(self):
        self.output.mkdir(); (self.output / 'keep').write_text('existing')
        with self.assertRaises(ValueError): self.export()
        self.assertEqual((self.output / 'keep').read_text(), 'existing')

    def test_existing_staged_notices_are_preserved(self):
        self.export(); destination = self.root / 'stage'; destination.mkdir()
        (destination / 'keep').write_text('existing notices')
        with self.assertRaises(ValueError): runtime.stage_offer(self.output, self.receipt, destination)
        self.assertEqual((destination / 'keep').read_text(), 'existing notices')

    def test_redirected_output_and_source_cache_are_rejected(self):
        elsewhere = self.root / 'elsewhere'; elsewhere.mkdir()
        redirected = self.root / 'redirected'; redirected.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(ValueError): runtime.export_offer(self.receipt, self.downloads, redirected / 'output')
        self.assertEqual(list(elsewhere.iterdir()), [])
        original = self.downloads / 'llvm-project.tar.gz'; preserved = self.root / 'preserved.tar.gz'
        original.rename(preserved); original.symlink_to(preserved)
        with self.assertRaises(ValueError): self.export()
        self.assertTrue(preserved.is_file()); self.assertFalse(self.output.exists())

    def test_rehashed_source_archive_still_requires_exact_pinned_inputs(self):
        receipt = self.export(); archive = self.output / receipt['source_archive']
        with tarfile.open(archive, 'w:xz') as stream:
            entry = tarfile.TarInfo('sources/llvm-project.tar.gz'); entry.size = 1; stream.addfile(entry, io.BytesIO(b'x'))
        receipt['source_archive_sha256'] = runtime.digest(archive)
        (self.output / 'runtime-source-receipt.json').write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): runtime.verify_offer(self.output, self.receipt)

    def test_distribution_export_verifies_without_loose_notice_directory(self):
        receipt = self.export(); dist = self.root / 'dist'; dist.mkdir()
        for name in [receipt['source_archive'], 'runtime-source-receipt.json']:
            shutil.copyfile(self.output / name, dist / name)
        runtime.verify_offer(dist, self.receipt, exported=True)
        (dist / receipt['source_archive']).write_bytes(b'Changed external source companion')
        with self.assertRaises(ValueError): runtime.verify_offer(dist, self.receipt, exported=True)

    def test_package_notice_changes_and_missing_external_companion_fail(self):
        receipt = self.export(); portable = self.root / 'portable'; installed = self.root / 'installed'
        a = portable / 'LICENSES/ffmpeg/runtime'; b = installed / 'app/licenses/LICENSES/ffmpeg/runtime'
        runtime.stage_offer(self.output, self.receipt, a); shutil.copytree(a, b)
        runtime.verify_packages(self.output, self.receipt, portable, installed)
        (b / 'llvm-project/COPYING').write_text('Changed notice')
        with self.assertRaises(ValueError): runtime.verify_packages(self.output, self.receipt, portable, installed)
        shutil.copyfile(a / 'llvm-project/COPYING', b / 'llvm-project/COPYING')
        (self.output / receipt['source_archive']).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): runtime.verify_packages(self.output, self.receipt, portable, installed)

    def test_rehashed_wrong_notice_cannot_replace_upstream_copyright(self):
        receipt = self.export(); source = self.output / receipt['source_archive']
        with tarfile.open(source) as archive:
            entries = [(item.name, archive.extractfile(item).read()) for item in archive.getmembers()]
        wrong = b'Synthetic changed copyright'
        with tarfile.open(source, 'w:xz') as archive:
            for name, raw in entries:
                if name == 'notices/llvm-project/COPYING': raw = wrong
                item = tarfile.TarInfo(name); item.size = len(raw); archive.addfile(item, io.BytesIO(raw))
        (self.output / 'licenses/llvm-project/COPYING').write_bytes(wrong)
        receipt['notices']['llvm-project/COPYING'] = hashlib.sha256(wrong).hexdigest()
        receipt['source_archive_sha256'] = runtime.digest(source)
        (self.output / 'runtime-source-receipt.json').write_text(json.dumps(receipt))
        (self.output / 'SOURCE.txt').write_text(runtime.pointer(receipt))
        with self.assertRaises(ValueError): runtime.verify_offer(self.output, self.receipt)


if __name__ == '__main__': unittest.main()
