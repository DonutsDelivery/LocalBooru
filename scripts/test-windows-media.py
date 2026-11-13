#!/usr/bin/env python3
"""Supply-chain and feature guards; synthetic fixtures, no compiler execution."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('windows_media', Path(__file__).with_name('build-windows-media.py'))
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)
FIXTURES = Path(os.environ['DMC_MEDIA_FIXTURE_ROOT']).resolve()
if not FIXTURES.is_absolute() or FIXTURES.is_relative_to(Path(__file__).resolve().parent.parent):
    raise SystemExit('Use an owned fixture directory outside the source repository')


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dmc-windows-media-fixture-', dir=FIXTURES)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = self.root / 'sources'; self.sources.mkdir()

    def archive(self, entries=None):
        path = self.root / 'source.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, data, link in entries or [('source/file.txt', b'synthetic source', None)]:
                member = tarfile.TarInfo(name)
                if link is not None:
                    member.type = tarfile.SYMTYPE; member.linkname = link
                    archive.addfile(member)
                else:
                    member.size = len(data); member.mode = 0o644
                    archive.addfile(member, io.BytesIO(data))
        pin = {'url': 'https://fixtures.invalid/source.tar.gz', 'root': 'source',
               'sha256': media.digest(path), 'license': 'file.txt'}
        return path, pin

    def test_wrong_archive_hash_preserves_existing_bytes(self):
        path, pin = self.archive(); original = path.read_bytes(); pin['sha256'] = '0' * 64
        with self.assertRaises(ValueError): media.download(pin, self.root)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError): media.extract(path, pin, self.sources)
        self.assertFalse((self.sources / 'source').exists())

    def test_nonpublic_url_and_escaping_root_are_rejected(self):
        path, pin = self.archive()
        pin['url'] = 'file:///unrelated-synthetic-file'
        with self.assertRaises(ValueError): media.download(pin, self.root)
        pin['root'] = '../outside'
        with self.assertRaises(ValueError): media.extract(path, pin, self.sources)

    def test_valid_relative_symlink_and_modified_tree(self):
        path, pin = self.archive([('source/file.txt', b'synthetic source', None),
                                  ('source/link', b'', 'file.txt')])
        tree = media.extract(path, pin, self.sources)
        self.assertTrue((tree / 'link').is_symlink())
        media.verify_tree(path, pin, tree)
        (tree / 'file.txt').write_text('changed synthetic source')
        with self.assertRaises(ValueError): media.extract(path, pin, self.sources)

    def test_added_source_entry_is_rejected(self):
        path, pin = self.archive(); tree = media.extract(path, pin, self.sources)
        (tree / 'added').write_text('synthetic modification')
        with self.assertRaises(ValueError): media.verify_tree(path, pin, tree)

    def test_new_attempt_excludes_previous_compiler_products(self):
        recipe = 'f' * 64
        first = media.fresh_work_directory(self.root, recipe)
        (first / 'nasm').mkdir()
        (first / 'nasm/nasm').write_bytes(b'Substituted synthetic executable, not runnable')
        (first / 'ffmpeg.o').write_bytes(b'Stale synthetic compiler product')
        second = media.fresh_work_directory(self.root, recipe)
        self.assertNotEqual(first, second)
        self.assertEqual(list(second.iterdir()), [])
        self.assertTrue((first / 'ffmpeg.o').is_file())

    def test_unsafe_archive_members_are_rejected(self):
        for name, link in [('../outside', None), ('/absolute', None),
                           ('source/link', '../../outside'), ('source/link', '..')]:
            with self.subTest(name=name, link=link):
                path, pin = self.archive([(name, b'synthetic', link)])
                with self.assertRaises(ValueError): media.extract(path, pin, self.sources)
        self.assertEqual(list(self.sources.iterdir()), [])

    def test_symlink_cannot_replace_cached_file(self):
        path, pin = self.archive(); tree = media.extract(path, pin, self.sources)
        outside = self.root / 'unrelated-synthetic-file'; outside.write_text('preserve')
        (tree / 'file.txt').unlink(); (tree / 'file.txt').symlink_to(outside)
        with self.assertRaises(ValueError): media.verify_tree(path, pin, tree)
        self.assertEqual(outside.read_text(), 'preserve')

    def test_missing_gpu_feature_and_nonfree_are_rejected(self):
        config = '#define CONFIG_GPL 1\n#define CONFIG_VERSION3 1\n#define CONFIG_NONFREE 0\n'
        config += ''.join('#define CONFIG_' + name + ' 1\n' for name in media.REQUIRED)
        self.assertEqual(media.check_features(config), list(media.REQUIRED))
        with self.assertRaises(ValueError):
            media.check_features(config.replace('#define CONFIG_SCALE_CUDA_FILTER 1\n', ''))
        with self.assertRaises(ValueError):
            media.check_features(config.replace('CONFIG_NONFREE 0', 'CONFIG_NONFREE 1'))

    def test_unresolved_dll_and_wrong_architecture_are_rejected(self):
        valid = 'Format: COFF-x86-64\nImport {\n Name: KERNEL32.dll\n}\n'
        with patch.object(media, 'run', return_value=valid):
            self.assertEqual(media.validate_pe(Path('synthetic.exe'), 'synthetic-readobj'), ['KERNEL32.dll'])
        with patch.object(media, 'run', return_value=valid + ' Name: private-codec.dll\n'):
            with self.assertRaises(ValueError): media.validate_pe(Path('synthetic.exe'), 'synthetic-readobj')
        with patch.object(media, 'run', return_value=valid.replace('COFF-x86-64', 'COFF-i386')):
            with self.assertRaises(ValueError): media.validate_pe(Path('synthetic.exe'), 'synthetic-readobj')

    def output_fixture(self):
        archive, pin = self.archive()
        pins = {'sources': {'fixture': pin}, 'toolchain': {'sha256': '0' * 64}}
        pins_path = self.root / 'pins.json'; pins_path.write_text(json.dumps(pins))
        output = self.root / 'output'; (output / 'bin').mkdir(parents=True)
        for name in media.TOOLS: (output / 'bin' / name).write_bytes(b'Synthetic unit-test bytes, not runnable')
        (output / 'licenses').mkdir()
        (output / 'licenses/COPYING').write_text('Synthetic license fixture')
        source_name = 'source-offer.tar.xz'
        with patch.object(media, 'PINS', pins_path):
            with tarfile.open(output / source_name, 'w:xz') as source:
                source.add(archive, arcname='sources/' + archive.name)
                for name, path in media.recipe_files().items(): source.add(path, arcname=name)
        receipt = {'recipe_sha256': 'f' * 64, 'target': media.TARGET,
                   'sources': pins['sources'], 'toolchain': pins['toolchain'],
                   'tool_versions': {'fixture': 'synthetic, not executed'},
                   'environment_sha256': hashlib.sha256(json.dumps(
                       {'fixture': 'synthetic, not executed'}, sort_keys=True).encode()).hexdigest(),
                   'required_components': list(media.REQUIRED), 'source_archive': source_name,
                   'licenses': {'COPYING': media.digest(output / 'licenses/COPYING')},
                   'source_archive_sha256': media.digest(output / source_name),
                   'binaries': {name: media.digest(output / 'bin' / name) for name in media.TOOLS}}
        (output / 'media-build-receipt.json').write_text(json.dumps(receipt))
        return output, pins_path, receipt

    def test_output_receipt_binds_payload_and_toolchain(self):
        output, pins_path, receipt = self.output_fixture()
        with patch.object(media, 'PINS', pins_path):
            media.verify_output(output, 'f' * 64)
            path = output / 'bin' / media.TOOLS[0]; original = path.read_bytes()
            path.write_bytes(b'changed synthetic bytes')
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)
            path.write_bytes(original)
            receipt['toolchain'] = {'sha256': '1' * 64}
            (output / 'media-build-receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)

    def test_receipt_cannot_redirect_source_offer(self):
        output, pins_path, receipt = self.output_fixture()
        receipt['source_archive'] = '../unrelated-synthetic-file'
        (output / 'media-build-receipt.json').write_text(json.dumps(receipt))
        with patch.object(media, 'PINS', pins_path):
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)

    def test_tool_version_receipt_change_is_rejected(self):
        output, pins_path, receipt = self.output_fixture()
        receipt['tool_versions']['fixture'] = 'changed synthetic version'
        (output / 'media-build-receipt.json').write_text(json.dumps(receipt))
        with patch.object(media, 'PINS', pins_path):
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)

    def test_changed_or_missing_notices_are_rejected(self):
        output, pins_path, receipt = self.output_fixture()
        with patch.object(media, 'PINS', pins_path):
            (output / 'licenses/COPYING').write_text('changed synthetic license')
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)
            (output / 'licenses/COPYING').unlink()
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)

    def test_release_staging_keeps_helpers_notices_and_sources_together(self):
        output, pins_path, receipt = self.output_fixture()
        release = self.root / 'fresh-release'; release.mkdir()
        with patch.object(media, 'PINS', pins_path), patch.object(media, 'recipe_hash', return_value='f' * 64):
            media.stage_output(output, release)
            for name in media.TOOLS:
                self.assertEqual((release / 'src-tauri/ffmpeg-bin' / name).read_bytes(),
                                 (output / 'bin' / name).read_bytes())
            notices = release / 'LICENSES/ffmpeg'
            for name in (receipt['source_archive'], 'media-build-receipt.json'):
                self.assertEqual((notices / name).read_bytes(), (output / name).read_bytes())
            self.assertEqual((notices / 'COPYING').read_text(), 'Synthetic license fixture')
            with self.assertRaises(ValueError): media.stage_output(output, release)

    def test_packaged_helpers_and_source_offer_must_match(self):
        output, pins_path, receipt = self.output_fixture()
        portable = self.root / 'portable'; installed = self.root / 'installed'
        for destination in (portable / 'bin', installed / 'app/bin'):
            destination.mkdir(parents=True)
            for name in media.TOOLS:
                (destination / name).write_bytes((output / 'bin' / name).read_bytes())
        for destination in (portable / 'LICENSES/ffmpeg', installed / 'app/licenses/LICENSES/ffmpeg'):
            destination.mkdir(parents=True)
            for name in (receipt['source_archive'], 'media-build-receipt.json'):
                (destination / name).write_bytes((output / name).read_bytes())
            (destination / 'COPYING').write_bytes((output / 'licenses/COPYING').read_bytes())
        with patch.object(media, 'PINS', pins_path), patch.object(media, 'recipe_hash', return_value='f' * 64):
            media.verify_packages(output, portable, installed)
            helper = installed / 'app/bin/ffmpeg.exe'; original = helper.read_bytes()
            helper.write_bytes(b'wrong synthetic payload')
            with self.assertRaises(ValueError): media.verify_packages(output, portable, installed)
            helper.write_bytes(original)
            (installed / 'app/licenses/LICENSES/ffmpeg' / receipt['source_archive']).unlink()
            with self.assertRaises(ValueError): media.verify_packages(output, portable, installed)

    def test_rehashed_source_offer_still_must_contain_pinned_inputs(self):
        output, pins_path, receipt = self.output_fixture()
        path = output / receipt['source_archive']
        with patch.object(media, 'PINS', pins_path):
            with tarfile.open(path, 'w:xz') as source:
                data = b'wrong synthetic source'
                member = tarfile.TarInfo('sources/source.tar.gz'); member.size = len(data)
                source.addfile(member, io.BytesIO(data))
                for name, original in media.recipe_files().items(): source.add(original, arcname=name)
            receipt['source_archive_sha256'] = media.digest(path)
            (output / 'media-build-receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): media.verify_output(output, 'f' * 64)


if __name__ == '__main__':
    unittest.main()
