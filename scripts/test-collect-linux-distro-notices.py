#!/usr/bin/env python3
"""Notice retention uses synthetic package metadata outside the checkout."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('notices', Path(__file__).with_name('collect-linux-distro-notices.py'))
notices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notices)


class NoticeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dmc-notice-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.system = self.base / 'system'
        self.app = self.base / 'AppDir'
        self.info = self.system / 'var/lib/dpkg/info'
        self.info.mkdir(parents=True)
        self.status = self.system / 'var/lib/dpkg/status'
        self.status.touch()
        common = self.system / 'usr/share/common-licenses'
        common.mkdir(parents=True)
        (common / 'GPL-3').write_text('Synthetic common license\n')

    def package(self, name, owned='/usr/lib/x86_64-linux-gnu/libexample.so.1', copyright='Synthetic copyright\n'):
        with self.status.open('a') as file:
            file.write(f'Package: {name}\nStatus: install ok installed\nVersion: 1.2-3build1\nSource: example (1.2-3)\n\n')
        (self.info / (name + ':amd64.list')).write_text(owned + '\n')
        doc = self.system / 'usr/share/doc' / name
        doc.mkdir(parents=True)
        if copyright is not None:
            (doc / 'copyright').write_text(copyright)
        return doc / 'copyright'

    def binary(self, relative='usr/lib/libexample.so.1', app=None):
        path = (app or self.app) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\x7fELF\x02\x01' + b'\0' * 10 + b'\x03\0\0\0' + b'synthetic-notice-fixture')

    def result(self, app=None):
        path = (app or self.app) / 'usr/share/doc/localbooru/distro-packages/manifest.json'
        return json.loads(path.read_text()), path.parent

    def test_flattened_alias_carries_source_version_notice_and_hash(self):
        self.package('libexample-dev')
        self.binary()
        notices.collect(self.app, self.system)
        manifest, output = self.result()
        item = manifest['packages'][0]
        self.assertEqual(item['binary_package'], 'libexample-dev')
        self.assertEqual(item['source_version'], '1.2-3')
        self.assertEqual(item['candidate_files'], ['usr/lib/libexample.so.1'])
        content = (output / item['copyright']['path']).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), item['copyright']['sha256'])

    def test_ambiguous_basename_retains_both_candidate_notices(self):
        self.package('gtk-three', '/usr/lib/gtk-3/print.so', 'Three')
        self.package('gtk-four', '/usr/lib/gtk-4/print.so', 'Four')
        self.binary('usr/lib/gtk-3/print.so')
        notices.collect(self.app, self.system)
        manifest, _ = self.result()
        self.assertEqual([x['binary_package'] for x in manifest['packages']], ['gtk-four', 'gtk-three'])
        self.assertIn('candidate', manifest['scope'])

    def test_identical_staged_alias_bytes_retain_dev_and_runtime_notices(self):
        self.package('example-dev', '/usr/lib/libexample.so')
        self.package('example-runtime', '/usr/lib/libexample.so.1')
        self.binary('usr/lib/libexample.so')
        self.binary('usr/lib/libexample.so.1')
        notices.collect(self.app, self.system)
        manifest, _ = self.result()
        self.assertEqual([x['binary_package'] for x in manifest['packages']], ['example-dev', 'example-runtime'])

    def test_public_doc_and_common_license_symlinks_are_materialized(self):
        copyright = self.package('libexample-dev', copyright=None)
        actual = self.system / 'usr/share/doc/runtime'
        actual.mkdir()
        (actual / 'copyright').write_text('Runtime copyright')
        copyright.symlink_to('../runtime/copyright')
        (self.system / 'usr/share/common-licenses/GPL').symlink_to('GPL-3')
        self.binary()
        notices.collect(self.app, self.system)
        manifest, output = self.result()
        copied = output / manifest['packages'][0]['copyright']['path']
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_text(), 'Runtime copyright')
        self.assertEqual((output / 'common-licenses/GPL').read_text(), 'Synthetic common license\n')

    def test_copyright_link_outside_public_directories_is_rejected(self):
        copyright = self.package('libexample-dev', copyright=None)
        secret = self.system / 'synthetic-private'
        secret.write_text('Not a public license')
        copyright.symlink_to(secret)
        self.binary()
        with self.assertRaisesRegex(ValueError, 'public'):
            notices.collect(self.app, self.system)
        self.assertFalse((self.app / 'usr/share/doc/localbooru/distro-packages').exists())

    def test_missing_notice_fails_with_package_identity(self):
        self.package('libexample-dev', copyright=None)
        self.binary()
        with self.assertRaisesRegex(ValueError, 'libexample-dev'):
            notices.collect(self.app, self.system)

    def test_manifest_is_deterministic_and_has_no_fixture_paths(self):
        self.package('libexample-dev')
        self.binary()
        other = self.base / 'AnotherAppDir'
        self.binary(app=other)
        notices.collect(self.app, self.system)
        notices.collect(other, self.system)
        first, _ = self.result()
        second, _ = self.result(other)
        self.assertEqual(first, second)
        self.assertNotIn(str(self.base), json.dumps(first))

    def test_output_cannot_follow_an_escaping_staging_symlink(self):
        self.package('libexample-dev')
        self.binary()
        outside = self.base / 'outside'
        outside.mkdir()
        doc = self.app / 'usr/share/doc'
        doc.parent.mkdir(parents=True)
        doc.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'output'):
            notices.collect(self.app, self.system)
        self.assertEqual(list(outside.iterdir()), [])

    def test_packaged_notice_tampering_is_detected(self):
        self.package('libexample-dev')
        self.binary()
        notices.collect(self.app, self.system)
        notices.verify(self.app)
        manifest, output = self.result()
        (output / manifest['packages'][0]['copyright']['path']).write_text('Changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            notices.verify(self.app)

    def test_unowned_distro_object_fails_but_custom_runtime_is_separate(self):
        self.binary('usr/lib/unknown.so')
        with self.assertRaisesRegex(ValueError, 'unknown.so'):
            notices.collect(self.app, self.system)
        (self.app / 'usr/lib/unknown.so').unlink()
        self.binary('usr/lib/localbooru/native-svp/lib/custom.so')
        notices.collect(self.app, self.system)
        manifest, _ = self.result()
        self.assertEqual(manifest['non_distro_files'], ['usr/lib/localbooru/native-svp/lib/custom.so'])


if __name__ == '__main__':
    unittest.main()
