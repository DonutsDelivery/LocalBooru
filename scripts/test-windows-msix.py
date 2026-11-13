#!/usr/bin/env python3
"""Owned synthetic packaging tests. Mock SDK outputs are never release artifacts."""
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

spec = importlib.util.spec_from_file_location('msix', Path(__file__).with_name('stage-windows-msix.py'))
msix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msix)


def pe(machine=0x8664):
    data = bytearray(512)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 128)
    data[128:132] = b'PE\0\0'
    struct.pack_into('<H', data, 132, machine)
    struct.pack_into('<H', data, 150, 2)
    struct.pack_into('<H', data, 152, 0x20b)
    return data


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = subprocess.check_output(['git', '-C', str(msix.ROOT), 'rev-parse', 'HEAD'], text=True).strip()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dmc-msix owned æ space-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.zip = self.root / 'portable fixture.zip'
        self.receipt = self.root / 'build receipt.json'
        self.files = {name: pe() for name in ('DonutMediaCenter.exe', 'bin/ffmpeg.exe', 'bin/ffprobe.exe', 'bin/codec.dll')}
        self.files.update({name: msix.git_file(self.source, name) for name in ('LICENSE', 'NOTICE', 'TRADEMARKS.md')})
        self.files['LICENSES/fixture.txt'] = b'Synthetic license fixture'
        self.write_fixture()

    def write_fixture(self, extra=None):
        with zipfile.ZipFile(self.zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in self.files.items(): archive.writestr(name, data)
            if extra: archive.writestr(*extra)
        record = dict(schema=1, source_commit=self.source, portable_sha256=msix.digest(self.zip),
                      target='x86_64-pc-windows-msvc',
                      recipe_sha256=msix.hashlib.sha256(msix.git_file(self.source, 'scripts/build-windows-docker.sh')).hexdigest())
        self.receipt.write_text(json.dumps(record))

    def stage(self):
        return msix.stage(self.zip, self.receipt, self.source, self.root / 'new package stage',
                          'Example.DMC.Test', 'CN=Fixture & Testing', 'Fixture & Testing', '2.1.1.0')[0]

    def test_valid_payload_manifest_and_spaces(self):
        stage = self.stage()
        manifest = ET.parse(stage / 'AppxManifest.xml')
        ns = {'m': 'http://schemas.microsoft.com/appx/manifest/foundation/windows10',
              'r': 'http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities'}
        self.assertEqual(manifest.find('m:Identity', ns).get('Publisher'), 'CN=Fixture & Testing')
        self.assertEqual(manifest.find('m:Applications/m:Application', ns).get('EntryPoint'), 'Windows.FullTrustApplication')
        self.assertEqual(manifest.find('m:Capabilities/r:Capability', ns).get('Name'), 'runFullTrust')
        self.assertNotIn('broadFileSystemAccess', (stage / 'AppxManifest.xml').read_text())
        self.assertEqual((stage / 'DonutMediaCenter.exe').read_bytes(), self.files['DonutMediaCenter.exe'])

    def test_tampered_zip_and_recipe_are_rejected_before_staging(self):
        with self.zip.open('ab') as stream: stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'receipt'): self.stage()
        self.assertFalse((self.root / 'new package stage').exists())
        self.write_fixture()
        record = json.loads(self.receipt.read_text()); record['recipe_sha256'] = '0' * 64
        self.receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'receipt'): self.stage()

    def test_unsafe_windows_paths_duplicates_and_private_payload_are_rejected(self):
        for name in ('../escape', '/absolute', 'LICENSES/./alias', 'LICENSES/a//b', 'LICENSES/NUL.txt',
                     'LICENSES/name.', 'bin/FFPROBE.EXE', 'LICENSES/name:stream', 'library.db'):
            with self.subTest(name=name):
                self.write_fixture((name, b'fixture'))
                with self.assertRaises(ValueError): self.stage()
                self.assertFalse((self.root / 'new package stage').exists())

    def test_symlinks_and_wrong_architecture_are_rejected(self):
        link = zipfile.ZipInfo('LICENSES/link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.write_fixture((link, b'/synthetic/never-read'))
        with self.assertRaisesRegex(ValueError, 'link'): self.stage()
        self.files['DonutMediaCenter.exe'] = pe(0x14c)
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'x64'): self.stage()
        self.assertFalse((self.root / 'new package stage').exists())

    def test_notice_missing_helper_and_version_checks(self):
        self.files['NOTICE'] = b'not the source notice'; self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'notice'): self.stage()
        del self.files['bin/ffprobe.exe']; self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'helpers'): self.stage()
        for version in ('2.1.1', '0.1.1.0', '2.1.65536.0', '2.01.1.0'):
            with self.assertRaises(ValueError): msix.validate_identity('Example.DMC.Test', 'CN=Fixture', version)

    def runner(self, stage, commands, fail=None, corrupt=False):
        def run(command, check):
            self.assertTrue(check)
            commands.append(command)
            action = command[1]
            if action == fail: raise subprocess.CalledProcessError(23, command)
            package_flag = '/p' if '/p' in command else '-p'
            directory_flag = '/d' if '/d' in command else '-d'
            if action == 'pack': Path(command[command.index(package_flag) + 1]).write_bytes(b'MOCK: not an MSIX release artifact')
            if action == 'unpack':
                output = Path(command[command.index(directory_flag) + 1]); shutil.copytree(stage, output)
                if corrupt: (output / 'DonutMediaCenter.exe').write_bytes(b'changed')
        return run

    def test_sdk_commands_validate_roundtrip_and_sign_explicitly(self):
        stage = self.stage(); commands = []; target = self.root / 'mock test.msix'
        result = msix.package(stage, target, '/mock SDK/MakeAppx.exe', '/mock SDK/SignTool.exe',
                              'A' * 40, 'https://timestamp.example.test', run=self.runner(stage, commands))
        self.assertTrue(result['signed_and_verified'])
        self.assertEqual([c[1] for c in commands], ['pack', 'unpack', 'sign', 'verify'])
        self.assertIn('/no', commands[0]); self.assertNotIn('/nv', commands[0])
        self.assertEqual(commands[0][commands[0].index('/d') + 1], str(stage))
        self.assertIn('/pa', commands[-1])

    def test_each_sdk_failure_propagates_and_removes_only_partial_new_package(self):
        stage = self.stage()
        for action in ('pack', 'unpack', 'sign', 'verify'):
            target = self.root / (action + '-mock.msix')
            with self.assertRaises(subprocess.CalledProcessError):
                msix.package(stage, target, 'mock', 'mock-sign', 'A' * 40, 'https://timestamp.example.test',
                             run=self.runner(stage, [], fail=action))
            self.assertFalse(target.exists())
            self.assertTrue(stage.is_dir())

    def test_native_sdk_skips_only_unsigned_signature_check_during_roundtrip(self):
        stage = self.stage(); commands = []; target = self.root / 'native-mock.msix'
        result = msix.package(stage, target, '/mock/makemsix', run=self.runner(stage, commands), packager_kind='makemsix')
        self.assertEqual(commands[0], ['/mock/makemsix', 'pack', '-d', str(stage), '-p', str(target)])
        self.assertIn('-ss', commands[1])
        self.assertNotIn('-ss', commands[0])
        self.assertFalse(result['signed_and_verified'])
        self.assertEqual(result['packager_kind'], 'makemsix')

    def test_corrupt_roundtrip_and_missing_sdk_never_produce_a_package(self):
        stage = self.stage(); target = self.root / 'corrupt-mock.msix'
        with self.assertRaisesRegex(ValueError, 'round-trip'):
            msix.package(stage, target, 'mock', run=self.runner(stage, [], corrupt=True))
        self.assertFalse(target.exists())
        with self.assertRaises(ValueError): msix.package(stage, target, None)
        with self.assertRaises(FileNotFoundError): msix.package(stage, target, '/nonexistent-owned-fixture/MakeAppx')
        self.assertFalse(target.exists())


if __name__ == '__main__': unittest.main()
