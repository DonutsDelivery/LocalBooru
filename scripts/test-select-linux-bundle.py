#!/usr/bin/env python3
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('selector', Path(__file__).with_name('select-linux-bundle.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.conf = self.root / 'config.json'
        self.conf.write_text(json.dumps({'productName': 'DonutMediaCenter', 'version': '2.1.1'}))
        self.names = {'deb': 'DonutMediaCenter_2.1.1_amd64.deb', 'rpm': 'DonutMediaCenter-2.1.1-1.x86_64.rpm', 'appimage': 'DonutMediaCenter_2.1.1_amd64.AppImage'}
        for kind, name in self.names.items():
            d = self.root / kind
            d.mkdir()
            (d / name.replace('2.1.1', '2.1.0')).write_text('old')
            (d / name.replace('2.1.1', '9.0.0')).write_text('new')
            (d / name).write_text('current')

    def test_exact_selection(self):
        for kind, name in self.names.items():
            self.assertEqual(m.select(self.conf, self.root, kind), self.root / kind / name)

    def test_missing_no_fallback(self):
        for kind, name in self.names.items():
            (self.root / kind / name).unlink()
            with self.assertRaises(ValueError): m.select(self.conf, self.root, kind)

    def test_invalid_and_redirected(self):
        p = self.root / 'deb' / self.names['deb']
        p.unlink(); p.symlink_to(self.root / 'config.json')
        with self.assertRaises(ValueError): m.select(self.conf, self.root, 'deb')
        self.conf.write_text('{"productName":"../bad","version":"2.1.1"}')
        with self.assertRaises(ValueError): m.select(self.conf, self.root, 'rpm')
        self.assertTrue(p.is_symlink())

    def test_exact_binary(self):
        a, b = self.root / 'a', self.root / 'b'
        a.write_bytes(b'current'); b.write_bytes(b'current')
        m.verify(a, b)
        b.write_bytes(b'stale')
        with self.assertRaises(ValueError): m.verify(a, b)

    def test_elf_sections(self):
        a, b = self.root / 'a', self.root / 'b'
        shutil.copyfile('/bin/ls', a); shutil.copyfile(a, b)
        m.verify(a, b, True)
        subprocess.run(['patchelf', '--set-rpath', '/fixture-only', str(b)], check=True)
        self.assertNotEqual(a.read_bytes(), b.read_bytes())
        m.verify(a, b, True)
        data = bytearray(b.read_bytes()); contents = m.sections(b)[b'.text']; pos = data.find(contents); data[pos] ^= 1; b.write_bytes(data)
        with self.assertRaises(ValueError): m.verify(a, b, True)
        shutil.copyfile(a, b)
        data = bytearray(b.read_bytes()); pos = data.find(b'.rodata\0'); self.assertGreater(pos, 0); data[pos + 1] = ord('X'); b.write_bytes(data)
        with self.assertRaises(ValueError): m.verify(a, b, True)
        b.write_bytes(b'not ELF')
        with self.assertRaises(ValueError): m.verify(a, b, True)


if __name__ == '__main__': unittest.main()
