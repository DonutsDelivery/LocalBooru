#!/usr/bin/env python3
"""Select a current Tauri bundle and verify its compiled payload without execution."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


def regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f'not an unredirected regular file: {path}')
    return path


def config(path):
    data = json.loads(regular(path).read_text())
    name, version = data['productName'], data['version']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._-]*', name):
        raise ValueError('invalid product name')
    if not isinstance(version, str) or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('invalid version')
    return name, version


def select(conf, bundle, kind):
    name, version = config(conf)
    filename = {'deb': f'{name}_{version}_amd64.deb',
                'rpm': f'{name}-{version}-1.x86_64.rpm',
                'appimage': f'{name}_{version}_amd64.AppImage'}[kind]
    return regular(Path(bundle) / kind / filename)


def sections(path):
    data = regular(path).read_bytes()
    if len(data) < 64 or data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 62:
        raise ValueError('expected ELF64 little-endian x86_64')
    off = struct.unpack_from('<Q', data, 40)[0]
    size, count, names = struct.unpack_from('<HHH', data, 58)
    if size < 64 or not count or names >= count or off + size * count > len(data):
        raise ValueError('invalid section table')
    def section(i):
        start = off + i * size
        name = struct.unpack_from('<I', data, start)[0]
        pos, length = struct.unpack_from('<QQ', data, start + 24)
        if struct.unpack_from('<I', data, start + 4)[0] == 8:  # SHT_NOBITS
            return name, b''
        if pos + length > len(data):
            raise ValueError('invalid section bounds')
        return name, data[pos:pos + length]
    strings = section(names)[1]
    result = {}
    wanted = {b'.text', b'.rodata', b'.note.gnu.build-id'}
    for i in range(count):
        n, content = section(i)
        end = strings.find(b'\0', n)
        if n >= len(strings) or end < 0:
            raise ValueError('invalid section name')
        name = strings[n:end]
        if name in wanted:
            if name in result or not content:
                raise ValueError('duplicate or empty section')
            result[name] = content
    if result.keys() != wanted:
        raise ValueError('missing required sections')
    return result


def verify(actual, expected, loader_adjusted=False):
    a, b = regular(actual), regular(expected)
    if loader_adjusted:
        if sections(a) != sections(b):
            raise ValueError('ELF code, rodata or build ID mismatch')
        return 'ELF .text, .rodata and GNU build ID match (loader adjustments allowed)'
    if hashlib.sha256(a.read_bytes()).digest() != hashlib.sha256(b.read_bytes()).digest():
        raise ValueError('binary SHA-256 mismatch')
    return 'binary SHA-256 matches'


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select'); s.add_argument('config'); s.add_argument('bundle'); s.add_argument('kind', choices=['deb', 'rpm', 'appimage'])
    v = sub.add_parser('verify-binary'); v.add_argument('actual'); v.add_argument('expected'); v.add_argument('--loader-adjusted', action='store_true')
    args = p.parse_args()
    try:
        print(select(args.config, args.bundle, args.kind) if args.command == 'select' else verify(args.actual, args.expected, args.loader_adjusted))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        p.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
