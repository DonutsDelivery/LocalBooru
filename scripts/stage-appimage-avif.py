#!/usr/bin/env python3
"""Retain the image's AVIF codec dependency chain after replacing WebKit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = 'libavif.so.16'
MANIFEST = Path('usr/share/doc/localbooru/avif-dependencies.json')
# These remain host platform dependencies, as in the base Tauri AppImage.
PLATFORM = {'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0',
            'librt.so.1', 'libresolv.so.2', 'libstdc++.so.6', 'libgcc_s.so.1',
            'ld-linux-x86-64.so.2'}


def inspect(path):
    dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
    symbols = subprocess.check_output(['readelf', '--dyn-syms', '--wide', str(path)], text=True)
    needed = re.findall(r'\(NEEDED\).*?\[([^]]+)\]', dynamic)
    sonames = re.findall(r'\(SONAME\).*?\[([^]]+)\]', dynamic)
    if re.search(r'\((?:RPATH|RUNPATH)\)', dynamic):
        raise ValueError(f'Codec carries an unexpected runtime path: {path}')
    if len(sonames) != 1:
        raise ValueError(f'Expected one library SONAME: {path}')
    imports, exports = set(), set()
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0].endswith(':'):
            name = fields[7].split('@', 1)[0]
            if fields[6] == 'UND' and fields[4] == 'GLOBAL':
                imports.add(name)
            elif fields[6] != 'UND' and fields[4] in ('GLOBAL', 'WEAK'):
                exports.add(name)
    return {'soname': sonames[0], 'needed': sorted(set(needed)),
            'sharp_imports': sorted(n for n in imports if n.startswith('SharpYuv')),
            'sharp_exports': sorted(n for n in exports if n.startswith('SharpYuv'))}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    if name != Path(name).name or not re.fullmatch(r'[A-Za-z0-9_+.-]+', name) or name in ('.', '..'):
        raise ValueError(f'Unsafe library name: {name}')


def checked_path(root, relative):
    path = root / relative
    # Package directories must not redirect writes or verification outside AppDir.
    if path.parent.resolve() != path.parent or (path.is_symlink() and not path.resolve().is_relative_to(root)):
        raise ValueError(f'Redirected package path: {relative}')
    return path


def closure(resolve, inspect_library=inspect):
    pending, records = [ROOT], {}
    while pending:
        name = pending.pop()
        safe_name(name)
        if name in PLATFORM or name in records:
            continue
        path = resolve(name)
        info = inspect_library(path)
        if info['soname'] != name:
            raise ValueError(f'Unexpected SONAME for {name}')
        for dependency in info['needed']:
            safe_name(dependency)
        records[name] = {'path': path, **info}
        pending.extend(info['needed'])
    provided = set().union(*(set(info['sharp_exports']) for info in records.values()))
    required = set().union(*(set(info['sharp_imports']) for info in records.values()))
    if required - provided:
        raise ValueError('AVIF/SharpYUV mismatch: ' + ', '.join(sorted(required - provided)))
    return records


def package_root(root):
    if not root.is_absolute() or root.resolve() != root or not root.is_dir() or root == Path('/'):
        raise ValueError('Expected an absolute, unredirected staged AppDir')
    if not (root / 'AppRun').is_file() or not (root / 'usr/lib').is_dir():
        raise ValueError('Missing staged AppImage layout')


def stage(root, directories, inspect_library=inspect):
    package_root(root)
    directories = [directory.resolve(strict=True) for directory in directories]
    def resolve(name):
        matches = [(directory / name).resolve(strict=True)
                   for directory in directories if (directory / name).exists()]
        if not matches or not matches[0].is_file() or not any(matches[0].is_relative_to(d) for d in directories):
            raise ValueError(f'Missing image codec dependency: {name}')
        return matches[0]
    records = closure(resolve, inspect_library)
    # Validate every input and destination before changing the staged package.
    destinations = {name: checked_path(root, Path('usr/lib') / name) for name in records}
    manifest_path = checked_path(root, MANIFEST)
    if manifest_path.is_symlink() or any(path.exists() and path.is_dir() for path in destinations.values()) or manifest_path.is_dir():
        raise ValueError('Package file destination is a directory')
    manifest = {'schema': 1, 'root': ROOT, 'platform_dependencies': sorted(PLATFORM), 'libraries': {}}
    for name, info in sorted(records.items()):
        manifest['libraries'][name] = {key: value for key, value in info.items() if key != 'path'}
        manifest['libraries'][name]['sha256'] = digest(info['path'])
    for name, info in records.items():
        destination = destinations[name]
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.avif-', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with info['path'].open('rb') as source:
                    shutil.copyfileobj(source, stream)
                stream.flush()
                temporary.chmod(0o644)
                if digest(temporary) != manifest['libraries'][name]['sha256']:
                    raise ValueError(f'Codec input changed during copy: {name}')
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def verify(root, inspect_library=inspect):
    package_root(root)
    manifest_path = checked_path(root, MANIFEST)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Expected a regular AVIF dependency manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 1 or manifest['root'] != ROOT or manifest['platform_dependencies'] != sorted(PLATFORM):
        raise ValueError('Unexpected AVIF dependency manifest')
    def resolve(name):
        path = checked_path(root, Path('usr/lib') / name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing regular packaged codec: {name}')
        return path
    records = closure(resolve, inspect_library)
    if set(records) != set(manifest['libraries']):
        raise ValueError('AVIF dependency manifest does not match closure')
    for name, info in records.items():
        actual = {key: value for key, value in info.items() if key != 'path'}
        actual['sha256'] = digest(info['path'])
        if actual != manifest['libraries'][name]:
            raise ValueError(f'Packaged codec differs from manifest: {name}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('appdir', type=Path)
    parser.add_argument('--library-dir', action='append', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    manifest = verify(args.appdir) if args.verify else stage(args.appdir, args.library_dir or [Path('/usr/lib/x86_64-linux-gnu')])
    print(f"Verified AVIF codec chain: {len(manifest['libraries'])} libraries")
