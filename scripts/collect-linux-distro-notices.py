#!/usr/bin/env python3
"""Retain distro notices for package candidates found in a staged ELF payload."""
import argparse
from collections import defaultdict
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import struct


def public_file(path, system_root):
    try:
        target = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f'Missing notice: {path.name} for {path.parent.name}') from error
    allowed = [system_root / 'usr/share/doc', system_root / 'usr/share/common-licenses']
    if not any(target.is_relative_to(root) for root in allowed) or not target.is_file():
        raise ValueError(f'Notice for {path.parent.name} escapes public license directories')
    return target.read_bytes()


def dynamic_elf(path):
    with path.open('rb') as stream:
        header = stream.read(20)
    return (len(header) == 20 and header[:4] == b'\x7fELF' and header[5] in {1, 2}
            and struct.unpack('<H' if header[5] == 1 else '>H', header[16:18])[0] == 3)


def collect(appdir, system_root):
    appdir, system_root = Path(appdir).resolve(strict=True), Path(system_root).resolve(strict=True)
    packages = {}
    for paragraph in (system_root / 'var/lib/dpkg/status').read_text().split('\n\n'):
        item = Parser().parsestr(paragraph)
        if item.get('Status') != 'install ok installed':
            continue
        name = item['Package']
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', name):
            raise ValueError('Invalid distro package name')
        source = re.fullmatch(r'([^ ]+)(?: \(([^)]+)\))?', item.get('Source', name))
        if not source:
            raise ValueError(f'Invalid source metadata for {name}')
        packages[name] = dict(binary_package=name, binary_version=item['Version'],
                              source_package=source[1], source_version=source[2] or item['Version'])
    owners = defaultdict(set)
    for listing in sorted((system_root / 'var/lib/dpkg/info').glob('*.list')):
        name = listing.name[:-5].split(':', 1)[0]
        if name not in packages:
            continue
        if listing.is_symlink():
            raise ValueError(f'Unexpected package metadata symlink for {name}')
        for entry in listing.read_text().splitlines():
            if entry.startswith('/'):
                owners[Path(entry).name].add(name)

    used, separate = defaultdict(set), []
    for path in sorted(appdir.rglob('*')):
        if path.is_symlink() or not path.is_file() or not dynamic_elf(path):
            continue
        relative = path.relative_to(appdir).as_posix()
        candidates = owners[path.name]
        if candidates:
            for name in candidates:
                used[name].add(relative)
        elif (relative in {'usr/bin/donutmediacenter', 'usr/lib/localbooru/donutmediacenter'}
              or relative.startswith('usr/lib/localbooru/native-svp/')):
            separate.append(relative)
        else:
            raise ValueError(f'No distro notice owner for {relative}')

    contents, records = {}, []
    for name in sorted(used):
        relative = f'packages/{name}/copyright'
        data = public_file(system_root / 'usr/share/doc' / name / 'copyright', system_root)
        contents[relative] = data
        records.append(dict(**packages[name], candidate_files=sorted(used[name]),
                            copyright=dict(path=relative, sha256=hashlib.sha256(data).hexdigest())))
    common = []
    common_root = system_root / 'usr/share/common-licenses'
    if common_root.resolve(strict=True) != common_root:
        raise ValueError('Common license directory escapes public license root')
    for path in sorted(common_root.iterdir()):
        data = public_file(path, system_root)
        relative = 'common-licenses/' + path.name
        contents[relative] = data
        common.append(dict(path=relative, sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(schema=1, scope='Distro package candidate notices by ELF basename ownership; not binary provenance or legal clearance.',
                    packages=records, common_licenses=common, non_distro_files=sorted(separate))
    output = appdir / 'usr/share/doc/localbooru/distro-packages'
    if not output.resolve().is_relative_to(appdir):
        raise ValueError('Distro notice output escapes staging directory')
    if output.exists() or output.is_symlink():
        raise ValueError('Distro notice output already exists; stage a fresh payload')
    # Validate every input before creating any output. Copies are regular files,
    # including notices whose public distro sources were symlinks.
    for relative, data in sorted(contents.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def verify(appdir):
    appdir = Path(appdir).resolve()
    output = appdir / 'usr/share/doc/localbooru/distro-packages'
    if output.is_symlink() or not output.resolve(strict=True).is_relative_to(appdir):
        raise ValueError('Distro notice output escapes staging directory')
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported distro notice manifest')
    entries = [item['copyright'] for item in manifest['packages']] + manifest['common_licenses']
    for entry in entries:
        path = output / entry['path']
        if path.is_symlink() or not path.resolve().is_relative_to(output) or not path.is_file():
            raise ValueError('Distro notice file is missing or redirected')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Distro notice content hash differs')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('appdir', type=Path)
    parser.add_argument('--system-root', type=Path, default=Path('/'), help='Synthetic root for fixture tests; production runs inside the build image.')
    parser.add_argument('--verify', action='store_true', help='Verify notice files retained in an already staged payload.')
    args = parser.parse_args()
    try:
        manifest = verify(args.appdir) if args.verify else collect(args.appdir, args.system_root)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Distro notice retention failed: {error}\n')
    print(f'Retained notices for {len(manifest["packages"])} distro package candidates')
