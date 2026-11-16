#!/usr/bin/env python3
"""Keep the AppImage on the platform Wayland client used by host graphics."""
import argparse
import json
from pathlib import Path
import re
import stat

RECEIPT = Path('usr/share/doc/localbooru/wayland-client-policy.json')
POLICY = 'Use the platform Wayland client for host graphics; do not bundle it.'
PATTERN = re.compile(r'libwayland-client\.so(?:\.[0-9]+)*\Z')


def apply(root, verify=False):
    if not root.is_absolute() or root == Path('/') or root.resolve() != root or not root.is_dir():
        raise ValueError('Expected an absolute, unredirected AppDir')
    if not (root / 'AppRun').is_file() or (root / 'AppRun').is_symlink():
        raise ValueError('Missing AppRun')
    lib = root / 'usr/lib'
    receipt = root / RECEIPT
    for directory in (lib, receipt.parent):
        if not directory.is_dir() or directory.resolve() != directory:
            raise ValueError(f'Missing or redirected directory: {directory}')
    if receipt.is_symlink() or (receipt.exists() and not receipt.is_file()):
        raise ValueError('Invalid receipt destination')
    entries = sorted(p for p in lib.iterdir() if PATTERN.fullmatch(p.name))
    for path in entries:
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise ValueError(f'Invalid library entry: {path}')
        if path.is_symlink() and not path.resolve().is_relative_to(lib):
            raise ValueError(f'Redirected library alias: {path}')
    if verify:
        if entries:
            raise ValueError('Bundled Wayland client retained')
        if not receipt.is_file():
            raise ValueError('Missing Wayland client policy receipt')
        data = json.loads(receipt.read_text())
        if data.get('schema') != 1 or data.get('policy') != POLICY or not isinstance(data.get('removed'), list) or any(not isinstance(n, str) or not PATTERN.fullmatch(n) for n in data['removed']):
            raise ValueError('Invalid Wayland client policy receipt')
        return data
    data = {'schema': 1, 'policy': POLICY, 'removed': [p.name for p in entries]}
    # A failed receipt write must leave all libraries intact. Verification also
    # rejects a partial removal if a later unlink fails.
    receipt.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    for path in entries:
        path.unlink()
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('appdir', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(apply(args.appdir, args.verify), sort_keys=True))
