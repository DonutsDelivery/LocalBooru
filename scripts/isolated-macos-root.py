#!/usr/bin/env python3
"""Own one disposable test root inside a verified Mac Lite container."""
import argparse
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import tempfile


def parent_for(app, parent, home):
    if not app.is_absolute() or '\n' in str(parent) or '\r' in str(parent):
        raise ValueError('Use absolute candidate paths without line breaks')
    with (app / 'Contents/Info.plist').open('rb') as stream:
        identifier = plistlib.load(stream).get('CFBundleIdentifier')
    if identifier != 'com.localbooru.app.lite':
        raise ValueError('Container acceptance requires the Mac Lite candidate')
    expected = home.resolve() / 'Library/Containers' / identifier / 'Data/Library/Application Support'
    if not parent.is_absolute() or parent != expected or parent.resolve() != parent:
        raise ValueError('Use the verified Lite container Application Support directory')
    info = parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError('Container parent must be an owned directory without shared write access')


def root_identity(root):
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('Disposable root ownership or permissions changed; preserving it')
    return f'{info.st_dev}:{info.st_ino}'


def create(app, parent, home):
    parent_for(app, parent, home)
    root = Path(tempfile.mkdtemp(prefix='localbooru-isolated.', dir=parent))
    return root, root_identity(root)


def cleanup(app, parent, home, root, identity):
    parent_for(app, parent, home)
    if root.parent != parent or not re.fullmatch(r'localbooru-isolated\.[A-Za-z0-9_-]+', root.name):
        raise ValueError('Refusing cleanup outside the disposable root')
    if root_identity(root) != identity:
        raise ValueError('Disposable root identity changed; preserving it')
    shutil.rmtree(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'cleanup'))
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--identity')
    args = parser.parse_args()
    try:
        if args.action == 'create':
            root, identity = create(args.app, args.parent, Path.home())
            print(root)
            print(identity)
        else:
            if not args.root or not args.identity:
                raise ValueError('Cleanup requires the owned root and identity')
            cleanup(args.app, args.parent, Path.home(), args.root, args.identity)
    except (OSError, ValueError, plistlib.InvalidFileException) as error:
        parser.exit(1, str(error) + '\n')
