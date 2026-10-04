#!/usr/bin/env python3
"""Trim oversized Rust compiler caches while preserving finished outputs.

Call only while holding the host heavy-build gate and with no compiler writing
this target. Native runtime/toolchain caches and release packages are excluded.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

COMPILER_DIRS = ('deps', 'build', '.fingerprint', 'incremental', 'examples')


def trim(root, limit):
    root = root.resolve()
    if root in (Path('/'), Path.home().resolve(), Path.cwd().resolve()):
        raise ValueError(f'Refusing unsafe Cargo cache root: {root}')
    if not root.is_dir():
        return
    profiles = [root/'debug', root/'release']
    profiles += [child/profile for child in root.iterdir() if child.is_dir() and not child.is_symlink()
                 and child.name not in ('debug', 'release') for profile in ('debug', 'release')]
    paths = [profile/name for profile in profiles
             if (profile/'.fingerprint').is_dir() and not profile.is_symlink()
             and not (profile/'.fingerprint').is_symlink()
             for name in COMPILER_DIRS if (profile/name).is_dir() and not (profile/name).is_symlink()]
    if not paths:
        return
    if not (root/'.rustc_info.json').is_file():
        raise ValueError(f'Refusing unrecognized Rust target cache: {root}')
    size = int(subprocess.check_output(['du', '-sk', '-c', *map(str, paths)], text=True).splitlines()[-1].split()[0]) * 1024
    if size <= limit:
        return
    print(f'Cargo cache hygiene: trimming {size / 1024**3:.1f} GiB compiler files in {root} '
          f'(budget {limit / 1024**3:.1f} GiB); keeping binaries, bundles and packages.', file=sys.stderr)
    for path in paths:
        shutil.rmtree(path)


def main():
    budget = os.environ.get('LOCALBOORU_CARGO_CACHE_LIMIT_GB', '20')
    if not re.fullmatch(r'[1-9][0-9]{0,4}', budget):
        raise ValueError('LOCALBOORU_CARGO_CACHE_LIMIT_GB must be a positive integer (at most 99999)')
    args = sys.argv[1:]
    source_args = args[:args.index('--')] if '--' in args else args
    if args[:1] == ['--target-directory']:
        if len(args) != 2:
            raise ValueError('Expected --target-directory PATH')
        root = Path(args[1])
    else:
        # Cargo itself resolves global/project configuration and CARGO_TARGET_DIR.
        command = ['cargo', 'metadata', '--no-deps', '--format-version=1', '--locked']
        for index, arg in enumerate(source_args):
            if arg == '--manifest-path':
                if index+1 == len(source_args):
                    raise ValueError('--manifest-path requires a path')
                command += [arg, source_args[index+1]]
            elif arg.startswith('--manifest-path='):
                command.append(arg)
            elif source_args[:1] != ['tauri'] and arg == '--config':
                if index+1 == len(source_args):
                    raise ValueError('--config requires a value')
                command += [arg, source_args[index+1]]
            elif source_args[:1] != ['tauri'] and arg.startswith('--config='):
                command.append(arg)
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise ValueError('Cannot resolve Cargo target for cache hygiene: '+result.stderr.strip())
        root = Path(json.loads(result.stdout)['target_directory'])
        for index, arg in enumerate(source_args):
            if arg == '--target-dir':
                if index+1 == len(source_args):
                    raise ValueError('--target-dir requires a path')
                root = Path(source_args[index+1])
            elif arg.startswith('--target-dir='):
                root = Path(arg.split('=', 1)[1])
    trim(root, int(budget) * 1024**3)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit('ERROR: '+str(error))
