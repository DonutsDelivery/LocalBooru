#!/usr/bin/env python3
"""Build Microsoft's pinned native packager under build-msix-tools-local.sh's gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

COMMIT = '25a65f5c1690930813bcc10cdf1d59fa865f2bb1'
ARCHIVE_SHA256 = 'a9c28df8f9bad329b811c0fae6ddf7870aa866327c3836a39c3966064d0a549b'
URL = 'https://codeload.github.com/microsoft/msix-packaging/tar.gz/' + COMMIT
MARKER = 'dmc-native-msix-tools-v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_archive(path):
    if digest(path) != ARCHIVE_SHA256:
        raise ValueError('Microsoft MSIX source archive checksum does not match the pinned source')


def prepare_cache(path):
    path = Path(path).absolute()
    if path == Path('/') or path == Path.home() or path == Path.cwd():
        raise ValueError('Choose a dedicated tool cache')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Tool cache must not contain a symlink component')
    marker = path / '.dmc-msix-tools-cache'
    if path.exists() and any(path.iterdir()) and (not marker.is_file() or marker.read_text() != MARKER):
        raise ValueError('Refusing an unrecognized nonempty tool cache')
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text(MARKER)
    return path


def build(cache, archive=None):
    os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = '1'
    os.environ['MAKEFLAGS'] = '-j1'
    cache = prepare_cache(cache)
    source_archive = cache / (COMMIT + '.tar.gz')
    if archive:
        verify_archive(archive)
        if Path(archive).resolve() != source_archive.resolve(): shutil.copyfile(archive, source_archive)
    if not source_archive.exists():
        partial = cache / 'source-download.part'
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            verify_archive(partial)
            partial.replace(source_archive)
        finally:
            partial.unlink(missing_ok=True)
    verify_archive(source_archive)
    source = cache / ('msix-packaging-' + COMMIT)
    if not source.exists():
        with tarfile.open(source_archive) as archive_file:
            expected = source.name + '/'
            if any(not (m.name == source.name or m.name.startswith(expected)) for m in archive_file.getmembers()):
                raise ValueError('Unexpected source archive layout')
            archive_file.extractall(cache, filter='data')
    if not (source / 'CMakeLists.txt').is_file():
        raise ValueError('Pinned MSIX source is incomplete')
    # Newer system ICU headers require C++17. Keep the compatibility change
    # bounded to the verified upstream file instead of accepting cache edits.
    with tarfile.open(source_archive) as archive_file:
        original = archive_file.extractfile(source.name + '/CMakeLists.txt').read()
    token = b'set(CMAKE_CXX_STANDARD 14)'
    if original.count(token) != 1:
        raise ValueError('Pinned SDK C++ standard declaration changed')
    compatible = original.replace(token, b'set(CMAKE_CXX_STANDARD 17)')
    project = source / 'CMakeLists.txt'
    if project.read_bytes() not in (original, compatible):
        raise ValueError('Unrecognized local MSIX source edits')
    project.write_bytes(compatible)
    target = cache / ('build-' + COMMIT)
    command = ['cmake', '-S', str(source), '-B', str(target), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DLINUX=on',
               '-DCMAKE_TOOLCHAIN_FILE=' + str(source / 'cmake/linux.cmake'),
               '-DMSIX_PACK=on', '-DUSE_VALIDATION_PARSER=on',
               '-DMSIX_SAMPLES=off', '-DMSIX_TESTS=off']
    subprocess.run(command, check=True, timeout=300)
    subprocess.run(['cmake', '--build', str(target), '--target', 'makemsix', '--parallel', '1'],
                   check=True, timeout=1800)
    binary = target / 'bin/makemsix'
    help_text = subprocess.check_output([str(binary), '-?'], text=True, stderr=subprocess.STDOUT, timeout=15)
    if 'pack' not in help_text or 'unpack' not in help_text:
        raise ValueError('Native tool was built without packaging support')
    receipt = {'source_commit': COMMIT, 'source_archive_sha256': ARCHIVE_SHA256,
               'tool_sha256': digest(binary), 'bootstrap_sha256': digest(Path(__file__)),
               'validation_parser': True, 'pack_enabled': True}
    (cache / 'native-msix-tool-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Native Microsoft packager:', binary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--archive')
    args = parser.parse_args()
    try:
        build(args.cache, args.archive)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, 'Native MSIX tool build failed: ' + str(error) + '\n')
