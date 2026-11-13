#!/usr/bin/env python3
"""Build universal static media tools from pinned sources on the locked Mac worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
PINS = HERE / 'macos-media-sources.json'
ARCHS = ('arm64', 'x86_64')
TOOLS = ('ffmpeg', 'ffprobe')
PREFIX = '/DMC/helpers'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, cwd=None, env=None, capture=False, timeout=3600):
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None,
                            timeout=timeout)
    return result.stdout.strip() if capture else None


def owned_directory(path, identity):
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('Build directories must be absolute, regular directories')
    path.mkdir(parents=True, exist_ok=True)
    marker = path / '.dmc-media-build.json'
    if marker.is_symlink():
        raise ValueError('Refusing symlink build marker')
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('Build directory belongs to another recipe')
    elif any(path.iterdir()):
        raise ValueError('Refusing a nonempty unrecognized build directory')
    else:
        marker.write_text(json.dumps(identity, sort_keys=True) + '\n')


def download(pin, downloads):
    path = downloads / pin['url'].rsplit('/', 1)[1]
    if path.is_symlink():
        raise ValueError('Refusing symlink source archive')
    if not path.exists():
        partial = path.with_name(path.name + '.part')
        with partial.open('xb') as output:
            with urllib.request.urlopen(pin['url'], timeout=60) as source:
                shutil.copyfileobj(source, output)
        partial.rename(path)
    if digest(path) != pin['sha256']:
        raise ValueError('Source archive checksum mismatch: ' + path.name)
    return path


def extract(archive, pin, sources):
    target = sources / pin['root']
    marker = sources / (pin['root'] + '.sha256')
    if target.exists():
        if target.is_symlink() or not marker.is_file() or marker.read_text().strip() != pin['sha256']:
            raise ValueError('Unrecognized extracted source tree')
        return target
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or not name.parts
                    or name.parts[0] != pin['root'] or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe source archive member')
        source.extractall(sources, filter='data')
    marker.write_text(pin['sha256'] + '\n')
    return target


def validate_tool(path, archs=ARCHS):
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('Missing regular executable: ' + str(path))
    if set(run(['lipo', '-archs', path], capture=True).split()) != set(archs):
        raise ValueError('Media helper architecture mismatch')
    # Every slice must depend exclusively on Apple system libraries/frameworks.
    for arch in archs:
        lines = run(['otool', '-arch', arch, '-L', path], capture=True).splitlines()[1:]
        for line in lines:
            dependency = line.strip().split(' (', 1)[0]
            if not dependency.startswith(('/usr/lib/', '/System/Library/')):
                raise ValueError('Non-system helper dependency: ' + dependency)
        commands = run(['otool', '-arch', arch, '-l', path], capture=True)
        if 'LC_RPATH' in commands:
            raise ValueError('Unexpected media helper runtime search path')
        minimums = re.findall(r'^\s*(?:minos|version) (\d+\.\d+(?:\.\d+)?)\s*$', commands, re.M)
        if not minimums or any(tuple(map(int, value.split('.')[:2])) > (11, 0) for value in minimums):
            raise ValueError('Media helper exceeds the minimum macOS version')


def build(cache, output):
    if platform.system() != 'Darwin' or os.environ.get('DMC_MAC_BUILD_LOCK_HELD') != '1':
        raise ValueError('Native Mac shared build lock is required')
    if any(character.isspace() for character in str(cache)):
        raise ValueError('Choose a build cache path without whitespace for upstream configure tools')
    for tool in ('clang', 'make', 'lipo', 'otool', 'pkg-config', 'xcrun', 'strip'):
        if not shutil.which(tool):
            raise ValueError('Missing build prerequisite: ' + tool)
    pins = json.loads(PINS.read_text())
    recipe = hashlib.sha256(Path(__file__).read_bytes() + PINS.read_bytes()).hexdigest()
    owned_directory(cache, {'kind': 'dmc-macos-media-cache', 'schema': 1})
    if output.exists():
        raise ValueError('Use a fresh helper output directory')
    if not output.is_absolute() or output.is_symlink():
        raise ValueError('Output must be a new absolute directory')
    downloads = cache / 'downloads'; downloads.mkdir(exist_ok=True)
    work = cache / recipe; work.mkdir(exist_ok=True)
    sources = work / 'sources'; sources.mkdir(exist_ok=True)
    archives = {name: download(pin, downloads) for name, pin in pins.items()}
    trees = {name: extract(archives[name], pin, sources) for name, pin in pins.items()}
    environment = os.environ.copy()
    environment.update(MAKEFLAGS='-j1', CMAKE_BUILD_PARALLEL_LEVEL='1',
                       MACOSX_DEPLOYMENT_TARGET='11.0', LC_ALL='C',
                       SOURCE_DATE_EPOCH='1773705600')
    # NASM is a build-only dependency. Bootstrap locally; never install Homebrew packages.
    nasm = work / 'nasm-build'; nasm.mkdir(exist_ok=True)
    if not (nasm / 'nasm').is_file():
        run([os.path.relpath(trees['nasm'] / 'configure', nasm), '--prefix=/DMC/build-tools'],
            nasm, environment)
        run(['make', '-j1', 'nasm'], nasm, environment)
    environment['PATH'] = str(nasm) + os.pathsep + environment['PATH']
    versions = {name: run(command, capture=True).splitlines()[0] for name, command in {
        'clang': ['clang', '--version'], 'sdk': ['xcrun', '--show-sdk-version'],
        'nasm': [nasm / 'nasm', '-v']}.items()}
    for arch in ARCHS:
        area = work / arch; area.mkdir(exist_ok=True)
        stage = area / 'stage'; stage.mkdir(exist_ok=True)
        env = environment.copy()
        # Prefix maps sanitize __FILE__ and debug paths without embedding private paths
        # in FFmpeg's public configure argument string.
        flags = (f'-O2 -arch {arch} -mmacosx-version-min=11.0 '
                 f'-ffile-prefix-map={work}=/DMC/source -fdebug-prefix-map={work}=/DMC/source')
        env.update(CC='clang', CXX='clang++', CFLAGS=flags, CXXFLAGS=flags,
                   LDFLAGS=f'-arch {arch} -mmacosx-version-min=11.0',
                   PKG_CONFIG_PATH='', PKG_CONFIG_LIBDIR=str(stage / 'DMC/helpers/lib/pkgconfig'),
                   PKG_CONFIG_SYSROOT_DIR=str(stage))
        host = ('aarch64' if arch == 'arm64' else 'x86_64') + '-apple-darwin'
        x264 = area / 'x264'; x264.mkdir(exist_ok=True)
        run([os.path.relpath(trees['x264'] / 'configure', x264), '--prefix=' + PREFIX,
             '--host=' + host, '--enable-static', '--enable-pic', '--disable-cli',
             '--disable-opencl'], x264, env)
        run(['make', '-j1'], x264, env)
        run(['make', '-j1', 'install-lib-static', 'install-lib-dev', 'DESTDIR=' + str(stage)], x264, env)
        webp = area / 'webp'; webp.mkdir(exist_ok=True)
        run([os.path.relpath(trees['webp'] / 'configure', webp), '--prefix=' + PREFIX,
             '--host=' + host, '--disable-shared', '--enable-static', '--disable-dependency-tracking',
             '--disable-gl', '--disable-sdl', '--disable-png', '--disable-jpeg',
             '--disable-tiff', '--disable-gif'], webp, env)
        run(['make', '-j1'], webp, env)
        run(['make', '-j1', 'install', 'DESTDIR=' + str(stage)], webp, env)
        ffmpeg = area / 'ffmpeg'; ffmpeg.mkdir(exist_ok=True)
        configure = [os.path.relpath(trees['ffmpeg'] / 'configure', ffmpeg),
                     '--prefix=' + PREFIX, '--arch=' + ('aarch64' if arch == 'arm64' else arch),
                     '--target-os=darwin', '--enable-cross-compile', '--cc=clang', '--cxx=clang++',
                     '--disable-autodetect', '--disable-shared', '--enable-static',
                     '--enable-gpl', '--enable-version3', '--enable-libx264', '--enable-libwebp',
                     '--enable-videotoolbox', '--enable-audiotoolbox', '--enable-securetransport',
                     '--enable-zlib', '--enable-bzlib', '--enable-iconv', '--enable-pthreads',
                     '--pkg-config-flags=--static', '--disable-debug', '--disable-doc', '--disable-ffplay']
        run(configure, ffmpeg, env)
        run(['make', '-j1', *TOOLS], ffmpeg, env)
        for name in TOOLS:
            validate_tool(ffmpeg / name, (arch,))
    output.mkdir(parents=True)
    binaries = output / 'bin'; binaries.mkdir()
    licenses = output / 'licenses'; licenses.mkdir()
    for name in TOOLS:
        binary = binaries / name
        run(['lipo', '-create', *(work / arch / 'ffmpeg' / name for arch in ARCHS), '-output', binary])
        run(['strip', '-S', binary])
        binary.chmod(0o755)
        validate_tool(binary)
        data = binary.read_bytes()
        if str(cache).encode() in data or b'/opt/homebrew/' in data or b'/Users/' in data:
            raise ValueError('Private build path embedded in helper')
    encoders = run([binaries / 'ffmpeg', '-hide_banner', '-encoders'], capture=True)
    for encoder in ('libx264', 'libwebp', 'aac', 'png', 'mjpeg'):
        if not any(line.split()[1:2] == [encoder] for line in encoders.splitlines()):
            raise ValueError('Required media encoder absent: ' + encoder)
    for name, pin in pins.items():
        destination = licenses / name; destination.mkdir()
        for path in trees[name].iterdir():
            if path.is_file() and (path.name.startswith(('COPYING', 'LICENSE', 'PATENTS')) or path.name == 'AUTHORS'):
                shutil.copy2(path, destination / path.name)
        if not (destination / pin['license']).is_file():
            raise ValueError('Source license missing: ' + name)
    receipt = {'schema': 1, 'recipe_sha256': recipe, 'sources': pins, 'architectures': ARCHS,
               'minimum_macos': '11.0', 'toolchain': versions,
               'binaries': {name: digest(binaries / name) for name in TOOLS}}
    (output / 'media-build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Complete sources and exact build instructions accompany the package separately.
    with tarfile.open(output / 'DonutMediaCenter-macOS-media-sources.tar.xz', 'w:xz') as offer:
        for name, archive in archives.items():
            offer.add(archive, arcname='sources/' + archive.name)
        for relative in ('scripts/build-macos-media.py', 'scripts/macos-media-sources.json',
                         'scripts/macos-build-lock.py', 'scripts/package-macos-media.py',
                         'src-tauri/macos/media-helper.entitlements.plist',
                         'src-tauri/macos/store-lite.entitlements.plist', 'docs/MACOS_MEDIA_HELPERS.md'):
            offer.add(HERE.parent / relative, arcname=relative)
        offer.add(output / 'media-build-receipt.json', arcname='media-build-receipt.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        build(args.cache, args.output)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
