#!/usr/bin/env python3
"""Cross-build source-pinned Windows media helpers inside the release container."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PINS = HERE / 'windows-media-sources.json'
BASE_HELPER = HERE / 'build-macos-media.py'
DOCKERFILE = HERE.parent / 'Dockerfile.windows-media'
spec = importlib.util.spec_from_file_location('dmc_media_inputs', BASE_HELPER)
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)
digest, run, owned_directory = inputs.digest, inputs.run, inputs.owned_directory
TARGET = 'x86_64-w64-mingw32'
PREFIX = '/DMC/helpers'
TOOLS = ('ffmpeg.exe', 'ffprobe.exe')
REQUIRED = (
    'LIBX264_ENCODER', 'LIBWEBP_ENCODER', 'AAC_ENCODER', 'AAC_DECODER',
    'H264_DECODER', 'LIBDAV1D_DECODER', 'H264_NVENC_ENCODER',
    'H264_NVDEC_HWACCEL', 'SCALE_CUDA_FILTER', 'PNG_ENCODER', 'PNG_DECODER',
    'MJPEG_ENCODER', 'FLAC_DECODER', 'HLS_MUXER', 'MPEGTS_MUXER',
)
SYSTEM_DLLS = {
    'advapi32.dll', 'avrt.dll', 'bcrypt.dll', 'crypt32.dll', 'd3d11.dll',
    'd3d12.dll', 'dcomp.dll', 'dnsapi.dll', 'dwmapi.dll', 'dxgi.dll',
    'dxva2.dll', 'gdi32.dll', 'imm32.dll', 'kernel32.dll', 'mf.dll',
    'mfplat.dll', 'mfreadwrite.dll', 'msvcrt.dll', 'ntdll.dll',
    'ole32.dll', 'oleaut32.dll', 'psapi.dll', 'secur32.dll', 'setupapi.dll',
    'shell32.dll', 'shlwapi.dll', 'ucrtbase.dll', 'user32.dll', 'userenv.dll',
    'vfw32.dll', 'version.dll', 'winmm.dll', 'ws2_32.dll',
}


def filename(pin):
    name = pin.get('filename', pin['url'].rsplit('/', 1)[-1])
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Invalid archive basename')
    return name


def download(pin, directory):
    url = urlsplit(pin['url'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        raise ValueError('Source downloads require public HTTPS URLs')
    path = directory / filename(pin)
    if path.is_symlink():
        raise ValueError('Refusing a symlink archive')
    if not path.exists():
        partial = path.with_name(path.name + '.part')
        with partial.open('xb') as output, urllib.request.urlopen(pin['url'], timeout=90) as source:
            shutil.copyfileobj(source, output)
        partial.rename(path)
    if digest(path) != pin['sha256']:
        raise ValueError('Archive checksum mismatch: ' + path.name)
    return path


def members(archive, pin):
    if Path(pin['root']).name != pin['root'] or pin['root'] in ('', '.', '..'):
        raise ValueError('Invalid archive root')
    entries = archive.getmembers()
    seen = set()
    for entry in entries:
        name = PurePosixPath(entry.name)
        if (name.is_absolute() or '..' in name.parts or not name.parts
                or name.parts[0] != pin['root'] or name.as_posix() in seen
                or not (entry.isfile() or entry.isdir() or entry.issym())):
            raise ValueError('Unsafe or duplicate archive member')
        seen.add(name.as_posix())
        if entry.issym():
            target = PurePosixPath(entry.linkname)
            resolved = PurePosixPath(posixpath.normpath(str(name.parent / target)))
            if target.is_absolute() or not resolved.parts or resolved.parts[0] != pin['root']:
                raise ValueError('Archive symlink escapes its source tree')
    return entries


def verify_tree(path, pin, tree):
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError('Invalid cached source directory')
    expected = set()
    with tarfile.open(path) as archive:
        for entry in members(archive, pin):
            relative = PurePosixPath(entry.name).relative_to(pin['root'])
            if relative == PurePosixPath('.'):
                continue
            expected.add(relative.as_posix())
            expected.update(p.as_posix() for p in relative.parents if p != PurePosixPath('.'))
            location = tree / relative
            if any((tree / part).is_symlink() for part in relative.parents if part != PurePosixPath('.')):
                raise ValueError('Symlink replaced a cached source parent')
            if entry.issym():
                if not location.is_symlink() or os.readlink(location) != entry.linkname:
                    raise ValueError('Cached source symlink changed')
            elif location.is_symlink():
                raise ValueError('Symlink replaced a cached source file')
            elif entry.isdir():
                if not location.is_dir():
                    raise ValueError('Cached source directory missing')
            elif (not location.is_file() or location.stat().st_size != entry.size
                  or location.stat().st_mode & 0o111 != entry.mode & 0o111):
                raise ValueError('Cached source file changed')
            else:
                with archive.extractfile(entry) as original:
                    if digest(location) != hashlib.file_digest(original, 'sha256').hexdigest():
                        raise ValueError('Cached source contents changed')
    if {p.relative_to(tree).as_posix() for p in tree.rglob('*')} != expected:
        raise ValueError('Cached source contains added or missing entries')


def extract(path, pin, directory):
    if Path(pin['root']).name != pin['root'] or pin['root'] in ('', '.', '..'):
        raise ValueError('Invalid archive root')
    if digest(path) != pin['sha256']:
        raise ValueError('Archive checksum mismatch')
    tree = directory / pin['root']
    marker = directory / (pin['root'] + '.sha256')
    if tree.exists() or tree.is_symlink():
        if marker.is_symlink() or not marker.is_file() or marker.read_text().strip() != pin['sha256']:
            raise ValueError('Unrecognized cached source')
    else:
        with tarfile.open(path) as archive:
            members(archive, pin)
            archive.extractall(directory, filter='data')
        marker.write_text(pin['sha256'] + '\n')
    verify_tree(path, pin, tree)
    return tree


def recipe_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in (
        Path(__file__), PINS, BASE_HELPER, DOCKERFILE,
        HERE.parent / 'Dockerfile.windows-release'))).hexdigest()


def fresh_work_directory(cache, recipe):
    # Input trees are reusable; writable compiler products are not inputs.
    return Path(tempfile.mkdtemp(prefix='attempt-' + recipe[:16] + '-', dir=cache))


def check_features(text):
    if not re.search(r'^#define CONFIG_GPL 1$', text, re.M) or not re.search(
            r'^#define CONFIG_VERSION3 1$', text, re.M):
        raise ValueError('Expected GPLv3-compatible FFmpeg configuration')
    if re.search(r'^#define CONFIG_NONFREE 1$', text, re.M):
        raise ValueError('Nonfree configuration is forbidden')
    for component in REQUIRED:
        if not re.search(r'^#define CONFIG_' + component + r' 1$', text, re.M):
            raise ValueError('Required FFmpeg component missing: ' + component)
    return list(REQUIRED)


def validate_pe(path, readobj):
    text = run([readobj, '--file-headers', '--coff-imports', path], capture=True)
    if 'Format: COFF-x86-64' not in text:
        raise ValueError('Expected Windows x86-64 media helper')
    imports = re.findall(r'^\s*Name: ([^\s]+\.dll)\s*$', text, re.M | re.I)
    if not imports:
        raise ValueError('Media helper import inventory is empty')
    for name in imports:
        lower = name.lower()
        if lower not in SYSTEM_DLLS and not re.fullmatch(r'(api|ext)-ms-win-[a-z0-9-]+\.dll', lower):
            raise ValueError('Non-system DLL dependency: ' + name)
    return sorted(set(imports))


def verify_output(output, expected_recipe=None):
    record = output / 'media-build-receipt.json'
    if output.is_symlink() or record.is_symlink():
        raise ValueError('Refusing symlink media output')
    receipt = json.loads(record.read_text())
    if receipt['recipe_sha256'] != (expected_recipe or recipe_hash()):
        raise ValueError('Media output belongs to another recipe')
    for name in TOOLS:
        path = output / 'bin' / name
        if path.is_symlink() or not path.is_file() or digest(path) != receipt['binaries'][name]:
            raise ValueError('Media output hash mismatch')
    pins = json.loads(PINS.read_text())
    if (receipt['sources'] != pins['sources'] or receipt['toolchain'] != pins['toolchain']
            or receipt['target'] != TARGET or set(receipt['required_components']) != set(REQUIRED)):
        raise ValueError('Media source receipt does not match the pins')
    if receipt['environment_sha256'] != hashlib.sha256(
            json.dumps(receipt['tool_versions'], sort_keys=True).encode()).hexdigest():
        raise ValueError('Media tool-version receipt changed')
    licenses = output / 'licenses'
    if licenses.is_symlink() or any(p.is_symlink() for p in licenses.rglob('*')):
        raise ValueError('Refusing symlink notices')
    actual_notices = {p.relative_to(licenses).as_posix(): digest(p)
                      for p in licenses.rglob('*') if p.is_file()}
    if not actual_notices or receipt['licenses'] != actual_notices:
        raise ValueError('Media notices changed or are missing')
    name = receipt['source_archive']
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Invalid source archive basename')
    source = output / name
    if source.is_symlink() or digest(source) != receipt['source_archive_sha256']:
        raise ValueError('Media source offer hash mismatch')
    with tarfile.open(source) as archive:
        for key, pin in pins['sources'].items():
            member = archive.getmember('sources/' + filename(pin))
            if not member.isfile():
                raise ValueError('Source offer contains a non-file archive')
            with archive.extractfile(member) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != pin['sha256']:
                    raise ValueError('Source offer input checksum mismatch: ' + key)
        for name, path in recipe_files().items():
            member = archive.getmember(name)
            if not member.isfile() or archive.extractfile(member).read() != path.read_bytes():
                raise ValueError('Source offer recipe does not match: ' + name)
    return receipt


def recipe_files():
    return {
        'scripts/build-windows-media.py': Path(__file__),
        'scripts/windows-media-sources.json': PINS,
        'scripts/build-macos-media.py': BASE_HELPER,
        'Dockerfile.windows-media': DOCKERFILE,
        'Dockerfile.windows-release': HERE.parent / 'Dockerfile.windows-release',
        'docs/WINDOWS_MEDIA_HELPERS.md': HERE.parent / 'docs/WINDOWS_MEDIA_HELPERS.md',
        'LICENSE': HERE.parent / 'LICENSE', 'NOTICE': HERE.parent / 'NOTICE',
        'LICENSES/MIT-legacy.txt': HERE.parent / 'LICENSES/MIT-legacy.txt',
    }


def stage_output(output, source_root):
    """Stage verified tools and their source offer into a fresh release tree."""
    receipt = verify_output(output)
    helpers = source_root / 'src-tauri/ffmpeg-bin'
    notices = source_root / 'LICENSES/ffmpeg'
    for destination in (helpers, notices):
        if destination.exists() or destination.is_symlink():
            raise ValueError('Media staging needs fresh release destinations')
    helpers.mkdir(parents=True)
    shutil.copytree(output / 'licenses', notices)
    for name in TOOLS:
        shutil.copy2(output / 'bin' / name, helpers / name)
    for name in (receipt['source_archive'], 'media-build-receipt.json'):
        shutil.copy2(output / name, notices / name)
    (notices / 'SOURCE.txt').write_text(
        'FFmpeg and codec inputs are pinned in the accompanying source archive.\n'
        'media-build-receipt.json records the source pins, recipe, configuration,\n'
        'tool versions, PE imports and helper/source hashes. It is unsigned\n'
        'build provenance, not Windows runtime or independent attestation.\n')
    return receipt


def verify_packages(output, portable, installed):
    receipt = verify_output(output)
    prefix = 'licenses/LICENSES/ffmpeg/'
    payloads = {'bin/' + name: output / 'bin' / name for name in TOOLS}
    payloads.update({prefix + name: output / name
                     for name in (receipt['source_archive'], 'media-build-receipt.json')})
    payloads.update({prefix + name: output / 'licenses' / name for name in receipt['licenses']})
    for relative, expected in payloads.items():
        actual = portable / (relative.removeprefix('licenses/'))
        matches = [p for p in installed.rglob(Path(relative).name)
                   if p.as_posix().lower().endswith('/' + relative.lower())]
        if len(matches) != 1 or any(p.is_symlink() for p in (actual, *matches)):
            raise ValueError('Missing, duplicate or symlink packaged media: ' + relative)
        if digest(expected) != digest(actual) or digest(expected) != digest(matches[0]):
            raise ValueError('Packaged media hash mismatch: ' + relative)
    return receipt


def build(cache, output):
    if (platform.system() != 'Linux' or platform.machine() != 'x86_64'
            or not Path('/.dockerenv').exists()
            or os.environ.get('DMC_WINDOWS_MEDIA_BUILD_LOCKED') != '1'):
        raise ValueError('Use the canonical gated Windows release container')
    if any(c.isspace() for c in str(cache)):
        raise ValueError('Upstream configure tools require a cache path without whitespace')
    for tool in ('make', 'pkg-config', 'ninja', '/usr/bin/clang'):
        if not shutil.which(tool):
            raise ValueError('Missing build prerequisite: ' + tool)
    if 'nvptx' not in run(['/usr/bin/clang', '--print-targets'], capture=True):
        raise ValueError('Host clang needs the NVPTX target for scale_cuda')
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError('Use a fresh absolute output directory')
    owned_directory(cache, {'kind': 'dmc-windows-media-cache', 'schema': 1})
    pins = json.loads(PINS.read_text())
    downloads = cache / 'downloads'; downloads.mkdir(exist_ok=True)
    sources = cache / 'sources'; sources.mkdir(exist_ok=True)
    tool_archive = download(pins['toolchain'], downloads)
    compiler = extract(tool_archive, pins['toolchain'], sources)
    archives = {key: download(pin, downloads) for key, pin in pins['sources'].items()}
    trees = {key: extract(archives[key], pin, sources) for key, pin in pins['sources'].items()}
    versions = {name: run(command, capture=True).splitlines()[0] for name, command in {
        'target_clang': [compiler / 'bin' / (TARGET + '-clang'), '--version'],
        'host_clang': ['/usr/bin/clang', '--version'], 'ninja': ['ninja', '--version'],
        'pkg_config': ['pkg-config', '--version']}.items()}
    recipe = recipe_hash()
    work = fresh_work_directory(cache, recipe)
    env = os.environ.copy()
    env.pop('GIT_DIR', None)
    env.pop('GIT_WORK_TREE', None)
    env.update(MAKEFLAGS='-j1', LC_ALL='C', SOURCE_DATE_EPOCH='1773705600',
               PYTHONDONTWRITEBYTECODE='1',
               CC='/usr/bin/clang', CXX='/usr/bin/clang++',
               CFLAGS='-O2', CXXFLAGS='-O2', LDFLAGS='')
    nasm = work / 'nasm'; nasm.mkdir(exist_ok=True)
    for directory in trees['nasm'].rglob('*'):
        if directory.is_dir():
            (nasm / directory.relative_to(trees['nasm'])).mkdir(parents=True, exist_ok=True)
    run([os.path.relpath(trees['nasm'] / 'configure', nasm), '--prefix=/DMC/build-tools'], nasm, env)
    run(['make', '-j1', 'nasm'], nasm, env)
    wrappers = work / 'tools'; wrappers.mkdir(exist_ok=True)
    cuda = wrappers / 'host-cuda-clang'
    maps = [f'-ffile-prefix-map={cache}=/DMC/source', f'-fdebug-prefix-map={cache}=/DMC/source']
    cuda.write_text('#!/bin/sh\nexec /usr/bin/clang ' + shlex.join(maps) + ' "$@"\n')
    cuda.chmod(0o755)
    env['PATH'] = os.pathsep.join((str(wrappers), str(compiler / 'bin'), str(nasm), env['PATH']))
    versions['nasm'] = run([nasm / 'nasm', '-v'], capture=True)
    environment_hash = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    stage = work / 'stage'; stage.mkdir(exist_ok=True)
    cflags = '-O2 ' + ' '.join(maps)
    env.update(CC=TARGET + '-clang', CXX=TARGET + '-clang++', AR='llvm-ar', RANLIB='llvm-ranlib',
               CFLAGS=cflags, CXXFLAGS=cflags, CHOST=TARGET,
               LDFLAGS='-static -static-libgcc -static-libstdc++ -Wl,--no-insert-timestamp',
               PKG_CONFIG_PATH='', PKG_CONFIG_LIBDIR=str(stage / 'DMC/helpers/lib/pkgconfig'),
               PKG_CONFIG_SYSROOT_DIR=str(stage))
    for key, arguments, install in (
        ('zlib', ['--static'], ['install']),
        ('x264', ['--host=' + TARGET, '--enable-static', '--disable-cli', '--disable-opencl'],
         ['install-lib-static', 'install-lib-dev']),
        ('webp', ['--host=' + TARGET, '--disable-shared', '--enable-static',
                  '--disable-dependency-tracking', '--disable-gl', '--disable-sdl',
                  '--disable-png', '--disable-jpeg', '--disable-tiff', '--disable-gif'], ['install']),
    ):
        area = work / key; area.mkdir(exist_ok=True)
        run([os.path.relpath(trees[key] / 'configure', area), '--prefix=' + PREFIX, *arguments], area, env)
        run(['make', '-j1'], area, env)
        run(['make', '-j1', *install, 'DESTDIR=' + str(stage)], area, env)
    cross = work / 'windows-cross.ini'
    cross.write_text(
        "[binaries]\nc = '" + TARGET + "-clang'\ncpp = '" + TARGET + "-clang++'\n"
        "ar = 'llvm-ar'\nstrip = 'llvm-strip'\nwindres = 'llvm-windres'\npkg-config = 'pkg-config'\n"
        "[host_machine]\nsystem = 'windows'\ncpu_family = 'x86_64'\ncpu = 'x86_64'\nendian = 'little'\n"
        "[properties]\nneeds_exe_wrapper = true\n")
    dav1d = work / 'dav1d'
    meson = [sys.executable, trees['meson'] / 'meson.py']
    configure = [*meson, 'setup', dav1d, trees['dav1d'], '--cross-file', cross,
                 '--prefix=' + PREFIX, '--libdir=lib', '--buildtype=release',
                 '--default-library=static', '-Denable_tools=false', '-Denable_tests=false',
                 '-Dc_args=' + repr(cflags.split()), '-Dc_link_args=' + repr(env['LDFLAGS'].split())]
    if (dav1d / 'meson-private/coredata.dat').exists():
        configure.append('--reconfigure')
    run(configure, work, env)
    run(['ninja', '-C', dav1d, '-j1'], work, env)
    install_env = env.copy(); install_env['DESTDIR'] = str(stage)
    run([*meson, 'install', '-C', dav1d, '--no-rebuild'], work, install_env)
    # Header-only input: the upstream install target modifies its source tree.
    headers = stage / 'DMC/helpers/include/ffnvcodec'
    headers.mkdir(parents=True, exist_ok=True)
    for path in (trees['ffnvcodec'] / 'include/ffnvcodec').glob('*.h'):
        shutil.copy2(path, headers / path.name)
    pc = (trees['ffnvcodec'] / 'ffnvcodec.pc.in').read_text()
    if '@@PREFIX@@' not in pc:
        raise ValueError('Unknown nv-codec-headers pkg-config template')
    (stage / 'DMC/helpers/lib/pkgconfig/ffnvcodec.pc').write_text(pc.replace('@@PREFIX@@', PREFIX))
    ffmpeg = work / 'ffmpeg'; ffmpeg.mkdir(exist_ok=True)
    configure = [os.path.relpath(trees['ffmpeg'] / 'configure', ffmpeg), '--prefix=' + PREFIX,
                 '--arch=x86_64', '--target-os=mingw32', '--enable-cross-compile',
                 '--cc=' + TARGET + '-clang', '--cxx=' + TARGET + '-clang++',
                 '--ar=llvm-ar', '--ranlib=llvm-ranlib', '--nm=llvm-nm',
                 '--strip=llvm-strip', '--windres=llvm-windres', '--nvcc=host-cuda-clang',
                 '--disable-autodetect', '--disable-shared', '--enable-static',
                 '--enable-gpl', '--enable-version3', '--disable-nonfree',
                 '--enable-libx264', '--enable-libwebp', '--enable-libdav1d', '--enable-zlib',
                 '--enable-w32threads', '--disable-pthreads', '--enable-schannel',
                 '--enable-dxva2', '--enable-d3d11va', '--enable-d3d12va',
                 '--enable-mediafoundation', '--enable-ffnvcodec', '--enable-cuda-llvm',
                 '--enable-nvenc', '--enable-nvdec', '--enable-cuvid',
                 '--pkg-config-flags=--static', '--disable-debug', '--disable-doc', '--disable-ffplay']
    run(configure, ffmpeg, env)
    run(['make', '-j1', *TOOLS], ffmpeg, env)
    features = check_features((ffmpeg / 'config.h').read_text() + '\n' +
                              (ffmpeg / 'config_components.h').read_text())
    output.mkdir(parents=True)
    binaries = output / 'bin'; binaries.mkdir()
    imports = {}
    for name in TOOLS:
        destination = binaries / name
        shutil.copy2(ffmpeg / name, destination)
        run([compiler / 'bin/llvm-strip', '--strip-all', destination])
        imports[name] = validate_pe(destination, compiler / 'bin/llvm-readobj')
        if str(cache).encode() in destination.read_bytes():
            raise ValueError('Private build path embedded in helper')
    licenses = output / 'licenses'; licenses.mkdir()
    for key, pin in pins['sources'].items():
        verify_tree(archives[key], pin, trees[key])
        destination = licenses / key; destination.mkdir()
        for source in trees[key].iterdir():
            if source.is_file() and source.name.startswith(('COPYING', 'LICENSE', 'PATENTS', 'AUTHORS')):
                shutil.copy2(source, destination / source.name)
        target = destination / pin['license']
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trees[key] / pin['license'], target)
    verify_tree(tool_archive, pins['toolchain'], compiler)
    shutil.copy2(compiler / 'LICENSE.TXT', licenses / 'LLVM-LICENSE.TXT')
    runtime_notices = licenses / 'mingw-runtime'; runtime_notices.mkdir()
    for source in (compiler / TARGET / 'share/mingw32').glob('COPYING*'):
        shutil.copy2(source, runtime_notices / source.name)
    source_name = 'DonutMediaCenter-Windows-media-sources.tar.xz'
    with tarfile.open(output / source_name, 'w:xz') as archive:
        for key, path in archives.items():
            archive.add(path, arcname='sources/' + path.name)
        for name, path in recipe_files().items():
            archive.add(path, arcname=name)
    receipt = {'schema': 1, 'target': TARGET, 'recipe_sha256': recipe,
               'environment_sha256': environment_hash, 'toolchain': pins['toolchain'],
               'tool_versions': versions, 'sources': pins['sources'],
               'required_components': features, 'system_dll_imports': imports,
               'configuration': configure, 'windows_runtime_acceptance': 'not_run',
               'source_archive': source_name, 'source_archive_sha256': digest(output / source_name),
               'licenses': {p.relative_to(licenses).as_posix(): digest(p)
                            for p in licenses.rglob('*') if p.is_file()},
               'binaries': {name: digest(binaries / name) for name in TOOLS}}
    (output / 'media-build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    verify_output(output, recipe)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-output', action='store_true')
    parser.add_argument('--stage-into', type=Path)
    parser.add_argument('--portable', type=Path)
    parser.add_argument('--installed', type=Path)
    args = parser.parse_args()
    try:
        if args.portable or args.installed:
            if not args.portable or not args.installed or args.stage_into or args.cache or args.check_output:
                parser.error('Package verification needs only --portable and --installed')
            result = verify_packages(args.output, args.portable, args.installed)
        elif args.stage_into:
            if args.cache or args.check_output:
                parser.error('Staging does not accept --cache or --check-output')
            result = stage_output(args.output, args.stage_into)
        elif args.check_output:
            result = verify_output(args.output)
        else:
            if args.cache is None:
                parser.error('Building requires --cache')
            result = build(args.cache, args.output)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, str(error) + '\n')
