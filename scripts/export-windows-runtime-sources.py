#!/usr/bin/env python3
"""Export matching runtime sources separately from small packaged notices."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
PINS = HERE / 'windows-runtime-sources.json'
DOC = HERE.parent / 'docs/WINDOWS_RUNTIME_SOURCES.md'
ARCHIVE = 'DonutMediaCenter-Windows-runtime-sources.tar.xz'
RECEIPT = 'runtime-source-receipt.json'
spec = importlib.util.spec_from_file_location('windows_media', HERE / 'build-windows-media.py')
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)
digest = media.digest


def no_redirect(path):
    if not path.is_absolute():
        raise ValueError('Use an absolute source-offer path')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Refusing redirected source-offer paths')


def recipe_files():
    return {'scripts/export-windows-runtime-sources.py': Path(__file__),
            'scripts/windows-runtime-sources.json': PINS,
            'docs/WINDOWS_RUNTIME_SOURCES.md': DOC}


def recipe_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in recipe_files().values())).hexdigest()


def inputs(receipt_path):
    no_redirect(receipt_path)
    pins = json.loads(PINS.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (pins['schema'] != 1 or set(pins['sources']) != {'llvm-project', 'mingw-w64', 'llvm-mingw'}
            or receipt['toolchain']['sha256'] != pins['toolchain_sha256']
            or receipt['target'] != media.TARGET):
        raise ValueError('Runtime sources do not match the media toolchain')
    commit = pins['sources']['llvm-project']['commit']
    if not re.fullmatch('[a-f0-9]{40}', commit) or not re.search(
            r'\b' + commit + r'\b', receipt['tool_versions']['target_clang']):
        raise ValueError('Runtime source compiler revision mismatch')
    if (set(receipt['binaries']) != set(media.TOOLS)
            or any(not re.fullmatch('[a-f0-9]{64}', v) for v in receipt['binaries'].values())):
        raise ValueError('Runtime source helper binding missing')
    return pins, receipt


def notice_bytes(path, pin, fileobj=None):
    """Read named regular files only; never extract an upstream tree."""
    result = {}
    if Path(pin['root']).name != pin['root'] or pin['root'] in ('', '.', '..'):
        raise ValueError('Invalid runtime source root')
    with tarfile.open(path, fileobj=fileobj) as archive:
        entries = archive.getmembers()
        for name in pin['licenses']:
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise ValueError('Invalid runtime notice path')
            matches = [m for m in entries if m.name == pin['root'] + '/' + name]
            if len(matches) != 1 or not matches[0].isfile() or not 0 < matches[0].size <= 2 * 1024**2:
                raise ValueError('Required regular runtime notice missing: ' + name)
            result[name] = archive.extractfile(matches[0]).read()
    return result


def pointer(receipt):
    return ('Matching runtime sources: ' + receipt['source_archive'] + '\nSHA-256: '
            + receipt['source_archive_sha256'] + '\n'
            'The full source companion accompanies the Windows binary downloads.\n'
            'This directory retains component notices and exact source identities.\n'
            'Windows UCRT DLLs are platform dependencies, not bundled UCRT binaries.\n')


def export_offer(media_receipt, downloads, output):
    pins, compiled = inputs(media_receipt)  # Refuse incompatible inputs before mutation/download.
    no_redirect(output); no_redirect(downloads)
    if output.exists() or not downloads.is_dir():
        raise ValueError('Use fresh output and an existing source download directory')
    archives = {}; notices = {}
    for key, pin in pins['sources'].items():
        archives[key] = media.download(pin, downloads)
        notices.update({key + '/' + name: data for name, data in notice_bytes(archives[key], pin).items()})
    output.mkdir()
    licenses = output / 'licenses'; licenses.mkdir()
    for name, data in notices.items():
        path = licenses / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    with tarfile.open(output / ARCHIVE, 'w:xz', preset=0) as archive:
        def add(name, stream, size, mode=0o644):
            entry = tarfile.TarInfo(name); entry.size = size; entry.mode = mode
            archive.addfile(entry, stream)  # Default owner/names/time are deterministic zero/empty.
        for key, path in archives.items():
            with path.open('rb') as stream: add('sources/' + media.filename(pins['sources'][key]), stream, path.stat().st_size)
        for name, path in recipe_files().items():
            raw = path.read_bytes(); add(name, io.BytesIO(raw), len(raw))
        for name, raw in notices.items(): add('notices/' + name, io.BytesIO(raw), len(raw))
    receipt = {'schema': 1, 'sources': pins['sources'], 'toolchain_sha256': pins['toolchain_sha256'],
               'association': pins.get('association', 'Pinned source/toolchain association'),
               'media_receipt_sha256': digest(media_receipt), 'binaries': compiled['binaries'],
               'recipe_sha256': recipe_hash(), 'source_archive': ARCHIVE,
               'source_archive_sha256': digest(output / ARCHIVE),
               'notices': {name: hashlib.sha256(raw).hexdigest() for name, raw in notices.items()}}
    (output / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
    (output / 'SOURCE.txt').write_text(pointer(receipt))
    verify_offer(output, media_receipt)
    return receipt


def verify_offer(output, media_receipt, exported=False):
    pins, compiled = inputs(media_receipt)
    no_redirect(output)
    for path in output.rglob('*'): no_redirect(path)
    receipt = json.loads((output / RECEIPT).read_text())
    if (receipt['schema'] != 1 or receipt['sources'] != pins['sources']
            or receipt['toolchain_sha256'] != pins['toolchain_sha256']
            or receipt['media_receipt_sha256'] != digest(media_receipt)
            or receipt['binaries'] != compiled['binaries'] or receipt['recipe_sha256'] != recipe_hash()
            or receipt['source_archive'] != ARCHIVE or digest(output / ARCHIVE) != receipt['source_archive_sha256']):
        raise ValueError('Runtime source receipt or companion changed')
    expected = {key + '/' + name for key, pin in pins['sources'].items() for name in pin['licenses']}
    actual = receipt['notices'] if exported else {
        p.relative_to(output / 'licenses').as_posix(): digest(p)
        for p in (output / 'licenses').rglob('*') if p.is_file()}
    if set(actual) != expected or actual != receipt['notices']:
        raise ValueError('Runtime component notices changed or missing')
    if not exported and (output / 'SOURCE.txt').read_text() != pointer(receipt):
        raise ValueError('Runtime source pointer changed')
    with tarfile.open(output / ARCHIVE) as archive:
        names = [m.name for m in archive.getmembers()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate runtime source-offer entry')
        for pin in pins['sources'].values():
            member = archive.getmember('sources/' + media.filename(pin))
            if not member.isfile() or hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest() != pin['sha256']:
                raise ValueError('Runtime source input changed')
        for key, pin in pins['sources'].items():
            member = archive.getmember('sources/' + media.filename(pin))
            with archive.extractfile(member) as stream:
                originals = notice_bytes(None, pin, fileobj=stream)
            for name, raw in originals.items():
                if actual[key + '/' + name] != hashlib.sha256(raw).hexdigest():
                    raise ValueError('Runtime notice differs from pinned source')
        for name, path in recipe_files().items():
            member = archive.getmember(name)
            if not member.isfile() or archive.extractfile(member).read() != path.read_bytes():
                raise ValueError('Runtime source export recipe changed')
        for name, wanted in actual.items():
            member = archive.getmember('notices/' + name)
            if not member.isfile() or hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest() != wanted:
                raise ValueError('Archived runtime notice changed')
    return receipt


def stage_offer(output, media_receipt, destination):
    receipt = verify_offer(output, media_receipt)
    no_redirect(destination)
    if destination.exists():
        raise ValueError('Runtime notices need a fresh staging destination')
    shutil.copytree(output / 'licenses', destination)
    for name in (RECEIPT, 'SOURCE.txt'): shutil.copy2(output / name, destination / name)
    return receipt


def verify_packages(output, media_receipt, portable, installed):
    receipt = verify_offer(output, media_receipt)
    no_redirect(portable); no_redirect(installed)
    expected = {name: output / 'licenses' / name for name in receipt['notices']}
    expected.update({name: output / name for name in (RECEIPT, 'SOURCE.txt')})
    for name, original in expected.items():
        relative = 'LICENSES/ffmpeg/runtime/' + name
        a = portable / relative
        found = [p for p in installed.rglob(Path(name).name)
                 if p.as_posix().lower().endswith('/licenses/' + relative.lower())]
        if len(found) != 1:
            raise ValueError('Missing or duplicate installed runtime notice: ' + name)
        for path in (a, found[0]):
            no_redirect(path)
            if not path.is_file() or digest(path) != digest(original):
                raise ValueError('Packaged runtime notice changed: ' + name)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--downloads', type=Path)
    parser.add_argument('--stage-into', type=Path)
    parser.add_argument('--check-output', action='store_true')
    parser.add_argument('--verify-export', action='store_true')
    parser.add_argument('--portable', type=Path)
    parser.add_argument('--installed', type=Path)
    args = parser.parse_args()
    if args.portable or args.installed:
        if not args.portable or not args.installed or args.downloads or args.stage_into or args.check_output or args.verify_export:
            parser.error('Package verification needs only both payloads, media receipt and output')
        result = verify_packages(args.output, args.media_receipt, args.portable, args.installed)
    elif args.check_output or args.verify_export:
        if args.downloads or args.stage_into: parser.error('Verification does not stage or download')
        result = verify_offer(args.output, args.media_receipt, exported=args.verify_export)
    else:
        if not args.downloads: parser.error('Export needs a source download directory')
        result = export_offer(args.media_receipt, args.downloads, args.output)
        if args.stage_into: stage_offer(args.output, args.media_receipt, args.stage_into)
    print(json.dumps(result, indent=2))
