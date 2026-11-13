#!/usr/bin/env python3
"""Validate canonical Windows outputs, stage, and optionally package using the SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from xml.sax.saxutils import escape, quoteattr

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = {'DonutMediaCenter.exe', 'LICENSE', 'NOTICE', 'TRADEMARKS.md',
            'bin/ffmpeg.exe', 'bin/ffprobe.exe'}
DEVICES = re.compile(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$', re.I)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git_file(source, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', source + ':' + path])


def validate_identity(identity, publisher, version):
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]{2,49}', identity)
            or identity.endswith('.') or DEVICES.fullmatch(identity)):
        raise ValueError('Invalid package identity')
    if not publisher.startswith('CN=') or not publisher[3:].strip() or any(ord(c) < 32 for c in publisher):
        raise ValueError('An explicit certificate/Partner Center publisher is required')
    if (not re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', version)
            or any(int(x) > 65535 for x in version.split('.')) or int(version.split('.')[0]) == 0):
        raise ValueError('Use a four-part Windows version, first component 1..65535')


def archive_entries(archive):
    """Reject traversal, Windows aliases, payload surprises, and expansion bombs."""
    entries, nodes, total = [], {}, 0
    for info in archive.infolist():
        name = info.filename.rstrip('/')
        parts = name.split('/')
        if (not name or info.filename.startswith('/') or '\\' in name
                or any(not p or p in ('.', '..') or p.endswith(('.', ' '))
                       or re.search(r'[<>:"|?*\x00-\x1f]', p)
                       or DEVICES.fullmatch(p.split('.')[0]) for p in parts)
                or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError('Unsafe ZIP path or encoding: ' + name)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError('ZIP contains a link or special file')
        folded = name.casefold()
        if folded in nodes:
            raise ValueError('Duplicate Windows ZIP path: ' + name)
        nodes[folded] = info.is_dir()
        if not info.is_dir():
            allowed = (name in REQUIRED or name.startswith('LICENSES/')
                       or (len(parts) == 2 and parts[0] == 'bin' and name.lower().endswith('.dll')))
            if not allowed:
                raise ValueError('Unexpected portable payload: ' + name)
            if info.file_size > 512 * 1024 ** 2:
                raise ValueError('Portable file exceeds packaging limit')
        total += info.file_size
        entries.append(info)
    if len(entries) > 4096 or total > 1024 ** 3:
        raise ValueError('Portable ZIP exceeds packaging limits')
    for name in nodes:
        parts = name.split('/')
        if any(nodes.get('/'.join(parts[:i])) is False for i in range(1, len(parts))):
            raise ValueError('ZIP has a file/directory collision')
    files = {i.filename for i in entries if not i.is_dir()}
    if not REQUIRED <= files or not any(n.startswith('LICENSES/') for n in files):
        raise ValueError('Portable ZIP lacks executable, helpers, or notices')
    return entries


def validate_pe(path):
    with path.open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b'MZ':
            raise ValueError('Missing PE header: ' + path.name)
        offset = struct.unpack_from('<I', header, 0x3c)[0]
        if not 64 <= offset <= 65536:
            raise ValueError('Invalid PE header offset')
        stream.seek(offset)
        pe = stream.read(26)
    if (len(pe) != 26 or pe[:4] != b'PE\0\0' or struct.unpack_from('<H', pe, 4)[0] != 0x8664
            or struct.unpack_from('<H', pe, 24)[0] != 0x20b
            or not struct.unpack_from('<H', pe, 22)[0] & 2):
        raise ValueError('Expected x64 PE32+ executable/library: ' + path.name)


def stage(zip_path, receipt_path, source, output, identity, publisher, display_name, version):
    validate_identity(identity, publisher, version)
    if not re.fullmatch('[0-9a-f]{40}', source) or not display_name.strip():
        raise ValueError('Use an exact source commit and publisher display name')
    receipt = json.loads(Path(receipt_path).read_text())
    expected_recipe = hashlib.sha256(git_file(source, 'scripts/build-windows-docker.sh')).hexdigest()
    if (not isinstance(receipt, dict) or receipt.get('schema') != 1 or receipt.get('source_commit') != source
            or receipt.get('portable_sha256') != digest(zip_path)
            or receipt.get('target') != 'x86_64-pc-windows-msvc'
            or receipt.get('recipe_sha256') != expected_recipe):
        raise ValueError('Canonical build receipt does not match source, recipe and portable ZIP')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.resolve() == ROOT or ROOT in output.resolve().parents:
        raise ValueError('Output must be a new directory outside the checkout')
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive_entries(archive)
        for name in ('LICENSE', 'NOTICE', 'TRADEMARKS.md'):
            if archive.read(name) != git_file(source, name):
                raise ValueError('Portable notice differs from source: ' + name)
        output.mkdir(parents=True)
        try:
            for info in entries:
                archive.extract(info, output)
            for path in output.rglob('*'):
                if path.suffix.lower() in ('.exe', '.dll'):
                    validate_pe(path)
            for icon in ('Square150x150Logo.png', 'Square44x44Logo.png', 'StoreLogo.png'):
                (output / icon).write_bytes(git_file(source, 'src-tauri/icons/' + icon))
            q = quoteattr
            manifest = f'''<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10" xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10" xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities" IgnorableNamespaces="uap rescap">
  <Identity Name={q(identity)} Publisher={q(publisher)} Version={q(version)} ProcessorArchitecture="x64" />
  <Properties><DisplayName>DonutMediaCenter</DisplayName><PublisherDisplayName>{escape(display_name)}</PublisherDisplayName><Logo>StoreLogo.png</Logo></Properties>
  <Resources><Resource Language="en-us" /></Resources>
  <Dependencies><TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.19041.0" /></Dependencies>
  <Applications><Application Id="App" Executable="DonutMediaCenter.exe" EntryPoint="Windows.FullTrustApplication"><uap:VisualElements DisplayName="DonutMediaCenter" Description="Local image, video and music library" BackgroundColor="transparent" Square150x150Logo="Square150x150Logo.png" Square44x44Logo="Square44x44Logo.png" /><Extensions><uap:Extension Category="windows.protocol"><uap:Protocol Name="donutmediacenter"><uap:DisplayName>DonutMediaCenter</uap:DisplayName></uap:Protocol></uap:Extension></Extensions></Application></Applications>
  <Capabilities><rescap:Capability Name="runFullTrust" /></Capabilities>
</Package>
'''
            (output / 'AppxManifest.xml').write_text(manifest, encoding='utf-8')
            (output / 'DMC-Build-Receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        except BaseException:
            shutil.rmtree(output)
            raise
    return output, receipt


def package(stage_dir, package_path, makeappx, signtool=None, thumbprint=None, timestamp=None, run=subprocess.run):
    target = Path(package_path).absolute()
    if target.exists() or target.is_symlink() or target.suffix.lower() != '.msix':
        raise ValueError('MSIX output must be a new .msix file')
    if Path(stage_dir).resolve() in target.resolve().parents or ROOT in target.resolve().parents:
        raise ValueError('MSIX output must be outside its input stage and the checkout')
    if not makeappx or bool(signtool) != bool(thumbprint) or bool(thumbprint) != bool(timestamp):
        raise ValueError('Use MakeAppx; signing requires SignTool, certificate thumbprint and timestamp URL together')
    if thumbprint and (not re.fullmatch('[0-9A-Fa-f]{40}', thumbprint) or not timestamp.startswith('https://')):
        raise ValueError('Use a certificate SHA-1 thumbprint and HTTPS RFC3161 timestamp')
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        run([makeappx, 'pack', '/d', str(stage_dir), '/p', str(target), '/h', 'SHA256', '/no'], check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError('MakeAppx did not produce an MSIX')
        with tempfile.TemporaryDirectory(prefix='dmc-msix-verify-') as temporary:
            unpacked = Path(temporary) / 'unpacked'
            run([makeappx, 'unpack', '/p', str(target), '/d', str(unpacked), '/no'], check=True)
            for path in Path(stage_dir).rglob('*'):
                if path.is_file() and digest(path) != digest(unpacked / path.relative_to(stage_dir)):
                    raise ValueError('SDK round-trip changed staged content')
        if signtool:
            run([signtool, 'sign', '/fd', 'SHA256', '/sha1', thumbprint, '/tr', timestamp, '/td', 'SHA256', str(target)], check=True)
            run([signtool, 'verify', '/pa', '/all', '/v', str(target)], check=True)
    except BaseException:
        if target.is_file(): target.unlink()
        raise
    return {'msix_sha256': digest(target), 'signed_and_verified': bool(signtool)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('zip', 'receipt', 'source', 'output', 'identity', 'publisher', 'publisher-display-name', 'version'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--package', help='Optional new .msix output; requires the real SDK MakeAppx tool')
    parser.add_argument('--makeappx', help='Executable path, or a wrapper invoking SDK MakeAppx under Wine')
    parser.add_argument('--signtool')
    parser.add_argument('--certificate-thumbprint')
    parser.add_argument('--timestamp-url')
    args = parser.parse_args()
    if args.package and Path(str(args.package) + '.receipt.json').exists():
        parser.error('The MSIX output receipt already exists')
    if not args.package and any((args.makeappx, args.signtool, args.certificate_thumbprint, args.timestamp_url)):
        parser.error('SDK/signing options require --package')
    output, receipt = stage(args.zip, args.receipt, args.source.lower(), args.output, args.identity,
                            args.publisher, args.publisher_display_name, args.version)
    if args.package:
        result = package(output, args.package, args.makeappx, args.signtool,
                         args.certificate_thumbprint, args.timestamp_url)
        result.update(source_commit=receipt['source_commit'], portable_sha256=receipt['portable_sha256'],
                      identity=args.identity, publisher=args.publisher, version=args.version)
        Path(str(args.package) + '.receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print('SDK package verified:', args.package)
    else:
        print('Staging validated; no MSIX built:', output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile, struct.error) as error:
        print('MSIX packaging failed:', error, file=sys.stderr)
        sys.exit(1)
