#!/usr/bin/env python3
"""Validate and ad-hoc sign the exact bundled media payload before DMG creation."""
import argparse
import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dmc_media_builder', HERE / 'build-macos-media.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def validate_payload(directory, receipt):
    if set(receipt['architectures']) != set(builder.ARCHS):
        raise ValueError('Expected a universal media payload')
    if receipt['sources'] != json.loads(builder.PINS.read_text()):
        raise ValueError('Media payload does not match the pinned sources')
    for name in builder.TOOLS:
        binary = directory / 'bin' / name
        builder.validate_tool(binary)
        if builder.digest(binary) != receipt['binaries'][name]:
            raise ValueError('Media payload checksum mismatch: ' + name)


def smoke(directory):
    """Synthetic encode/decode, including sound samples; no app or audio device opens."""
    ffmpeg = directory / 'bin/ffmpeg'
    ffprobe = directory / 'bin/ffprobe'
    with tempfile.TemporaryDirectory(prefix='dmc-media-smoke-') as temporary:
        clip = Path(temporary) / 'synthetic tone.mp4'
        builder.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x96:rate=10',
                     '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
                     '-t', '1', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac',
                     '-pix_fmt', 'yuv420p', clip], timeout=60)
        streams = json.loads(builder.run([ffprobe, '-v', 'error', '-show_streams',
                                         '-of', 'json', clip], capture=True, timeout=60))['streams']
        if {x['codec_name'] for x in streams} != {'h264', 'aac'}:
            raise ValueError('Synthetic audio/video encoding failed')
        builder.run([ffmpeg, '-v', 'error', '-i', clip, '-f', 'null', '-'], timeout=60)
        image = Path(temporary) / 'thumbnail.webp'
        builder.run([ffmpeg, '-v', 'error', '-i', clip, '-frames:v', '1',
                     '-c:v', 'libwebp', image], timeout=60)
        if image.read_bytes()[:4] != b'RIFF' or image.read_bytes()[8:12] != b'WEBP':
            raise ValueError('WebP thumbnail encoding failed')


def sign(app, payload, edition):
    receipt = json.loads((payload / 'media-build-receipt.json').read_text())
    validate_payload(payload, receipt)
    bundled = app / 'Contents/Resources'
    validate_payload(bundled, receipt)
    smoke(bundled)
    helper_entitlements = HERE / '../src-tauri/macos/media-helper.entitlements.plist'
    for name in builder.TOOLS:
        binary = bundled / 'bin' / name
        args = ['codesign', '--force', '--sign', '-']
        if edition == 'store-lite':
            args.extend(['--entitlements', helper_entitlements])
        builder.run([*args, binary])
        builder.run(['codesign', '--verify', '--strict', binary])
        if edition == 'store-lite':
            entitlements = plistlib.loads(builder.run(
                ['codesign', '-d', '--entitlements', ':-', binary], capture=True).encode())
            if entitlements != {'com.apple.security.app-sandbox': True,
                                'com.apple.security.inherit': True}:
                raise ValueError('Media helper does not have only sandbox inheritance')
    args = ['codesign', '--force', '--sign', '-']
    if edition == 'store-lite':
        args.extend(['--entitlements', HERE / '../src-tauri/macos/store-lite.entitlements.plist'])
    builder.run([*args, app])
    builder.run(['codesign', '--verify', '--deep', '--strict', app])
    return {name: builder.digest(bundled / 'bin' / name) for name in builder.TOOLS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--edition', choices=('full', 'store-lite'), required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    try:
        hashes = sign(args.app, args.payload, args.edition)
        args.receipt.write_text(json.dumps({'schema': 1, 'edition': args.edition,
                                           'signature': 'adhoc', 'binaries': hashes,
                                           'synthetic_encode_decode': True}, indent=2) + '\n')
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
