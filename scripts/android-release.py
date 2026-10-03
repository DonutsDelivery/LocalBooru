#!/usr/bin/env python3
"""Build and verify permanently signed DMC Android release artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args, env=None, capture=False):
    return subprocess.run(args, cwd=ROOT, env=env, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def native_alignment(archive):
    evidence = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('.so') or not any('/'+abi+'/' in '/'+name for abi in ['arm64-v8a', 'x86_64']):
                continue
            data = z.read(name)
            if data[:6] != b'\x7fELF\x02\x01':
                raise ValueError(f'{name}: expected a little-endian 64-bit ELF')
            offset = struct.unpack_from('<Q', data, 32)[0]
            size, count = struct.unpack_from('<HH', data, 54)
            loads = 0
            for i in range(count):
                typ, _, _, address, _, _, memory, alignment = struct.unpack_from('<IIQQQQQQ', data, offset+i*size)
                if typ == 1:
                    loads += 1
                    if alignment < 16384:
                        raise ValueError(f'{name}: load segment alignment is {alignment}, below 16KB')
                if typ == 0x6474e552 and (address+memory) % 16384:
                    raise ValueError(f'{name}: RELRO end is not 16KB aligned')
            if not loads:
                raise ValueError(f'{name}: no load segments')
            evidence.append(name)
    if not evidence:
        raise ValueError(f'{archive}: missing 64-bit native libraries')
    return evidence


def signing_env(config):
    env = os.environ.copy()
    required = ['ANDROID_KEYSTORE', 'ANDROID_KEY_ALIAS', 'ANDROID_STORE_PASS', 'ANDROID_KEY_PASS']
    if not all(env.get(k) for k in required):
        folder = Path(env.get('XDG_CONFIG_HOME', str(Path.home()/'.config'))) / 'donutsdelivery/release-secrets/android' / config['signingKeyRef']
        props = {}
        for line in (folder/'signing.properties').read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1); props[k.strip()] = v.strip()
        keys = list(folder.glob('*.jks')) + list(folder.glob('*.keystore'))
        if len(keys) != 1:
            raise ValueError('Expected exactly one permanent Android keystore')
        env.update(ANDROID_KEYSTORE=str(keys[0]), ANDROID_KEY_ALIAS=props['keyAlias'],
                   ANDROID_STORE_PASS=props['storePassword'], ANDROID_KEY_PASS=props['keyPassword'])
    certificate = run('keytool', '-list', '-v', '-keystore', env['ANDROID_KEYSTORE'],
                      '-alias', env['ANDROID_KEY_ALIAS'], '-storepass:env', 'ANDROID_STORE_PASS', env=env, capture=True)
    match = re.search(r'SHA256:\s*([0-9A-Fa-f:]+)', certificate)
    if not match or match[1].replace(':', '').lower() != config['signingCertificateSha256']:
        raise ValueError('Android release key does not match the pinned certificate')
    if 'CN=Android Debug' in certificate:
        raise ValueError('Debug Android certificates cannot sign a release')
    sdk = Path(env.get('ANDROID_SDK_ROOT') or env.get('ANDROID_HOME') or Path.home()/'Android/Sdk')
    tools = sorted((sdk/'build-tools').glob('*'), key=lambda p: tuple(int(n) for n in re.findall(r'\d+', p.name)))
    if not tools:
        raise ValueError('Android SDK build tools are missing')
    env['PATH'] = str(tools[-1])+os.pathsep+env['PATH']
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aab', action='store_true', help='Also prepare the Google Play app bundle')
    parser.add_argument('--sign-only', action='store_true', help='Verify and sign existing release outputs')
    parser.add_argument('--keep-cache', action='store_true', help='Retained for compatibility; release caches are preserved')
    args = parser.parse_args()
    config = json.loads((ROOT/'release/android.json').read_text())
    env = signing_env(config)
    version = json.loads((ROOT/'src-tauri/tauri.conf.json').read_text())['version']
    major, minor, patch = map(int, version.split('-', 1)[0].split('.'))
    version_code = major * 1000000 + minor * 1000 + patch
    source = run('git', 'rev-parse', 'HEAD', capture=True).strip()
    if run('git', 'status', '--porcelain', '--untracked-files=all', capture=True).strip():
        raise ValueError('Release source must be committed and clean before building/signing')
    marker = ROOT/'src-tauri/gen/android/app/build/dmc-release-source.json'
    bundletool = env.get('ANDROID_BUNDLETOOL')
    if args.aab and (not bundletool or not Path(bundletool).is_file()):
        raise ValueError('Set ANDROID_BUNDLETOOL to the verified Google bundletool JAR')
    if not args.sign_only:
        command = [str(ROOT/'scripts/run-cargo.sh'), 'tauri', 'android', 'build', '--apk', '--ci']
        if args.aab:
            command += ['--aab']
        run(*command, env=env)
        if run('git', 'rev-parse', 'HEAD', capture=True).strip() != source or run('git', 'status', '--porcelain', '--untracked-files=all', capture=True).strip():
            raise ValueError('Release source changed during the build; rebuild from clean exact source')
        marker.write_text(json.dumps({'sourceCommit': source, 'version': version})+'\n')
    if not marker.is_file() or json.loads(marker.read_text()).get('sourceCommit') != source:
        raise ValueError('Existing Android outputs lack matching exact-source provenance; rebuild them')
    outputs = ROOT/'src-tauri/gen/android/app/build/outputs'
    unsigned = outputs/'apk/universal/release/app-universal-release-unsigned.apk'
    unsigned_aab = outputs/'bundle/universalRelease/app-universal-release.aab'
    if not unsigned.is_file() or (args.aab and not unsigned_aab.is_file()):
        raise ValueError('Expected Android release build outputs are missing')
    # Keep signing staging outside the repository; never copy credentials.
    with tempfile.TemporaryDirectory(prefix='dmc-android-release-') as folder:
        folder = Path(folder); apk = folder/'DonutMediaCenter.apk'; aligned = folder/'aligned.apk'
        run('zipalign', '-f', '-P', '16', '4', str(unsigned), str(aligned), env=env)
        run('apksigner', 'sign', '--ks', env['ANDROID_KEYSTORE'], '--ks-key-alias', env['ANDROID_KEY_ALIAS'],
            '--ks-pass', 'env:ANDROID_STORE_PASS', '--key-pass', 'env:ANDROID_KEY_PASS',
            '--v4-signing-enabled', 'false', '--out', str(apk), str(aligned), env=env)
        signing = run('apksigner', 'verify', '--verbose', '--print-certs', str(apk), env=env, capture=True)
        if config['signingCertificateSha256'] not in signing.lower():
            raise ValueError('Signed APK certificate mismatch')
        run('zipalign', '-c', '-P', '16', '4', str(apk), env=env)
        badging = run('aapt2', 'dump', 'badging', str(apk), env=env, capture=True)
        for expected in [f"name='{config['packageId']}'", f"versionName='{version}'", f"versionCode='{version_code}'", f"targetSdkVersion:'{config['targetSdk']}'", "application-label:'DonutMediaCenter'"]:
            if expected not in badging:
                raise ValueError(f'APK metadata mismatch: {expected}')
        if 'application-debuggable' in badging:
            raise ValueError('Release APK is debuggable')
        with zipfile.ZipFile(apk) as z:
            abis = sorted({name.split('/')[1] for name in z.namelist() if name.startswith('lib/') and name.endswith('.so')})
        if abis != sorted(config['abis']):
            raise ValueError(f'APK ABI set is {abis}, expected {config["abis"]}')
        artifacts = [(apk, native_alignment(apk))]
        if args.aab:
            aab = folder/'DonutMediaCenter.aab'
            run('jarsigner', '-keystore', env['ANDROID_KEYSTORE'], '-storepass:env', 'ANDROID_STORE_PASS',
                '-keypass:env', 'ANDROID_KEY_PASS', '-signedjar', str(aab), str(unsigned_aab), env['ANDROID_KEY_ALIAS'], env=env)
            verification = run('jarsigner', '-verify', str(aab), env=env, capture=True)
            if 'jar verified.' not in verification:
                raise ValueError('App bundle signature failed verification')
            cert = run('keytool', '-printcert', '-jarfile', str(aab), env=env, capture=True)
            match = re.search(r'SHA256:\s*([0-9A-Fa-f:]+)', cert)
            if not match or match[1].replace(':','').lower() != config['signingCertificateSha256']:
                raise ValueError('App bundle signing certificate mismatch')
            run('java', '-jar', bundletool, 'validate', '--bundle='+str(aab), env=env)
            bundle_config = run('java', '-jar', bundletool, 'dump', 'config', '--bundle='+str(aab), env=env, capture=True)
            if 'PAGE_ALIGNMENT_16K' not in bundle_config:
                raise ValueError('App bundle is missing 16KB native packaging alignment')
            artifacts.append((aab, native_alignment(aab)))
        evidence = {'version': version, 'versionCode': version_code, 'sourceCommit': source,
                    'packageId': config['packageId'], 'targetSdk': config['targetSdk'], 'abis': abis,
                    'signingCertificateSha256': config['signingCertificateSha256'], 'artifacts': []}
        for artifact, alignment in artifacts:
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            evidence['artifacts'].append({'filename': artifact.name, 'size': artifact.stat().st_size,
                                           'sha256': digest, 'alignedNativeLibraries': alignment})
        # Publish only after every requested artifact passes its gates.
        for artifact, _ in artifacts:
            target = ROOT/artifact.name; staged = target.with_suffix(target.suffix+'.staging')
            shutil.copyfile(artifact, staged); os.replace(staged, target)
        (ROOT/'DonutMediaCenter-Android.json').write_text(json.dumps(evidence,indent=2)+'\n')
        (ROOT/'SHA256SUMS-Android').write_text(''.join(f'{a["sha256"]}  {a["filename"]}\n' for a in evidence['artifacts']))
        print(f'Verified DMC {version} Android release artifacts:', ', '.join(a['filename'] for a in evidence['artifacts']))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Android release failed: {error}')
