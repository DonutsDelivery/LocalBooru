#!/usr/bin/env python3
"""Export one exact Git revision without working files, caches or parent history."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


def export(repo, revision, output):
    commit = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', '--verify', revision + '^{commit}'], text=True).strip()
    prefix = 'DonutMediaCenter-' + commit + '/'
    with output.open('xb') as raw:
        with subprocess.Popen(['git', '-C', str(repo), 'archive', '--format=tar',
                               '--prefix=' + prefix, commit], stdout=subprocess.PIPE) as archive:
            try:
                with lzma.open(raw, 'wb') as compressed:
                    shutil.copyfileobj(archive.stdout, compressed)
            finally:
                archive.stdout.close()
            if archive.wait(timeout=60) != 0:
                raise ValueError('Git source export failed')
    with tarfile.open(output) as archive:
        if not archive.getmembers() or any(
            not member.name.startswith(prefix) and not (member.isdir() and member.name == prefix.rstrip('/'))
            for member in archive
        ):
            raise ValueError('Invalid application source archive')
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'source_commit': commit, 'archive': output.name, 'sha256': digest}


def verify(receipt_path, expected_commit=None):
    receipt = json.loads(receipt_path.read_text())
    commit, name, expected = (receipt[key] for key in ('source_commit', 'archive', 'sha256'))
    if not isinstance(commit, str) or not re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}', commit):
        raise ValueError('Invalid source commit in receipt')
    if expected_commit and commit != expected_commit:
        raise ValueError('Application source does not match the binary revision')
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Source archive must be a basename beside its receipt')
    if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError('Invalid source archive SHA-256')
    output = receipt_path.parent / name
    if output.is_symlink() or not output.is_file():
        raise ValueError('Source archive must be a regular file')
    with output.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('Application source archive hash does not match')
    prefix = 'DonutMediaCenter-' + commit + '/'
    with tarfile.open(output) as archive:
        entries = archive.getmembers()
        if not entries or any(not entry.name.startswith(prefix) and not (
                entry.isdir() and entry.name == prefix.rstrip('/')) for entry in entries):
            raise ValueError('Source archive does not identify the declared revision')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-receipt', type=Path)
    args = parser.parse_args()
    try:
        if args.verify_receipt:
            if args.repo or args.output:
                parser.error('Receipt verification does not accept --repo or --output')
            result = verify(args.verify_receipt, args.commit)
        else:
            if not args.repo or not args.output:
                parser.error('Source export requires --repo and --output')
            result = export(args.repo, args.commit or 'HEAD', args.output)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, TypeError, OSError, tarfile.TarError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')
