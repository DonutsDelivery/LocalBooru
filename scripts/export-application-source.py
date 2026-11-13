#!/usr/bin/env python3
"""Export one exact Git revision without working files, caches or parent history."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--commit', default='HEAD')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.repo, args.commit, args.output), indent=2))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')
