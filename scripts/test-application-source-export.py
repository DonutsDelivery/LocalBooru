#!/usr/bin/env python3
"""Source-offer provenance tests using a disposable synthetic Git repository."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('source_export', Path(__file__).with_name('export-application-source.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class SourceTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix='dmc source fixture ')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); repo = root / 'repository'; repo.mkdir()
        subprocess.run(['git', 'init', '-q', '--template=', repo], check=True)
        (repo / 'source.txt').write_text('committed synthetic source')
        subprocess.run(['git', '-C', repo, 'add', 'source.txt'], check=True)
        subprocess.run(['git', '-C', repo, '-c', 'user.name=DMC fixture', '-c',
                        'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                        'commit', '-qm', 'synthetic source'], check=True)
        return root, repo

    def test_archive_is_exact_commit_and_excludes_working_files(self):
        root, repo = self.fixture()
        (repo / 'source.txt').write_text('uncommitted fixture change')
        (repo / 'untracked.txt').write_text('untracked synthetic data')
        output = root / 'source.tar.xz'
        receipt = module.export(repo, 'HEAD', output)
        with tarfile.open(output) as archive:
            files = [entry for entry in archive.getmembers() if entry.isfile()]
            self.assertEqual(len(files), 1)
            self.assertEqual(archive.extractfile(files[0]).read(), b'committed synthetic source')
            self.assertIn(receipt['source_commit'], files[0].name)

    def test_existing_output_is_preserved(self):
        root, repo = self.fixture(); output = root / 'existing.tar.xz'; output.write_bytes(b'keep')
        with self.assertRaises(FileExistsError): module.export(repo, 'HEAD', output)
        self.assertEqual(output.read_bytes(), b'keep')

    def test_invalid_revision_creates_no_archive(self):
        root, repo = self.fixture(); output = root / 'missing.tar.xz'
        with self.assertRaises(subprocess.CalledProcessError): module.export(repo, 'not-a-revision', output)
        self.assertFalse(output.exists())

    def test_verified_receipt_binds_archive_and_expected_revision(self):
        root, repo = self.fixture(); output = root / 'source.tar.xz'
        receipt = module.export(repo, 'HEAD', output)
        path = root / 'receipt.json'; path.write_text(json.dumps(receipt))
        self.assertEqual(module.verify(path, receipt['source_commit']), receipt)
        with self.assertRaises(ValueError): module.verify(path, '0' * 40)
        receipt['source_commit'] = '0' * 40; path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): module.verify(path)

    def test_changed_archive_is_rejected(self):
        root, repo = self.fixture(); output = root / 'source.tar.xz'
        receipt = module.export(repo, 'HEAD', output)
        path = root / 'receipt.json'; path.write_text(json.dumps(receipt))
        with output.open('ab') as stream: stream.write(b'changed synthetic archive')
        with self.assertRaises(ValueError): module.verify(path)

    def test_receipt_cannot_read_an_unrelated_path(self):
        root, repo = self.fixture(); output = root / 'source.tar.xz'
        receipt = module.export(repo, 'HEAD', output)
        receipt['archive'] = '../unrelated.tar.xz'
        path = root / 'receipt.json'; path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError): module.verify(path)


if __name__ == '__main__': unittest.main()
