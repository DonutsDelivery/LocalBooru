#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).with_name('cargo-cache-hygiene.py')
spec=importlib.util.spec_from_file_location('hygiene', SCRIPT)
hygiene=importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


class CacheHygieneTests(unittest.TestCase):
    def fixture(self, root):
        (root/'.rustc_info.json').write_text('{}')
        profile=root/'debug'
        for name in ['deps','build','.fingerprint','incremental','examples','bundle']:
            p=profile/name; p.mkdir(parents=True); (p/'fixture').write_text('synthetic')
        (profile/'donutmediacenter').write_text('synthetic executable')
        return profile

    def test_over_budget_removes_only_compiler_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'target';root.mkdir();profile=self.fixture(root)
            hygiene.trim(root, 1)
            for name in ['deps','build','.fingerprint','incremental','examples']:
                self.assertFalse((profile/name).exists())
            self.assertTrue((profile/'bundle/fixture').is_file())
            self.assertTrue((profile/'donutmediacenter').is_file())

    def test_under_budget_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'target';root.mkdir();profile=self.fixture(root)
            hygiene.trim(root, 1024**3)
            self.assertTrue((profile/'deps/fixture').exists())

    def test_unrecognized_root_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'target';root.mkdir();profile=self.fixture(root)
            (root/'.rustc_info.json').unlink()
            with self.assertRaises(ValueError): hygiene.trim(root, 1)
            self.assertTrue((profile/'deps/fixture').exists())

    def test_symlink_compiler_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'target';root.mkdir();profile=self.fixture(root)
            external=Path(tmp)/'external';external.mkdir();(external/'keep').write_text('synthetic')
            import shutil
            shutil.rmtree(profile/'deps');(profile/'deps').symlink_to(external, target_is_directory=True)
            hygiene.trim(root, 1)
            self.assertTrue((external/'keep').exists())
            self.assertTrue((profile/'deps').is_symlink())

    def test_cross_target_and_release_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'target';root.mkdir();self.fixture(root)
            profile=root/'x86_64-pc-windows-msvc/release'
            (profile/'.fingerprint').mkdir(parents=True);(profile/'deps').mkdir();(profile/'deps/fixture').write_text('synthetic')
            (profile/'donutmediacenter.exe').write_text('synthetic')
            hygiene.trim(root, 1)
            self.assertFalse((profile/'deps').exists())
            self.assertTrue((profile/'donutmediacenter.exe').exists())


if __name__=='__main__':unittest.main()
