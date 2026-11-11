"""Synthetic helper environment checks; no user profile or media."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as svp_app


class ContainedHelperEnvironmentTests(unittest.TestCase):
    def test_helper_inherits_container_roots_and_omits_unrelated_variables(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roots = {key: str(root / name) for key, name in (
                ('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                ('XDG_CACHE_HOME', 'cache'), ('XDG_STATE_HOME', 'state'))}
            for path in roots.values():
                Path(path).mkdir()
            with patch.dict(os.environ, {'HOME': str(root / 'host-home'),
                                       'PATH': '/usr/bin:/bin', 'PRIVATE_TEST_SECRET': 'not-forwarded',
                                       **roots}, clear=True), patch.object(svp_app, '_is_linux', return_value=True):
                env = svp_app.get_clean_env()
            self.assertNotIn('PRIVATE_TEST_SECRET', env)
            self.assertEqual({key: env[key] for key in roots}, roots)
            script = "import os,pathlib; [pathlib.Path(os.environ[k], 'probe').write_text(k) for k in " \
                     "('XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_CACHE_HOME','XDG_STATE_HOME')]"
            subprocess.run([sys.executable, '-c', script], env=env, check=True)
            for key, path in roots.items():
                self.assertEqual((Path(path) / 'probe').read_text(), key)
            self.assertFalse((root / 'host-home').exists())

    def test_absent_roots_are_not_invented(self):
        with patch.dict(os.environ, {'HOME': '/synthetic/home'}, clear=True), \
             patch.object(svp_app, '_is_linux', return_value=True):
            env = svp_app.get_clean_env()
        self.assertNotIn('XDG_DATA_HOME', env)
        self.assertNotIn('XDG_CONFIG_HOME', env)


if __name__ == '__main__':
    unittest.main()
