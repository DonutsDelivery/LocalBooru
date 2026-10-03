"""Synthetic platform discovery checks; never reads a user installation."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
import types

spec = importlib.util.spec_from_file_location("donut_create_installer", Path(__file__).parents[1] / "installer.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PythonDiscovery(unittest.TestCase):
    # AC: @donut-create-plugin ac-managed-setup
    def test_mac_discovery_without_shell_path(self):
        for expected in (
            "/opt/homebrew/bin/python3.12",
            "/usr/local/bin/python3.12",
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
        ):
            with self.subTest(expected=expected), patch.object(installer.sys, "version_info", (3, 9, 0)), \
                    patch.object(installer.platform, "system", return_value="Darwin"), \
                    patch.object(installer.shutil, "which", return_value=None), \
                    patch.object(Path, "is_file", lambda path: str(path) == expected), \
                    patch.object(installer.os, "access", return_value=True):
                self.assertEqual(installer.Installer.__new__(installer.Installer)._python_command(), [expected])

    def test_preserves_controller_python_312(self):
        with patch.object(installer.sys, "version_info", (3, 12, 4)), \
                patch.object(installer.sys, "executable", "/synthetic/App Support/日本語/bin/python"):
            self.assertEqual(installer.Installer.__new__(installer.Installer)._python_command(),
                             ["/synthetic/App Support/日本語/bin/python"])

    def test_missing_python_reports_requirement(self):
        with patch.object(installer.sys, "version_info", (3, 9, 0)), \
                patch.object(installer.platform, "system", return_value="Darwin"), \
                patch.object(installer.shutil, "which", return_value=None), \
                patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(installer.SetupError, "Install Python 3.12"):
                installer.Installer.__new__(installer.Installer)._python_command()

    def test_downloads_add_controller_roots_without_disabling_verification(self):
        context = Mock(check_hostname=True, verify_mode=installer.ssl.CERT_REQUIRED)
        certifi = types.SimpleNamespace(where=lambda: "/synthetic/CA roots/cacert.pem")
        with patch.dict(installer.sys.modules, {"certifi": certifi}), \
                patch.object(installer.ssl, "create_default_context", return_value=context), \
                patch.object(installer.urllib.request, "HTTPSHandler") as handler, \
                patch.object(installer.urllib.request, "build_opener"):
            installer.download_opener()
            context.load_verify_locations.assert_called_once_with(cafile="/synthetic/CA roots/cacert.pem")
            handler.assert_called_once_with(context=context)
            self.assertTrue(context.check_hostname)
            self.assertEqual(context.verify_mode, installer.ssl.CERT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
