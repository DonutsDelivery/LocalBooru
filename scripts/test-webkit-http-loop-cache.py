#!/usr/bin/env python3
"""Exercise the real cache preparation with owned public-source copies and fake CMake."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
RELATIVE = Path('Source/WebCore/platform/graphics/gstreamer/MediaPlayerPrivateGStreamer.cpp')
PATCHES = ('2.52.3-playbin-video-filter.patch', '2.52.3-flatpak-video-bridge.patch',
           '2.52.3-bundled-process-directory.patch', '2.52.3-http-loop-cache-upgrade.patch')
SOURCE = None


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dmc-synthetic-native-cache-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'recipe'
        for directory in ('scripts', 'patches/webkitgtk'):
            (self.repo / directory).mkdir(parents=True)
        for name in ('prepare-webkit-flatpak-bridge.py', 'webkit-http-loop-cache.sh'):
            shutil.copyfile(REPO / 'scripts' / name, self.repo / 'scripts' / name)
        for name in PATCHES:
            shutil.copyfile(REPO / 'patches/webkitgtk' / name, self.repo / 'patches/webkitgtk' / name)
        upgrade = (self.repo / 'patches/webkitgtk' / PATCHES[-1]).read_text().splitlines()
        self.previous = next(line.split()[2] for line in upgrade if line.startswith('# previous-stack '))
        self.before, self.after, _ = next(line.split()[2:] for line in upgrade if line.startswith('# preimage '))
        self.primary = hashlib.sha256((self.repo / 'patches/webkitgtk' / PATCHES[0]).read_bytes()).hexdigest()
        self.stack = hashlib.sha256(b''.join((self.repo / 'patches/webkitgtk' / name).read_bytes() for name in PATCHES[:3])).hexdigest()
        self.build = self.root / 'build'
        self.source = self.build / 'webkitgtk-2.52.3'
        for relative in (RELATIVE, Path('Source/WebKit/UIProcess/Launcher/glib/FlatpakLauncher.cpp'),
                         Path('Source/WebKit/Shared/glib/ProcessExecutablePathGLib.cpp')):
            destination = self.source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((SOURCE / relative).read_bytes())
        self.assertEqual(hashlib.sha256((self.source / RELATIVE).read_bytes()).hexdigest(), self.before)
        (self.source / ('.localbooru-patch-' + self.previous)).touch()
        self.native = self.build / 'webkit-build'
        (self.native / 'lib').mkdir(parents=True)
        (self.native / 'bin').mkdir()
        self.warm = self.native / 'keep-owned-warm-object.o'
        self.warm.write_bytes(b'synthetic warm object')
        self.warm_identity = self.warm.stat().st_ino
        for name in ('libwebkit2gtk-4.1.so.0', 'libjavascriptcoregtk-4.1.so.0', 'libwebkit2gtkinjectedbundle.so'):
            (self.native / 'lib' / name).write_bytes(b'synthetic runtime')
        for name in ('WebKitWebProcess', 'WebKitNetworkProcess', 'WebKitGPUProcess'):
            path = self.native / 'bin' / name
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o700)
        (self.native / '.localbooru-config-ubuntu24-gtk3-ruby34-v3').touch()
        (self.native / 'CMakeCache.txt').write_text('ENABLE_GPU_PROCESS:BOOL=ON\n')
        self.stamp = self.native / ('.localbooru-runtime-' + self.stack)
        fake_bin = self.root / 'bin'
        fake_bin.mkdir()
        cmake = fake_bin / 'cmake'
        cmake.write_text('#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\n'
                         'with Path(os.environ["CALL_LOG"]).open("a") as f: f.write(repr(sys.argv[1:])+"\\n")\n'
                         'if os.environ.get("FAIL_CMAKE"): sys.exit(23)\n'
                         'p=Path(sys.argv[2])/"bin/WebKitNetworkProcess"\n'
                         'p.write_text("#!/bin/sh\\nexit 0\\n");p.chmod(0o700)\n')
        cmake.chmod(0o700)
        self.log = self.root / 'cmake.log'
        self.env = {**os.environ, 'ROOT': str(self.repo), 'BUILD': str(self.build),
                    'WEBKIT_VERSION': '2.52.3', 'WEBKIT_SHA256': 'synthetic-not-downloaded',
                    'ALLOW_NATIVE_RUNTIME_BOOTSTRAP': '0', 'NATIVE_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS': '10',
                    'JOBS': '1', 'CALL_LOG': str(self.log), 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH']}
        # Execute the production function, rather than asserting on its source text.
        recipe = (REPO / 'scripts/build-linux-docker.sh').read_text()
        function = 'prepare_webkit() {' + recipe.split('prepare_webkit() {', 1)[1].split('prepare_vapoursynth() {', 1)[0]
        self.driver = self.root / 'prepare.sh'
        self.driver.write_text('set -euo pipefail\ndownload_checked() { :; }\n' + function + '\nprepare_webkit\n')

    def run_prepare(self, fail=False):
        env = dict(self.env)
        if fail:
            env['FAIL_CMAKE'] = '1'
        result = subprocess.run(['bash', str(self.driver)], env=env, text=True, capture_output=True, timeout=15)
        self.assertTrue(self.warm.exists(), result.stdout + result.stderr)
        self.assertEqual(self.warm.stat().st_ino, self.warm_identity)
        self.assertEqual(self.warm.read_bytes(), b'synthetic warm object')
        return result

    def recognize(self, primary=None):
        command = 'source "$ROOT/scripts/webkit-http-loop-cache.sh"; localbooru_webkit_http_loop_upgrade_available "$1" "$2" "$ROOT/patches/webkitgtk/2.52.3-http-loop-cache-upgrade.patch" "$ROOT/scripts/prepare-webkit-flatpak-bridge.py"'
        return subprocess.run(['bash', '-c', command, 'fixture', str(self.source), primary or self.primary],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def run_host(self):
        (self.build / '.localbooru-build-cache').write_text('localbooru-build-cache-v1\n')
        vs = self.build / 'vapoursynth-stage'
        vs.mkdir()
        (vs / '.localbooru-vapoursynth-c05906995662bacd5bddf853d8e68f19286987db').touch()
        docker = self.root / 'bin/docker'
        docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >>"$DOCKER_LOG"\nexit 0\n')
        docker.chmod(0o700)
        env = {**self.env, 'HOME': str(self.root / 'home'), 'XDG_STATE_HOME': str(self.root / 'state'),
               'LOCALBOORU_DOCKER_BUILD_ROOT': str(self.build), 'LOCALBOORU_CCACHE_DIR': str(self.root / 'ccache'),
               'LOCALBOORU_DIST_LINUX_DIR': str(self.root / 'output'), 'LOCALBOORU_SOURCE_REVISION': 'HEAD',
               'LOCALBOORU_BUILD_LOCK_TIMEOUT': '0', 'DOCKER_LOG': str(self.root / 'docker.log'),
               'LOCALBOORU_BUILD_MIN_FREE_GB': '1', 'LOCALBOORU_BUILD_ROOT_MIN_FREE_GB': '1'}
        # Tiny fake containers consume no build space; real build floors stay unchanged.
        return subprocess.run(['bash', str(REPO / 'scripts/build-linux-local.sh'), '--deb'],
                              env=env, text=True, capture_output=True, timeout=15)

    def test_host_recognition_does_not_mutate(self):
        before = {str(p.relative_to(self.build)): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
                  for p in self.build.rglob('*') if p.is_file()}
        self.assertEqual(self.recognize().returncode, 0)
        after = {str(p.relative_to(self.build)): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
                 for p in self.build.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.log.exists())

    def test_host_wrapper_admits_recognized_upgrade_without_mutating_source(self):
        result = self.run_host()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('LOCALBOORU_BUILD_STATUS=STARTED', result.stdout)
        self.assertIn('run --rm ', (self.root / 'docker.log').read_text())
        self.assertEqual(hashlib.sha256((self.source / RELATIVE).read_bytes()).hexdigest(), self.before)

    def test_host_wrapper_refuses_unknown_upgrade_before_container(self):
        path = self.source / RELATIVE
        path.write_bytes(path.read_bytes() + b'\n// synthetic unknown host source\n')
        edited = path.read_bytes()
        result = self.run_host()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LOCALBOORU_BUILD_STATUS=NEEDS_BOOTSTRAP', result.stderr)
        self.assertFalse((self.root / 'docker.log').exists())
        self.assertEqual(path.read_bytes(), edited)

    def test_upgrade_preserves_objects_and_builds_complete_stack(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(hashlib.sha256((self.source / RELATIVE).read_bytes()).hexdigest(), self.after)
        self.assertTrue(self.stamp.exists())
        self.assertTrue((self.source / ('.localbooru-patch-' + self.stack)).exists())
        self.assertFalse((self.source / ('.localbooru-patch-' + self.previous)).exists())
        self.assertEqual(len(self.calls()), 1)
        self.assertIn("'WebKitGPUProcess', '--parallel', '1'", self.calls()[0])

    def test_unknown_edit_refused_without_deletion_or_compilation(self):
        path = self.source / RELATIVE
        path.write_bytes(path.read_bytes() + b'\n// synthetic unrecognized edit\n')
        edited = path.read_bytes()
        self.assertNotEqual(self.recognize().returncode, 0)
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), edited)
        self.assertFalse(self.stamp.exists())
        self.assertEqual(self.calls(), [])

    def test_different_primary_patch_not_recognized(self):
        self.assertNotEqual(self.recognize('0' * 64).returncode, 0)
        self.assertEqual(hashlib.sha256((self.source / RELATIVE).read_bytes()).hexdigest(), self.before)

    def test_compile_failure_retries_then_reuses_completed_runtime(self):
        self.assertEqual(self.run_prepare(fail=True).returncode, 23)
        self.assertFalse(self.stamp.exists())
        self.assertEqual(hashlib.sha256((self.source / RELATIVE).read_bytes()).hexdigest(), self.after)
        self.assertEqual(self.run_prepare().returncode, 0)
        self.assertTrue(self.stamp.exists())
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(self.run_prepare().returncode, 0)
        self.assertEqual(len(self.calls()), 2)

    def test_interrupted_source_upgrade_still_builds(self):
        subprocess.run(['python3', str(self.repo / 'scripts/prepare-webkit-flatpak-bridge.py'), '--source', str(self.source),
                        '--patch', str(self.repo / 'patches/webkitgtk' / PATCHES[-1]), '--apply'], check=True, capture_output=True)
        self.assertEqual(self.recognize().returncode, 0)
        self.assertEqual(self.run_prepare().returncode, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_preimage_invalidates_any_old_completion_before_compile(self):
        self.stamp.touch()
        self.assertEqual(self.run_prepare(fail=True).returncode, 23)
        self.assertFalse(self.stamp.exists())

    def test_completed_cache_with_unknown_edit_refused(self):
        self.assertEqual(self.run_prepare().returncode, 0)
        path = self.source / RELATIVE
        path.write_bytes(path.read_bytes() + b'\n// synthetic edit after build\n')
        self.assertNotEqual(self.run_prepare().returncode, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_missing_packaged_helper_requires_rebuild(self):
        self.assertEqual(self.run_prepare().returncode, 0)
        (self.native / 'bin/WebKitNetworkProcess').unlink()
        self.assertEqual(self.run_prepare().returncode, 0)
        self.assertEqual(len(self.calls()), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Read-only recognized old public WebKit source')
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[__file__, *remaining])
