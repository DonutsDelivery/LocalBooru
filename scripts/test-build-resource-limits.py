#!/usr/bin/env python3
"""Exercise build safety without starting a compiler or real container."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/build-resource-limits.sh'


class BuildResourceTests(unittest.TestCase):
    def helper(self, command, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith('LOCALBOORU_BUILD_')}
        env.update(overrides)
        return subprocess.run(['bash', '-eu', '-c', 'source "$1"; '+command, 'test', str(HELPER)],
                              env=env, text=True, capture_output=True)

    def test_default_limits(self):
        result = self.helper('localbooru_build_resource_limits; printf "%s\\n" "${LOCALBOORU_CONTAINER_LIMITS[@]}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['--memory', '12g', '--memory-swap', '14g', '--cpus', '2'])

    def test_explicit_limits_and_no_swap(self):
        result = self.helper('localbooru_build_resource_limits; printf "%s\\n" "${LOCALBOORU_CONTAINER_LIMITS[@]}"',
                             LOCALBOORU_BUILD_MEMORY_GB='8', LOCALBOORU_BUILD_SWAP_GB='0', LOCALBOORU_BUILD_CPUS='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['--memory', '8g', '--memory-swap', '8g', '--cpus', '1'])

    def test_invalid_configuration_fails(self):
        for name, value in [('MEMORY_GB', '0'), ('SWAP_GB', '-1'), ('CPUS', 'all'), ('MIN_FREE_GB', '0'), ('ROOT_MIN_FREE_GB', '0'), ('JOBS', '0')]:
            with self.subTest(name=name):
                result = self.helper('localbooru_build_resource_limits', **{'LOCALBOORU_BUILD_'+name: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('LOCALBOORU_BUILD_'+name, result.stderr)

    def test_low_disk_fails_with_actionable_message(self):
        result = self.helper('localbooru_build_resource_limits; localbooru_build_check_disk /', LOCALBOORU_BUILD_MIN_FREE_GB='99999')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('free', result.stderr)
        self.assertIn('refusing build', result.stderr)

    def test_missing_build_path_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix='dmc build safety ') as tmp:
            result = self.helper('localbooru_build_resource_limits; localbooru_build_check_disk "$FIXTURE_PATH"',
                                 LOCALBOORU_BUILD_MIN_FREE_GB='1', LOCALBOORU_BUILD_ROOT_MIN_FREE_GB='1', FIXTURE_PATH=tmp+'/not created/cache')
            self.assertEqual(result.returncode, 0, result.stderr)

    def wrapper(self, platform, minimum='1', run_exit=0):
        with tempfile.TemporaryDirectory(prefix='dmc-build-safety-') as tmp:
            tmp = Path(tmp)
            binary = tmp/'bin'; binary.mkdir()
            docker = binary/'docker'
            docker.write_text('#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ["FAKE_DOCKER_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nsys.exit(int(os.environ.get("FAKE_DOCKER_EXIT", "0")) if sys.argv[1] == "run" else 0)\n')
            docker.chmod(0o755)
            cache = tmp/'cache'; cache.mkdir()
            (cache/'.localbooru-build-cache').write_text('localbooru-build-cache-v1\n')
            output = tmp/'dist'; output.mkdir()
            (output/'SHA256SUMS-Windows').write_text('')
            # Synthetic cache sentinels prevent an implicit native bootstrap.
            patch_hash = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:patches/webkitgtk/2.52.3-playbin-video-filter.patch'])
            import hashlib
            files = ['webkit-build/.localbooru-config-ubuntu24-gtk3-ruby34-v3',
                     'webkit-build/lib/libwebkit2gtk-4.1.so.0', 'webkit-build/lib/libjavascriptcoregtk-4.1.so.0',
                     'webkit-build/bin/WebKitWebProcess', 'webkitgtk-2.52.3/.localbooru-patch-'+hashlib.sha256(patch_hash).hexdigest(),
                     'vapoursynth-stage/.localbooru-vapoursynth-c05906995662bacd5bddf853d8e68f19286987db']
            for name in files:
                p = cache/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic\n'); p.chmod(0o755)
            log = tmp/'docker.jsonl'
            env = {k: v for k, v in os.environ.items() if not k.startswith('LOCALBOORU_BUILD_')}
            env.update(PATH=str(binary)+os.pathsep+env['PATH'], XDG_STATE_HOME=str(tmp/'state'),
                       FAKE_DOCKER_LOG=str(log), FAKE_DOCKER_EXIT=str(run_exit), LOCALBOORU_BUILD_LOCK_TIMEOUT='0', LOCALBOORU_BUILD_MIN_FREE_GB=minimum, LOCALBOORU_BUILD_ROOT_MIN_FREE_GB='1',
                       LOCALBOORU_DOCKER_BUILD_ROOT=str(cache), LOCALBOORU_CCACHE_DIR=str(tmp/'ccache'),
                       LOCALBOORU_DIST_LINUX_DIR=str(output), LOCALBOORU_WINDOWS_BUILD_ROOT=str(cache), LOCALBOORU_DIST_WINDOWS_DIR=str(output))
            # An empty checksum file is only a fake packaging fixture, so fake
            # its host verifier too; production artifact checks stay unchanged.
            sha = binary/'sha256sum'
            sha.write_text('#!/bin/bash\nif [[ "$1" == -c ]]; then exit 0; fi\nexec /usr/bin/sha256sum "$@"\n'); sha.chmod(0o755)
            result = subprocess.run([str(ROOT/'scripts'/f'build-{platform}-local.sh')], env=env, text=True, capture_output=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_both_wrappers_pass_limits_and_one_job(self):
        for platform in ['linux', 'windows']:
            with self.subTest(platform=platform):
                result, calls = self.wrapper(platform)
                self.assertEqual(result.returncode, 0, result.stderr)
                run = next(args for args in calls if args[0] == 'run')
                for option, value in [('--memory', '12g'), ('--memory-swap', '14g'), ('--cpus', '2')]:
                    self.assertEqual(run[run.index(option)+1], value)
                self.assertIn('LOCALBOORU_BUILD_JOBS=1', run)

    def test_failed_container_keeps_failure_status(self):
        for platform in ['linux', 'windows']:
            with self.subTest(platform=platform):
                result, calls = self.wrapper(platform, run_exit=42)
                self.assertEqual(result.returncode, 42, result.stderr)
                self.assertTrue(any(args[0] == 'run' for args in calls))

    def test_both_wrappers_refuse_low_disk_before_container(self):
        for platform in ['linux', 'windows']:
            with self.subTest(platform=platform):
                result, calls = self.wrapper(platform, '99999')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(args[0] in ['run', 'build'] for args in calls))
                self.assertIn('refusing build', result.stderr)


if __name__ == '__main__':
    unittest.main()
