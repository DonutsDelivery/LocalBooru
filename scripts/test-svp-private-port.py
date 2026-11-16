#!/usr/bin/env python3
"""Run the embedded SVP launcher with synthetic ASGI media-free fixtures."""
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request

BOOTSTRAP = Path(__file__).resolve().parents[1] / 'src-tauri/src/addons/svp_sidecar.py'
APP = '''
async def app(scope, receive, send):
    if scope['type'] == 'lifespan':
        while True:
            event = await receive()
            if event['type'] == 'lifespan.startup':
                await send({'type': 'lifespan.startup.complete'})
            elif event['type'] == 'lifespan.shutdown':
                await send({'type': 'lifespan.shutdown.complete'})
                return
    else:
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'synthetic SVP fixture'})
'''


class PrivatePortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dmc-synthetic-svp-port-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'app.py').write_text(APP)
        self.children = []
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for child, log in self.children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
            child.stdout.close()
            log.close()

    def start(self, windows_newlines=False):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        for key in ('HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'XDG_STATE_HOME', 'TMPDIR'):
            path = self.root / key
            path.mkdir(exist_ok=True, mode=0o700)
            env[key] = str(path)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        log = (self.root / f'child-{len(self.children)}.log').open('wb')
        bootstrap = BOOTSTRAP.read_text()
        if windows_newlines:
            bootstrap = "import sys; sys.stdout.reconfigure(newline='\\r\\n')\n" + bootstrap
        child = subprocess.Popen([sys.executable, '-u', '-c', bootstrap], cwd=self.root,
                                 env=env, stdout=subprocess.PIPE, stderr=log)
        self.children.append((child, log))
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(5), 'Private port handshake timed out')
            greeting = child.stdout.readline(64)
        self.assertTrue(greeting.startswith(b'DMC_SVP_PORT=') and greeting.endswith(b'\n'), greeting)
        port = int(greeting.removeprefix(b'DMC_SVP_PORT='))
        self.assertTrue(0 < port <= 65535)
        self.assertEqual(greeting, f'DMC_SVP_PORT={port}\n'.encode('ascii'))
        # The child binds before reporting. Health may follow ASGI startup.
        import time
        deadline = time.monotonic() + 5
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=.2) as response:
                    self.assertEqual(response.read(), b'synthetic SVP fixture')
                break
            except (OSError, TimeoutError):
                self.assertIsNone(child.poll())
                if time.monotonic() >= deadline:
                    raise
        return child, port

    def test_foreign_listener_remains_owned_and_unmodified(self):
        with socket.socket() as foreign:
            foreign.bind(('127.0.0.1', 0)); foreign.listen()
            address = foreign.getsockname()
            child, port = self.start()
            self.assertNotEqual(port, address[1])
            self.assertEqual(foreign.getsockname(), address)
            self.assertEqual(foreign.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)
            self.assertIsNone(child.poll())

    def test_independent_instances_get_distinct_endpoints(self):
        first, first_port = self.start()
        second, second_port = self.start()
        self.assertNotEqual(first_port, second_port)
        self.assertIsNone(first.poll()); self.assertIsNone(second.poll())

    def test_windows_text_newlines_cannot_change_wire_handshake(self):
        child, port = self.start(windows_newlines=True)
        self.assertIsNone(child.poll())
        self.assertTrue(0 < port <= 65535)

    def test_stopping_child_releases_only_its_socket(self):
        with socket.socket() as foreign:
            foreign.bind(('127.0.0.1', 0)); foreign.listen()
            child, port = self.start()
            child.terminate(); child.wait(timeout=5)
            with socket.socket() as replacement:
                # Accepted HTTP connections may remain in TCP TIME_WAIT. The
                # listener must be gone; an immediate rebind is not required.
                self.assertNotEqual(replacement.connect_ex(('127.0.0.1', port)), 0)
            self.assertEqual(foreign.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)


if __name__ == '__main__':
    unittest.main()
