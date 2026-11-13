#!/usr/bin/env python3
"""Hold the Mac-local cross-project build lock for one owned process group."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', default='/tmp/arbit-build.lock')
    parser.add_argument('--wait', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=21600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 <= args.wait <= 21600 or not 1 <= args.timeout <= 21600:
        parser.error('Provide a command and bounded wait/timeout values')
    lock_path = Path(args.lock)
    if lock_path.is_symlink():
        parser.error('Refusing a symlink build lock')
    with lock_path.open('a+') as lock:
        deadline = time.monotonic() + args.wait
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    parser.exit(75, 'Another Mac build owns the shared lock; no work was started.\n')
                time.sleep(0.2)
        environment = os.environ.copy()
        environment['DMC_MAC_BUILD_LOCK_HELD'] = '1'
        environment['CARGO_BUILD_JOBS'] = '1'
        environment['LOCALBOORU_BUILD_JOBS'] = '1'
        child = subprocess.Popen(command, env=environment, start_new_session=True)
        interrupted = None

        def forward(number, _frame):
            nonlocal interrupted
            interrupted = interrupted or time.monotonic()
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass

        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, forward)
        deadline = time.monotonic() + args.timeout
        timed_out = False
        while child.poll() is None:
            if time.monotonic() >= deadline and interrupted is None:
                timed_out = True
                forward(signal.SIGTERM, None)
            if interrupted is not None and time.monotonic() - interrupted >= 10:
                forward(signal.SIGKILL, None)
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        code = 124 if timed_out else (child.returncode if child.returncode >= 0 else 128 - child.returncode)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
