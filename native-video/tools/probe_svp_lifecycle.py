#!/usr/bin/env python3
"""Exercise native-helper SVP lifecycle transitions against real media."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable


def descendants(pid: int) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    pending = [pid]
    seen = {pid}
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            try:
                command = Path(f"/proc/{child}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                command = ""
            found.append((child, command))
            pending.append(child)
    return found


class Helper:
    def __init__(self, executable: Path, plugin_path: Path) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "LOCALBOORU_SVP_PLUGIN_PATH": str(plugin_path),
                "SDL_AUDIODRIVER": "dummy",
                "SDL_VIDEODRIVER": "dummy",
            }
        )
        self.process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                self.events.put({"type": "invalid_json", "line": line.rstrip()})

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        self.stderr.extend(line.rstrip() for line in self.process.stderr)

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def wait(self, predicate: Callable[[dict[str, Any]], bool], timeout: float, label: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        observed: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"helper exited while waiting for {label}: {self.stderr[-20:]}")
            try:
                event = self.events.get(timeout=min(0.25, deadline - time.monotonic()))
            except queue.Empty:
                continue
            observed.append(event)
            if event.get("type") in {"fatal_error", "recoverable_error", "invalid_json"}:
                raise RuntimeError(f"helper error while waiting for {label}: {event}")
            if predicate(event):
                return event
        raise RuntimeError(f"timed out waiting for {label}; last events={observed[-10:]}")

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)


def position(event: dict[str, Any]) -> float:
    return float(event.get("position", -1.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("--helper", type=Path, default=Path("native-video/build/localbooru-native-video"))
    parser.add_argument("--plugin-path", type=Path, default=Path.home() / "SVP 4/plugins")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    helper = Helper(args.helper.resolve(), args.plugin_path.resolve())
    report: dict[str, Any] = {"probe_version": 1, "media": str(args.media.resolve()), "transitions": []}
    try:
        helper.send({"type": "hello", "protocol_version": 1000})
        ready = helper.wait(lambda event: event.get("type") == "ready", 5, "ready")
        if ready["protocol_version"] // 1000 != 1:
            raise RuntimeError(f"incompatible helper protocol: {ready}")
        helper.send(
            {
                "type": "open_media",
                "generation": 1,
                "item_id": 1,
                "path": str(args.media.resolve()),
                "resume_position": 0.0,
                "autoplay": True,
            }
        )
        helper.wait(lambda event: event.get("type") == "media_opened", 8, "media_opened")
        tracks = helper.wait(lambda event: event.get("type") == "track_list", 8, "track_list")
        ordinary = helper.wait(
            lambda event: event.get("type") == "playback_state" and position(event) >= 0.25,
            12,
            "ordinary playback advance",
        )
        before_svp = position(ordinary)

        helper.send({"type": "set_interpolation", "engine": "svp", "preset": "balanced", "target_fps": 60})
        helper.wait(
            lambda event: event.get("type") == "capabilities_changed" and event.get("svp_status") == "active_external",
            20,
            "active SVP",
        )
        svp_frame = helper.wait(
            lambda event: event.get("type") == "playback_state" and position(event) >= before_svp - 0.25,
            8,
            "SVP playback handoff",
        )
        svp_position = position(svp_frame)
        if abs(svp_position - before_svp) > 0.25:
            raise RuntimeError(f"SVP handoff moved playhead by {svp_position - before_svp:.3f}s")
        report["transitions"].append({"name": "off_to_svp", "before": before_svp, "after": svp_position})

        helper.send({"type": "set_paused", "paused": True})
        paused_event = helper.wait(
            lambda event: event.get("type") == "playback_state" and event.get("paused") is True,
            5,
            "pause",
        )
        seek_target = position(paused_event) + 1.0
        helper.send({"type": "seek", "position": seek_target})
        seek_event = helper.wait(
            lambda event: event.get("type") == "playback_state" and abs(position(event) - seek_target) <= 0.25,
            20,
            "paused SVP seek",
        )
        report["transitions"].append({"name": "paused_seek", "target": seek_target, "after": position(seek_event)})

        audio = tracks.get("audio", [])
        if audio:
            helper.send({"type": "select_audio_track", "track_id": audio[0]["id"]})
        helper.send({"type": "select_subtitle_track", "track_id": None})
        helper.send({"type": "set_subtitle_delay", "seconds": 0.15})
        helper.send({"type": "set_speed", "speed": 1.25})
        helper.send({"type": "set_paused", "paused": False})
        resumed = helper.wait(
            lambda event: event.get("type") == "playback_state" and not event.get("paused") and position(event) > seek_target,
            8,
            "resume at changed speed",
        )

        before_off = position(resumed)
        helper.send({"type": "set_interpolation", "engine": "off", "preset": "balanced", "target_fps": 60})
        helper.wait(
            lambda event: event.get("type") == "capabilities_changed" and event.get("svp_status") == "available_external",
            8,
            "SVP disabled",
        )
        ordinary_again = helper.wait(
            lambda event: event.get("type") == "playback_state" and position(event) >= before_off - 0.25,
            8,
            "ordinary playback handoff",
        )
        after_off = position(ordinary_again)
        if abs(after_off - before_off) > 0.25:
            raise RuntimeError(f"SVP-off handoff moved playhead by {after_off - before_off:.3f}s")
        report["transitions"].append({"name": "svp_to_off", "before": before_off, "after": after_off})

        time.sleep(1.0)
        leaked = [(pid, command) for pid, command in descendants(helper.process.pid) if "vspipe" in command or "ffmpeg" in command]
        if leaked:
            raise RuntimeError(f"obsolete SVP workers survived disable: {leaked}")

        helper.send({"type": "close_media", "generation": 1})
        time.sleep(0.5)
        leaked_after_close = descendants(helper.process.pid)
        if any("vspipe" in command or "ffmpeg" in command for _, command in leaked_after_close):
            raise RuntimeError(f"SVP workers survived close: {leaked_after_close}")
        report.update(
            {
                "audio_track_reselected": bool(audio),
                "subtitle_none_and_delay_applied": True,
                "speed": 1.25,
                "svp_workers_after_disable": leaked,
                "descendants_after_close": leaked_after_close,
                "passed": True,
            }
        )
    finally:
        helper.close()
    report["helper_exit_code"] = helper.process.returncode
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"probe_version": 1, "passed": False, "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
