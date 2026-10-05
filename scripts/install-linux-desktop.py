#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Register DMC's desktop identity and icon without compiling or launching it."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
APP_ID = "com.localbooru.app"
LEGACY_IDS = ("localbooru.desktop", "DonutMediaCenter.desktop", "donutmediacenter.desktop")


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".dmc-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def quoted_exec(path):
    if any(character in path for character in "\n\r\0"):
        raise ValueError("Launcher path must fit on one desktop-entry line")
    # Exec quoting is applied after the desktop file's general string unescaping.
    escaped = path.replace("%", "%%")
    for character in ('\\', '"', '`', '$'):
        escaped = escaped.replace(character, '\\' + character)
    return ('"' + escaped + '"').replace('\\', '\\\\')


def refresh(data_dir):
    for command in (
        ["update-desktop-database", str(data_dir / "applications")],
    ):
        if shutil.which(command[0]):
            subprocess.run(command, check=False, stdout=subprocess.DEVNULL)
    # Keep an existing GNOME pin attached to the canonical window identity.
    if "GNOME" in os.environ.get("XDG_CURRENT_DESKTOP", "").upper() and shutil.which("gsettings"):
        result = subprocess.run(["gsettings", "get", "org.gnome.shell", "favorite-apps"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            try:
                favorites = ast.literal_eval(result.stdout)
            except (ValueError, SyntaxError):
                return
            if isinstance(favorites, list) and any(item in LEGACY_IDS for item in favorites):
                updated = []
                for item in favorites:
                    item = APP_ID + ".desktop" if item in LEGACY_IDS else item
                    if item not in updated:
                        updated.append(item)
                subprocess.run(["gsettings", "set", "org.gnome.shell", "favorite-apps",
                                json.dumps(updated)], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Staged usr/share directory; skips host cache updates")
    parser.add_argument("--launcher", default=str(ROOT / "run.sh"))
    args = parser.parse_args()
    staged = args.data_dir is not None
    data_dir = args.data_dir or Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    applications = data_dir / "applications"
    text = (ROOT / "src-tauri/linux/com.localbooru.app.desktop").read_text()
    text = text.replace("Exec=donutmediacenter", "Exec=" + quoted_exec(args.launcher))
    icon = data_dir / "icons/hicolor/512x512/apps" / (APP_ID + ".png")
    atomic_write(icon, (ROOT / "assets/icon.png").read_bytes())
    if not staged:
        text = text.replace("Icon=" + APP_ID, "Icon=" + str(icon))
    atomic_write(applications / (APP_ID + ".desktop"), text.encode())
    for name in LEGACY_IDS:
        legacy = applications / name
        if legacy.is_file() and "Name=DonutMediaCenter\n" in legacy.read_text():
            if staged:
                legacy.unlink()
            else:
                # Old bookmarks and launchers continue working without duplicate app-menu entries.
                lines = legacy.read_text().splitlines()
                lines = [line for line in lines if not line.startswith(("Icon=", "StartupWMClass=", "NoDisplay="))]
                lines[lines.index("[Desktop Entry]") + 1:lines.index("[Desktop Entry]") + 1] = [
                    "Icon=" + APP_ID, "StartupWMClass=localbooru", "NoDisplay=true"]
                atomic_write(legacy, ("\n".join(lines) + "\n").encode())
    if not staged:
        refresh(data_dir)
    print("Registered DonutMediaCenter desktop entry and icon:", applications / (APP_ID + ".desktop"))


if __name__ == "__main__":
    main()
