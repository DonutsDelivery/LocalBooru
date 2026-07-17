#!/usr/bin/env python3
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def read_text(path: str) -> str:
    return (ROOT / path).read_text()


def fail(label: str, actual, expected) -> None:
    print(f"ERROR: {label} is {actual!r}, expected {expected!r}", file=sys.stderr)
    raise SystemExit(1)


root_package = read_json("package.json")
expected = root_package["version"]
checks = {
    "package-lock.json": read_json("package-lock.json")["version"],
    "package-lock root package": read_json("package-lock.json")["packages"][""]["version"],
    "frontend/package.json": read_json("frontend/package.json")["version"],
    "frontend/package-lock.json": read_json("frontend/package-lock.json")["version"],
    "frontend lock root package": read_json("frontend/package-lock.json")["packages"][""]["version"],
    "Cargo workspace": tomllib.loads(read_text("Cargo.toml"))["workspace"]["package"]["version"],
    "src-tauri/Cargo.toml": tomllib.loads(read_text("src-tauri/Cargo.toml"))["package"]["version"],
    "src-tauri/tauri.conf.json": read_json("src-tauri/tauri.conf.json")["version"],
}

cargo_packages = tomllib.loads(read_text("Cargo.lock"))["package"]
localbooru_packages = [package for package in cargo_packages if package["name"] == "localbooru"]
if len(localbooru_packages) != 1:
    fail("Cargo.lock LocalBooru package count", len(localbooru_packages), 1)
checks["Cargo.lock LocalBooru package"] = localbooru_packages[0]["version"]

for label, actual in checks.items():
    if actual != expected:
        fail(label, actual, expected)

literal_checks = {
    "frontend updater fallback": ("frontend/src/services/appUpdater.js", rf"currentVersion: '{re.escape(expected)}'.*latestVersion: '{re.escape(expected)}'"),
    "Rust app-version fallback": ("src-tauri/src/commands.rs", rf'unwrap_or_else\(\|\| "{re.escape(expected)}"\.to_string\(\)\)'),
    "macOS bundle assertion": ("scripts/build-macos-ci.sh", rf'CFBundleShortVersionString[^\n]+"{re.escape(expected)}"'),
    "Android version name": ("frontend/android/app/build.gradle", rf'versionName "{re.escape(expected)}"'),
}
for label, (path, pattern) in literal_checks.items():
    if not re.search(pattern, read_text(path)):
        fail(label, "missing", pattern)

macos_config = read_json("src-tauri/tauri.macos.conf.json")
identity = macos_config["bundle"]["macOS"].get("signingIdentity")
if identity != "-":
    fail("macOS signing identity", identity, "-")

print(f"LocalBooru release metadata is consistent at {expected}")
