#!/usr/bin/env bash
set -euo pipefail

root="$(cd "${1:?add-on root is required}" && pwd)"
upstream="$(cd "${2:?upstream LADA checkout is required}" && pwd)"
stage="${3:?staging directory is required}"
repo="$(git -C "$root" rev-parse --show-toplevel)"
relative_root="${root#"$repo"/}"

rm -rf "$stage"
mkdir -p "$stage/localbooru-lada-addon" "$stage/lada"
git -C "$repo" archive HEAD:"$relative_root" | tar -xf - -C "$stage/localbooru-lada-addon"
git -C "$upstream" archive HEAD | tar -xf - -C "$stage/lada"
