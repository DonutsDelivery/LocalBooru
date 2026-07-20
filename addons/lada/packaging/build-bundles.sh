#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/dist}"
backend="${LADA_BACKEND:-cuda}"
case "$backend" in
  cuda) extra=nvidia ;;
  cuda-legacy) extra=nvidia-legacy ;;
  xpu) extra=intel ;;
  *) printf 'unsupported LADA_BACKEND: %s\n' "$backend" >&2; exit 2 ;;
esac

revision=20cb34a20a83c72c87a991d2c949032c70085b16
work="${LADA_BUILD_DIR:-$root/build/$backend}"
rm -rf "$work"
mkdir -p "$work" "$out"

git clone --filter=blob:none --no-checkout https://github.com/ladaapp/lada.git "$work/lada"
git -C "$work/lada" checkout --detach "$revision"
test "$(git -C "$work/lada" rev-parse HEAD)" = "$revision"
printf '%s  %s\n' \
  b330c6d25dbcbe32c9463dd4b1ec5416c198dd3034e056bd9bd26f9416e22ec1 "$work/lada/uv.lock" \
  6a6e5f15a3ca671ae04eb7128d983aab87448122b2cca7ffcd498975000fc9e3 "$work/lada/pyproject.toml" \
  | sha256sum --check --status

export UV_PYTHON_INSTALL_DIR="$work/python"
uv python install 3.12
uv venv --python 3.12 --relocatable "$work/runtime"
VIRTUAL_ENV="$work/runtime" uv sync --project "$work/lada" --active --frozen --extra "$extra" --no-install-project
VIRTUAL_ENV="$work/runtime" uv pip install --python "$work/runtime/bin/python" --no-deps "$work/lada"
VIRTUAL_ENV="$work/runtime" uv pip install --python "$work/runtime/bin/python" --no-deps "$root"

mkdir -p "$work/models"
python - "$root/manifests/models.json" "$work/models" <<'PY'
import hashlib, json, pathlib, sys, urllib.request
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
destination = pathlib.Path(sys.argv[2])
for model in manifest["models"]:
    target = destination / model["name"]
    urllib.request.urlretrieve(model["source_url"], target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != model["sha256"] or target.stat().st_size != model["size"]:
        raise SystemExit(f"model verification failed: {model['name']}")
PY

cp "$root/LICENSE" "$work/runtime/AGPL-3.0-only.txt"
cp "$root/THIRD_PARTY_NOTICES.md" "$work/runtime/THIRD_PARTY_NOTICES.md"
cp "$root/manifests/addon.json" "$work/runtime/addon.json"
cp "$root/manifests/models.json" "$work/runtime/models.json"

tar_args=(--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -I 'zstd -19 -T0')
tar "${tar_args[@]}" -cf "$out/linux-x86_64-$backend.tar.zst" -C "$work" runtime
tar "${tar_args[@]}" -cf "$out/models.tar.zst" -C "$work" models
tar "${tar_args[@]}" -cf "$out/source.tar.zst" -C "$(dirname "$root")" "$(basename "$root")" -C "$work" lada

PYTHONPATH="$root/src" "$work/runtime/bin/python" -m localbooru_lada build-manifest \
  --root "$root" \
  --source-archive "$out/source.tar.zst" \
  --bundle "linux_x86_64_$backend=$out/linux-x86_64-$backend.tar.zst" \
  --bundle "models=$out/models.tar.zst" \
  --output "$out/release-manifest.json"

printf '%s\n' "$out/release-manifest.json"
