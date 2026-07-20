#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/dist}"
revision=20cb34a20a83c72c87a991d2c949032c70085b16
work="${LADA_BUILD_DIR:-$root/build/release}"
rm -rf "$work"
mkdir -p "$work" "$out"

git clone --filter=blob:none --no-checkout https://github.com/ladaapp/lada.git "$work/upstream-checkout"
git -C "$work/upstream-checkout" checkout --detach "$revision"
test "$(git -C "$work/upstream-checkout" rev-parse HEAD)" = "$revision"
printf '%s  %s\n' \
  b330c6d25dbcbe32c9463dd4b1ec5416c198dd3034e056bd9bd26f9416e22ec1 "$work/upstream-checkout/uv.lock" \
  6a6e5f15a3ca671ae04eb7128d983aab87448122b2cca7ffcd498975000fc9e3 "$work/upstream-checkout/pyproject.toml" \
  | sha256sum --check --status

"$root/packaging/stage-source.sh" "$root" "$work/upstream-checkout" "$work/corresponding-source"
staged_addon="$work/corresponding-source/localbooru-lada-addon"
staged_lada="$work/corresponding-source/lada"
adapter_wheel="$("$root/packaging/build-adapter-wheel.sh" "$staged_addon" "$work/adapter-wheel" | tail -n 1)"

export UV_PYTHON_INSTALL_DIR="$work/python"
uv python install 3.12
template_parent="$work/template"
template_runtime="$template_parent/runtime"
uv venv --python 3.12 --relocatable "$template_runtime"

cuda_variant="${LADA_CUDA_VARIANT:-cuda}"
case "$cuda_variant" in
  cuda) cuda_extra=nvidia ;;
  cuda-legacy) cuda_extra=nvidia-legacy ;;
  *) printf 'unsupported LADA_CUDA_VARIANT: %s\n' "$cuda_variant" >&2; exit 2 ;;
esac

mkdir -p "$work/model-bundle/models"
python - "$staged_addon/manifests/models.json" "$work/model-bundle/models" <<'PY'
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

declare -A extras=( [cuda]="$cuda_extra" [xpu]=intel )
for backend in cuda xpu; do
  full_parent="$work/full-$backend"
  full_runtime="$full_parent/runtime"
  cp -a "$template_parent" "$full_parent"
  VIRTUAL_ENV="$full_runtime" uv sync \
    --project "$staged_lada" --active --frozen --extra "${extras[$backend]}" \
    --no-install-project --inexact
  VIRTUAL_ENV="$full_runtime" uv pip install \
    --python "$full_runtime/bin/python" --no-deps "$staged_lada"
  VIRTUAL_ENV="$full_runtime" uv pip install \
    --python "$full_runtime/bin/python" --no-deps "$adapter_wheel"
  cp "$staged_addon/LICENSE" "$full_runtime/AGPL-3.0-only.txt"
  cp "$staged_addon/THIRD_PARTY_NOTICES.md" "$full_runtime/THIRD_PARTY_NOTICES.md"
  cp "$staged_addon/manifests/addon.json" "$full_runtime/addon.json"
  cp "$staged_addon/manifests/models.json" "$full_runtime/models.json"
  "$full_runtime/bin/python" -c 'import torch, torchvision'
done

common_parent="$work/common"
common_runtime="$common_parent/runtime"
PYTHONPATH="$staged_addon/src" "$template_runtime/bin/python" -m localbooru_lada build-common \
  --cuda "$work/full-cuda/runtime" \
  --xpu "$work/full-xpu/runtime" \
  --output "$common_runtime"
if find "$common_runtime" -type d -path '*/site-packages/torch' -print -quit | grep -q .; then
  printf 'common runtime unexpectedly contains PyTorch\n' >&2
  exit 1
fi
for backend in cuda xpu; do
  layer_parent="$work/layer-$backend"
  PYTHONPATH="$staged_addon/src" "$template_runtime/bin/python" -m localbooru_lada build-layer \
    --base "$common_runtime" \
    --complete "$work/full-$backend/runtime" \
    --output "$layer_parent/runtime"
done

tar_args=(--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -I 'zstd -19 -T0')
tar "${tar_args[@]}" -cf "$out/linux-x86_64-common.tar.zst" -C "$common_parent" runtime
tar "${tar_args[@]}" -cf "$out/linux-x86_64-cuda.tar.zst" -C "$work/layer-cuda" runtime
tar "${tar_args[@]}" -cf "$out/linux-x86_64-xpu.tar.zst" -C "$work/layer-xpu" runtime
tar "${tar_args[@]}" -cf "$out/models.tar.zst" -C "$work/model-bundle" models
tar "${tar_args[@]}" -cf "$out/source.tar.zst" -C "$work/corresponding-source" \
  localbooru-lada-addon lada

size_of() {
  du -sb "$1" | cut -f1
}
PYTHONPATH="$staged_addon/src" "$template_runtime/bin/python" -m localbooru_lada build-manifest \
  --root "$staged_addon" \
  --source-archive "$out/source.tar.zst" \
  --bundle "linux_x86_64_common=$out/linux-x86_64-common.tar.zst" \
  --bundle "linux_x86_64_cuda=$out/linux-x86_64-cuda.tar.zst" \
  --bundle "linux_x86_64_xpu=$out/linux-x86_64-xpu.tar.zst" \
  --bundle "model_bundle=$out/models.tar.zst" \
  --installed-size "linux_x86_64_common=$(size_of "$common_runtime")" \
  --installed-size "linux_x86_64_cuda=$(size_of "$work/layer-cuda/runtime")" \
  --installed-size "linux_x86_64_xpu=$(size_of "$work/layer-xpu/runtime")" \
  --installed-size "model_bundle=$(size_of "$work/model-bundle/models")" \
  --output "$out/release-manifest.json"

printf '%s\n' "$out/release-manifest.json"
