#!/usr/bin/env bash
# In-container Windows x64 build and verification for LocalBooru.
set -euo pipefail

SOURCE_DIR="${LOCALBOORU_SOURCE_DIR:-/source}"
BUILD_ROOT="${LOCALBOORU_WINDOWS_BUILD_ROOT:-/build}"
WORKTREE="$BUILD_ROOT/worktree"
TARGET="x86_64-pc-windows-msvc"
TARGET_DIR="$BUILD_ROOT/target"
RELEASE_DIR="$TARGET_DIR/$TARGET/release"
DIST_DIR="${LOCALBOORU_DIST_WINDOWS_DIR:-/dist}"
JOBS="${LOCALBOORU_BUILD_JOBS:-$(nproc)}"

export CARGO_HOME="$BUILD_ROOT/cargo-home"
export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_BUILD_JOBS="$JOBS"
export SCCACHE_DIR="${SCCACHE_DIR:-/ccache}"
# Ubuntu's packaged sccache cannot detect MSVC /showIncludes output through
# msvc-wine, and cc-rs automatically applies RUSTC_WRAPPER to Windows resource
# compilation. Preserve the cache volume for a future compatible sccache, but
# do not let it break cl/rc builds today.
unset RUSTC_WRAPPER
export PATH="/opt/msvc/bin/x64:/opt/node/bin:/root/.cargo/bin:$PATH"
export CC_x86_64_pc_windows_msvc=cl
export CXX_x86_64_pc_windows_msvc=cl
export AR_x86_64_pc_windows_msvc=/opt/msvc/bin/x64/lib
# msvc-wine's optional msvctricks FIFO capture can deadlock after the Wine
# child exits before opening its pipes. Raw mode preserves linker output and
# bypasses that helper without changing the MSVC toolchain.
export WINE_MSVC_RAW_STDOUT=1

# Export native Unix paths for the genuine MSVC/Windows SDK libraries, then
# use LLD in MSVC mode for Rust's very large final link. Passing that link
# through Wine exceeds its reliable command/output path for this application.
BIN=/opt/msvc/bin/x64 . /opt/msvc-wine/msvcenv-native.sh
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=/usr/bin/lld-link

cleanup_ownership() {
  if [[ "${HOST_UID:-0}" != 0 ]]; then
    chown -R "${HOST_UID}:${HOST_GID:-$HOST_UID}" \
      "$DIST_DIR" "$BUILD_ROOT" "$SCCACHE_DIR" 2>/dev/null || true
  fi
}
trap cleanup_ownership EXIT

rm -rf "$WORKTREE"
mkdir -p "$WORKTREE" "$TARGET_DIR" "$CARGO_HOME" "$DIST_DIR" "$SCCACHE_DIR"
git -c safe.directory="$SOURCE_DIR" -C "$SOURCE_DIR" archive --format=tar HEAD \
  | tar -xf - -C "$WORKTREE"

cd "$WORKTREE"
rm -rf "$DIST_DIR"/*
LOCK_HASH_BEFORE="$(sha256sum Cargo.lock | cut -d' ' -f1)"

printf 'Windows Docker toolchain:\n'
rustc --version
cargo --version
cargo tauri --version
node --version
npm --version
makensis -VERSION
command -v cl link lib rc llvm-rc >/dev/null
wineserver -k 2>/dev/null || true
wineserver -p
cl >/tmp/localbooru-msvc-version.log 2>&1 || true
sed -n '1,4p' /tmp/localbooru-msvc-version.log
sccache --start-server >/dev/null 2>&1 || true

npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build

cargo check --locked --manifest-path src-tauri/Cargo.toml --target "$TARGET"
cargo tauri build \
  --ci \
  --target "$TARGET" \
  --config src-tauri/tauri.windows.conf.json
[[ "$(sha256sum Cargo.lock | cut -d' ' -f1)" == "$LOCK_HASH_BEFORE" ]]

BINARY="$RELEASE_DIR/localbooru.exe"
INSTALLER="$(find "$RELEASE_DIR/bundle/nsis" -maxdepth 1 -type f -name '*.exe' -print -quit)"
[[ -s "$BINARY" ]]
[[ -n "$INSTALLER" && -s "$INSTALLER" ]]

STAGE="$BUILD_ROOT/portable-stage"
INSTALLER_PAYLOAD="$BUILD_ROOT/nsis-payload"
rm -rf "$STAGE" "$INSTALLER_PAYLOAD"
mkdir -p "$STAGE" "$INSTALLER_PAYLOAD"
cp "$BINARY" "$STAGE/LocalBooru.exe"
cp LICENSE "$STAGE/LICENSE"

(
  cd "$STAGE"
  zip -9 -q "$DIST_DIR/LocalBooru-Windows.zip" LocalBooru.exe LICENSE
)
cp "$INSTALLER" "$DIST_DIR/LocalBooru-Windows-Setup.exe"

unzip -t "$DIST_DIR/LocalBooru-Windows.zip"
7z t "$DIST_DIR/LocalBooru-Windows-Setup.exe"
7z x -y -o"$INSTALLER_PAYLOAD" "$DIST_DIR/LocalBooru-Windows-Setup.exe" >/dev/null

python3 - "$STAGE/LocalBooru.exe" "$INSTALLER_PAYLOAD" <<'PY'
import pathlib, struct, sys
standalone = pathlib.Path(sys.argv[1])
payload = pathlib.Path(sys.argv[2])
forbidden = [
    b'/home/user', b'/mnt/storage', b'/build/worktree', b'/source/',
    b'C:\\a\\LocalBooru\\LocalBooru',
]

def machine(path):
    data = path.read_bytes()
    assert data[:2] == b'MZ', f'{path}: missing MZ header'
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    assert data[pe:pe + 4] == b'PE\0\0', f'{path}: missing PE signature'
    value = struct.unpack_from('<H', data, pe + 4)[0]
    return data, value

data, value = machine(standalone)
assert value == 0x8664, f'{standalone}: expected x64 PE, got {value:#x}'
for needle in forbidden:
    assert needle not in data, f'{standalone}: contains forbidden path {needle!r}'

installed = []
for path in payload.rglob('*.exe'):
    try:
        _, value = machine(path)
    except (AssertionError, OSError, struct.error):
        continue
    if path.name.lower() == 'localbooru.exe':
        assert value == 0x8664, f'{path}: expected x64 PE, got {value:#x}'
        installed.append(path)
assert installed, 'NSIS payload is missing x64 LocalBooru.exe'
print('Standalone and NSIS payload contain x64 PE32+ LocalBooru executables')
PY

(
  cd "$DIST_DIR"
  sha256sum LocalBooru-Windows-Setup.exe LocalBooru-Windows.zip > SHA256SUMS-Windows
  sha256sum -c SHA256SUMS-Windows
)

printf 'Windows Authenticode state (unsigned is expected for local builds):\n'
file "$STAGE/LocalBooru.exe" "$DIST_DIR/LocalBooru-Windows-Setup.exe"
sccache --show-stats || true
printf 'Windows Docker artifacts verified:\n'
find "$DIST_DIR" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' | sort
