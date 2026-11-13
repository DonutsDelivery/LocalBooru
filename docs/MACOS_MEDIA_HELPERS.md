# Bundled macOS media helpers

The canonical native build produces **both arm64 and x86_64** FFmpeg and FFprobe
and bundles them in `Contents/Resources/bin`. Finder launches therefore do not
require the user's shell PATH or a Homebrew installation. The source build uses
Apple's system frameworks, static x264 and static libwebp; it rejects non-system
dynamic libraries, unexpected runtime search paths and missing architectures.
The default minimum macOS version remains 11.0.

`scripts/macos-media-sources.json` pins the exact source archives and SHA-256
values. FFmpeg includes GPL and version-3 code, x264 and WebP encoding, native AAC,
VideoToolbox and AudioToolbox. Nonfree components are not enabled. NASM is built
as a local build tool from its own pinned source, never installed globally or
shipped as an application helper. Build prerequisites are Xcode's selected Apple
SDK and command-line tools, Python 3.12+, make, Perl and pkg-config.

Run the canonical build from a clean committed checkout. It holds the shared
Mac-local `/tmp/arbit-build.lock` with one compiler job. `CHECK_ONLY` source checks
skip helper compilation. The artifact directory must be fresh: old artifacts
are preserved rather than removed. The persistent helper cache must have no
whitespace in its path because of upstream configure flag parsing; this limit
does not apply to user media or application paths.

For a helper-only native build, outside any source checkout:

```sh
python3 scripts/macos-build-lock.py -- \
  python3 scripts/build-macos-media.py \
    --cache /absolute/owned/cache --output /absolute/fresh/helper-output
```

The cache keeps only recognized source/build directories; it refuses unknown
nonempty roots and altered source hashes. Every extracted source file, executable
mode and directory entry is compared with its pinned archive before reuse and
again before the receipt is emitted; a matching marker cannot hide an edited
cache tree. Failed or partial output directories
are never accepted as completed payloads. Choose a fresh output for a retry and
retain the warm compilation cache. No user profiles or libraries are inputs.

Before packaging, the actual bundled helpers must match the build receipt and
pass synthetic H.264/AAC encode, decode and WebP-thumbnail checks. The lite
helpers are signed with **only** `app-sandbox` and `inherit` entitlements. This
inherits static sandbox rights, **not** Powerbox grants acquired after launch.
The parent explicitly passes its native private bookmark-store directory to
each media command, and the linked constructor in both helpers resolves and
activates those bookmarks inside the child. Lite refuses host/PATH helper
fallbacks. Full editions remove the private transfer variable and retain normal
helper discovery. This transfer must pass native sandbox acceptance before lite
media access is considered complete. The full edition's
helpers do not receive sandbox inheritance. After nested helper signing, the app
is re-signed and verified, and only then is the DMG created. These are ad-hoc
inspection signatures, not App Store or Developer ID signing.

The release includes the exact upstream source archives, this recipe, pins,
licenses and a build receipt in `DonutMediaCenter-macOS-media-sources.tar.xz`.
Generated resource mappings and unsigned staging payloads stay in an owned
temporary packaging directory and are removed on exit; they are not release
artifacts. Compiler and verified source caches are retained separately.
Ship it alongside the application and its complete corresponding application
source. The original component licenses and patent notices are included in the
application resources. Do not substitute a different binary under the receipt.

Still required for release acceptance: execute the exact sandboxed app on real
Mac hardware with owned media, test Powerbox selection and bookmark persistence
and recovery, and prove that child helpers can read authorized media and write
authorized output. Test denial and revoked/moved folders without falling back to
Homebrew or another installed FFmpeg. Synthetic audio decoding does not establish
audible playback; arm64 execution does not establish Intel runtime acceptance.
Store signing, provisioning and submission require separate platform work.

References: [Apple sandbox inheritance](https://developer.apple.com/library/archive/documentation/Miscellaneous/Reference/EntitlementKeyReference/Chapters/EnablingAppSandbox.html#//apple_ref/doc/uid/TP40011195-CH3-SW24),
[FFmpeg licensing](https://ffmpeg.org/legal.html). These requirements do not by
themselves establish an App Store review outcome.
