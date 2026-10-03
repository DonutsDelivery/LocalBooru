# DonutMediaCenter release preparation

Use `docs/agents/release-and-infrastructure.md` for platform build ownership,
shared host build locking, disposable acceptance and publication boundaries.
Version sources and lockfiles must pass `scripts/check-release-version.py`.
Prepare releases from a clean committed worktree, not the user's library.

## Product identity

`assets/logo-donut.svg` is the canonical DMC platform icon source. Generate with
`cargo tauri icon assets/logo-donut.svg --ios-color '#15191D'`. The command updates
Tauri desktop icons, native Android resources and the iOS icon set. Mirror the
Android launcher resources into `frontend/android` and the primary PNG/ICO into
`assets` and `frontend/public` when retaining legacy builds. Check adaptive and
round icons as well as PNG, ICO and ICNS payloads. Flatten regenerated iOS
PNGs to RGB over `#15191D` (for example with Pillow), and run
`scripts/apply-ios-icons.py` after `cargo tauri ios init`. The iOS inspection
workflow copies those committed icons into the newly generated asset catalog
and rejects missing, transparent or incorrectly sized images.

## Android

Build the permanently release-signed APK and Google Play app bundle together:

```bash
ANDROID_BUNDLETOOL=/path/to/verified/bundletool-all.jar \
  ./scripts/build-android-apk.sh --aab --keep-cache
```

The helper uses the existing operator-owned signing configuration under
`$XDG_CONFIG_HOME/donutsdelivery/release-secrets/android/localbooru`, or explicit
`ANDROID_KEYSTORE`, `ANDROID_KEY_ALIAS`, `ANDROID_STORE_PASS`, `ANDROID_KEY_PASS`.
Never commit or print its contents. The public certificate is pinned in
`release/android.json`. Debug certificates fail preparation. Passwords are
passed through environment references, not command arguments.

The build participates in the host-wide gate. It preserves caches, verifies
source provenance, package ID, version, target SDK, release/debug status, ABIs,
certificate, APK ZIP alignment and 64-bit ELF LOAD/RELRO alignment. AAB checks
also validate its signature and Google bundletool structure/16KB configuration.
`--sign-only` requires a matching build provenance marker; stale binaries fail.

Outputs: `DonutMediaCenter.apk`, `DonutMediaCenter.aab`,
`DonutMediaCenter-Android.json`, `SHA256SUMS-Android`. These are ignored generated
artifacts. APK supports direct installation; AAB is the Google Play upload.
Store acceptance is separate from signing and package checks. Complete local
library, paired remote library, permission, offline and upgrade tests on the
exact signed APK, including a 16KB device/emulator when claiming that support.

## Shared release envelope

`release/profiles/localbooru.json` freezes DMC filenames while retaining the
established repository/product ID, destinations and support exclusions. Load
this product-owned profile with the DonutsDelivery shared release core rather
than using its obsolete LocalBooru filenames. Prepare separate desktop stable
and Android beta manifests. The store AAB and its evidence accompany the Android
preparation; the current shared core seals APK rows only.

`scripts/build-release-matrix.sh` invokes the Linux and Windows platform
wrappers sequentially with the same frozen source commit. Native Mac builds
use `scripts/build-macos-ci.sh`; Cargo calls take the host gate on macOS too.

Manifests remain PREPARED until every required artifact, checksum, format and
signing gate passes. Preserve the complete matrix when a platform fails. Do not
replace failed rows with older builds or silently omit platforms. Publication,
tags, website promotion and announcements are subsequent operator actions.

References:
- https://developer.android.com/guide/app-bundle
- https://developer.android.com/studio/publish/app-signing
- https://developer.android.com/guide/practices/page-sizes
