# DonutMediaCenter release preparation

Use `docs/agents/release-and-infrastructure.md` for platform build ownership,
shared host build locking, disposable acceptance and publication boundaries.
Version sources and lockfiles must pass `scripts/check-release-version.py`.
Prepare releases from a clean committed worktree, not the user's library.

## Version and release scope

Prepare the next release as **2.1.1**, with proposed tag **`v2.1.1`**.
The application metadata already uses this version; do not bump it again just
because Microsoft Store enrollment is pending. Keep published 2.1.0 and earlier
packages unchanged. [The 2.1.1 preparation sheet](2.1.1.md) records the candidate
scope and remaining gates; it is not a publication or support announcement.

Choose the distribution routes before freezing the release matrix. Direct
Linux/Windows downloads can proceed independently of Microsoft Store enrollment.
A deferred Store submission is a separate transaction with its own exact source,
identity, package, evidence and eventual Store version. Do not rewrite an already
published tag or replace its assets to add a later Store build.

The default remains one frozen source commit for the desktop matrix. Record the
actual source SHA and corresponding source archive for every artifact. If a
reviewed platform variant intentionally uses another commit, disclose it in the
release manifest and notes and provide its exact source. The shared core's
single-source manifest cannot silently represent a mixed-source set: rebuild
from its frozen source or extend and review the envelope first. An older receipt
does not qualify a rebuilt package, a new edition or another installation route.

## Distribution routes and promotion gates

| Route | Preparation path | Required promotion evidence |
| --- | --- | --- |
| Linux AppImage, DEB, RPM and portable ZIP | `scripts/build-linux-local.sh` | Exact package/source/notice checks plus runtime, installation or extraction, upgrade/recovery and media acceptance appropriate to each format. AppImage smoke results alone do not qualify DEB/RPM/ZIP. |
| Linux selected-folder Flatpak | `scripts/build-flatpak-local.sh ... selected-folder` from the verified AppImage | Actual installed native chooser, persistent access, revoke/reacquire, restart, helper/media and cleanup checks, with the permission profile and input AppImage SHA recorded. Keep experimental until its release/support gates pass. |
| Windows NSIS and portable ZIP | `scripts/build-windows-local.sh` | Exact source-built helper and source-offer binding, then native installer/portable, restart, migration, WebView2 and media/UI acceptance. Installed development MSIX results do not replace these route-specific checks. |
| Windows Microsoft Store MSIX | [MSIX procedure](../WINDOWS_MSIX.md), using the canonical Windows payload | DMC's own reserved Product identity values, Store-mode validation, installed production-identity behavior, dependencies/capabilities/listing review and official Store certification. Keep this route deferred while enrollment prevents reservation. |
| macOS universal / Store Lite | `scripts/build-macos-ci.sh` with the selected edition | Exact edition and artifact on physical Apple Silicon, permissions/bookmarks/helpers and real product acceptance; Intel hardware when claiming Intel support. Ad-hoc inspection packages and the existing stable exclusion are not shipping approval. Store signing, provisioning and submission are separate gates. |
| Android direct APK / Google Play AAB | `scripts/build-android-apk.sh --aab --keep-cache` | Permanent signing identity, exact source and package checks, native permission/offline/media/upgrade acceptance, then store-specific listing and policy checks for the AAB. |
| iOS, Flathub, AUR and other catalogues | Their reviewed platform/catalogue workflow | Explicit build, identity/signing where applicable, format validation, device/runtime evidence and destination support before adding a shipping row. A local Flatpak bundle is not a Flathub submission. |

Direct downloads, experimental packages and Store submissions have independent
promotion decisions. Record a deferred route and its missing gates explicitly;
do not make it a required row in an unrelated direct-download transaction. Once
a matrix is frozen, a failed required row stays visible and prevents sealing
that matrix. Changing the scope requires a newly reviewed manifest, not deletion
of the failing row. Do not widen the stable platform or architecture claims
merely because an inspection build succeeds.

Record native UI interaction, video motion, physical audio, GPU and sustained
use separately from backend tests and synthetic decode checks. Local WACK is an
optional diagnostic, not a substitute for official Store certification; see the
[MSIX procedure](../WINDOWS_MSIX.md) for its session and isolation requirements.

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

The stable profile also requires the exact Linux application and native source
archives, the audited Ubuntu distro and AVIF source companions, and the Windows
application, media and runtime source archives. Run the source-correspondence
audit and stage its verified supplemental archives under the declared names
before sealing; the Linux build wrapper alone does not produce those supplements.
Keep the matching receipts, source/package bindings and checksums with the release
evidence. Publish explicit links to each shipping platform's matching source
archives. Follow [licensing requirements](../LICENSING.md); do not equate a hash
or the shared core's `source-offer` bookkeeping with complete source verification.

The shared core currently supports stable/beta envelopes and the older desktop
formats/APK. Flatpak, macOS and Store formats need their own format, permission,
identity and destination gates wired into the envelope before inclusion. Do not
invent a channel or use a generic checksum-only row to claim that validation.
Update the shared core's product-profile schema, validator implementation and
tests together, then validate the product-owned profile and review the exact
matrix. Keep its centrally installed profile in sync through the distribution
owner; editing this repository's copy does not update that installation.

`scripts/build-release-matrix.sh` invokes the Linux and Windows platform
wrappers sequentially with the same frozen source commit. Native Mac builds
use `scripts/build-macos-ci.sh`; Cargo calls take the host gate on macOS too.
The matrix script is the direct desktop build path, not an all-platform build
or a Store publication command. Recheck disk space and active/queued jobs before
each admitted build, preserve warm caches and existing owners, and reuse verified
unchanged artifacts rather than starting duplicate pipelines.

Manifests remain PREPARED until every required artifact, checksum, format and
applicable signing gate passes. Shared-core SEALED records byte/format checks;
product acceptance, matching source obligations and explicit release approval
are still required. Preserve the complete matrix when a platform fails. Do not
replace failed rows with older builds or silently omit platforms.

Only after the selected matrix and release notes are approved, recheck that the
proposed tag is absent locally and on the destination remote. Freeze the exact
release commit and obtain explicit authorization for tag creation, push,
publication/upload, website promotion and announcements. Preparing notes,
profiles, manifests or local candidates does not perform or authorize them.

References:
- https://developer.android.com/guide/app-bundle
- https://developer.android.com/studio/publish/app-signing
- https://developer.android.com/guide/practices/page-sizes
