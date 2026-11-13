# Optional Windows MSIX packaging

MSIX uses the same x64 executable, FFmpeg/FFprobe, libraries and notices as the
canonical NSIS/portable build. It is a packaged Win32 **full-trust** application,
with `runFullTrust` and the existing `donutmediacenter` protocol. It does not use
AppContainer or request `broadFileSystemAccess`. This preserves the desktop media
server and helper model; it is not equivalent to Flatpak restrictions.

Run the canonical Windows build first, from a clean committed source revision:

```sh
./scripts/build-windows-local.sh
```

The wrapper verifies `Windows-build-receipt.json`, binding the source SHA, exact
container recipe hash, x64 target and final ZIP/installer digests. This record is
unsigned build provenance, not a publisher signature or tamper-proof attestation.
Older artifacts without the receipt must be rebuilt through the canonical path;
do not invent a receipt for an unknown executable.

Stage from that exact commit and its receipt. Supply the identity and publisher
from Partner Center or the intended sideload certificate, never a guessed
production identity. Use the reserved publisher's display name. Store submission
versions must follow Microsoft's requirements, including a zero fourth component.

```sh
python3 scripts/stage-windows-msix.py \
  --zip /absolute/release/DonutMediaCenter-Windows.zip \
  --receipt /absolute/release/Windows-build-receipt.json \
  --source FULL_40_CHARACTER_SOURCE_SHA \
  --output /absolute/new-msix-stage \
  --identity RESERVED_PACKAGE_ID \
  --publisher 'CN=EXACT_CERTIFICATE_OR_PARTNER_CENTER_PUBLISHER' \
  --publisher-display-name 'Reserved publisher display name' \
  --version 2.1.1.0
```

Staging rejects source/recipe/hash mismatches, unsafe ZIP paths and Windows path
aliases, links, duplicates, oversized expansions, unexpected payload, missing
helpers/notices and non-x64 executables. Notices and icons come from the specified
commit. It does not execute any staged executable or call a ZIP file an MSIX.

For an actual package, add `--package /absolute/new-package.msix` and
`--makeappx /absolute/path/to/MakeAppx-wrapper`. This must invoke the real Windows
SDK tool, using the existing release toolchain where available. A Linux/Wine
wrapper must convert file paths correctly. The Windows test laptop is a runtime
target, not a replacement build host. The script keeps SDK semantic validation
enabled, unpacks the result with the SDK, and checks staged bytes round-trip.
Missing tools or SDK failures are errors; no alternate ZIP fallback exists.

Optional signing requires all three explicit parameters: `--signtool`,
`--certificate-thumbprint` and `--timestamp-url` (HTTPS RFC3161). The thumbprint
selects a certificate already available to the signing environment. The script
uses SHA256, timestamps during signing, then verifies with SignTool's default
authentication policy. It does not import certificates, accept a password on the
command line, install packages, submit to a store or publish. The final sidecar
receipt records the package digest and whether signature verification ran.
Unsigned SDK packages are inspection inputs, not installable release claims.

The command interfaces follow Microsoft's [desktop manifest guidance](https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-manual-conversion),
[MakeAppx validation](https://learn.microsoft.com/en-us/windows/msix/package/create-app-package-with-makeappx-tool),
and [package signing contract](https://learn.microsoft.com/en-us/windows/win32/appxpkg/how-to-sign-a-package-using-signtool).
Passing MakeAppx validation does not establish installability or Store acceptance.

Before release, verify the exact package on an owned Windows fixture account:

- Package identity, publisher/certificate match, signature and Store capability
  review; WebView2 and VC runtime dependencies must be assessed on supported OSes.
- Isolated onboarding/settings across restart, Unicode folder selection,
  import/playback/audio/transcode and bundled helper discovery.
- Loopback/LAN listeners, pairing, firewall prompts, protocol activation and
  single-instance behavior; do not replace an existing user's listeners.
- Per-user paths and filesystem virtualization, package update/uninstall,
  explicit supported migration, saved media and references after cache eviction,
  unavailable/revoked folders and recovery without destructive cleanup.

Use the privacy and Windows acceptance procedure in
[release infrastructure](agents/release-and-infrastructure.md). No real SDK MSIX,
signature or Windows package acceptance has yet been produced by this change.
