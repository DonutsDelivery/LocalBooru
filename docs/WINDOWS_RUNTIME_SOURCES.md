# Windows helper runtime sources

The canonical Windows wrapper exports
`DonutMediaCenter-Windows-runtime-sources.tar.xz` and
`runtime-source-receipt.json` beside the binary, exact application source and
codec source archives. Keep all of them together in the download listing.
The runtime companion contains the pinned LLVM project, MinGW-w64 and
llvm-mingw packaging sources plus the export recipe and component notices.

Small component notices, the receipt and an exact companion filename/hash
pointer are included in portable, NSIS and derived MSIX payloads. The full
runtime sources are a separate download; they are not duplicated inside every
installer. Verification rejects a missing or changed companion or packaged notice.

`scripts/windows-runtime-sources.json` associates the checksum-pinned
llvm-mingw 20261104 toolchain with immutable source revisions. LLVM supplies
libcxx, libcxxabi, libunwind and compiler-rt; MinGW supplies startup/support CRT
and headers. The LLVM revision matches the compiler identification. The MinGW
revision comes from the published release recipe's default for that successful
push build; scheduled/nightly revision overrides are a different build path.
This association is not an independent vendor rebuild or signature attestation.
Windows UCRT DLL imports remain platform prerequisites, not bundled UCRT code.

The runtime exporter checks the media receipt's toolchain/compiler identities
before downloading or creating output. It verifies each source archive hash,
reads only named regular license files, sanitizes companion archive metadata,
and verifies sources, recipes, notices and receipts again before staging.
The existing nine-file media build recipe remains unchanged, so a metadata-only
runtime source export can reuse already verified helper binaries.

Synthetic checks use an owned directory outside the repository:

```sh
DMC_MEDIA_FIXTURE_ROOT=/absolute/owned-fixtures python3 scripts/test-windows-runtime-sources.py
```

Source/notices verification does not replace real installed-package, GUI, audio,
GPU, Store activation, migration, certification or production-identity checks.
