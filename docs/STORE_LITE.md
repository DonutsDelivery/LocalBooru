# Lite capability profile

The `store-lite` Cargo feature selects immutable native policy. Ordinary builds
keep the full feature set. This profile is an implementation prerequisite for
the Mac store edition, not proof of App Sandbox support or store readiness.

Lite blocks the community catalog (`/api/online`), its deep-link enrollment and
authorization UI, peer/offline supervisor startup, downloaded executable add-on
installation/repair/start/probing, Create's executable controller, and alternate
application updates. Requests through the paired-server proxy receive the same
policy. Persisted community and add-on settings are preserved without activating
those services. Compiled-in tools, including curation, remain available.

Images, videos, music, local collections, metadata sidecars, device pairing,
personal LAN libraries, and media sharing remain available. Capability discovery
uses native startup metadata or the serving app's `/api/edition` endpoint; it
never inherits a paired full server's capabilities. The backend enforces policy
independently of the UI.

The existing native Apple build entrypoint has an inspection variant:

```sh
LOCALBOORU_MACOS_EDITION=store-lite scripts/build-macos-ci.sh
```

It selects the feature and separate `com.localbooru.app.lite` identity, target
directory, application name, and artifacts. The normal full build is unchanged.
The output is still ad-hoc signed for inspection. Linux unit tests do not
establish native Apple behavior.

The macOS lite config now selects App Sandbox entitlements and its native folder
picker persists security-scoped bookmarks with startup restoration and recovery.
See [folder access](MACOS_FOLDER_ACCESS.md) for the implementation and remaining
native acceptance. Native compilation is verified; real external-folder
selection, bookmark recovery and helper authorization still need acceptance.

On 14 November 2026, the exact `8178386b` universal inspection package passed
14 backend checks on physical Apple Silicon, including saved uploads, media
helpers, transcodes, restart, thumbnail eviction and folder relocation inside
an owned disposable container. A separate Linux client verified real LAN login,
authenticated image/video and range transfer, audio/video decoding, read-only
restrictions and persistence across restart. LAN access was unavailable before
enabling it and after disabling it again. All owned fixtures were cleaned up.
These are backend/API checks, not visible playback, Powerbox, speaker, GPU,
Intel execution, production signing or Store approval. The tested package's
corresponding source remains the exact `8178386b` revision.

Before distribution through the Mac App Store, verify native bookmark/permission
behavior and App Sandbox entitlements, bundled media helpers and codec behavior, account/provisioning/store signing,
license/source obligations, migration and eviction behavior, and acceptance on
real supported Macs. This config intentionally does not assert those unfinished
properties or submit anything to a store.
