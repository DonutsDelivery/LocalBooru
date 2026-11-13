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
The output is still ad-hoc signed for inspection. Native Apple compilation has
not been verified by Linux unit tests.

Before distribution through the Mac App Store, implement and test persistent
security-scoped bookmarks and permission recovery, App Sandbox entitlements,
bundled media helpers and codec behavior, account/provisioning/store signing,
license/source obligations, migration and eviction behavior, and acceptance on
real supported Macs. This config intentionally does not assert those unfinished
properties or submit anything to a store.
