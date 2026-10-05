# DonutMediaCenter licensing

## Application license

Following the 2.1.0 release, the application is distributed under **GNU GPL
version 3 only**, SPDX `GPL-3.0-only`. The full text is in [LICENSE](../LICENSE);
[NOTICE](../NOTICE) applies that license and preserves copyright attribution.
First-party source without a separate license follows this grant.

You may sell copies, including an official paid APK. Distribution of covered
modified or unmodified binaries requires corresponding source under GPLv3.
Recipients may modify and redistribute the covered work. GPL does not cap a
reseller's price. Private modifications do not require public publication.
GPLv3 alone does not impose AGPL's network-service source requirement.

## Historical and third-party permissions

The last main revision before this transition is
`a433698a220daec60e25e3f29e8c4675b46200c2`. Published 2.1.0 artifacts and earlier
MIT versions retain their permissions. The original MIT notice is preserved
verbatim in [LICENSES/MIT-legacy.txt](../LICENSES/MIT-legacy.txt). Retain it for
previously MIT-licensed portions; do not describe those old versions as having
been retroactively restricted by this change.

Third-party files, libraries and separately licensed components retain their
own licenses. Preserve their notices, attribution and source obligations:

- `vendor/wayland-scanner`: its own MIT copyright/license file.
- Generated Gradle wrapper files: upstream Apache-2.0 notices.
- `addons/lada`: its own AGPL-3.0-only license and third-party/model notices.
- Native runtime libraries, including WebKitGTK, GStreamer, FFmpeg, MPV and
  VapourSynth: the licenses of the exact components/configurations shipped.
- ComfyUI, custom nodes, model weights and other downloaded add-on dependencies:
  their respective upstream terms. Do not claim GPL relicensing of them.
- SVP Manager/SVPflow: user-supplied components; no redistribution permission is
  created by this change.

The software license does not change rights in library media or generated
outputs. DMC's own name and logo are addressed in [TRADEMARKS.md](../TRADEMARKS.md),
which preserves truthful reference and permitted independent redistribution.

## Contributions

New first-party contributions are accepted under GPL-3.0-only unless explicitly
agreed and documented otherwise. Contributors retain copyright. There is no
copyright assignment or blanket relicensing agreement in this policy. Preserve
upstream permissions and attribution when proposing third-party code; disclose
its source and license during review. See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Release requirements

For the next release, bump the version rather than replacing published 2.1.0
packages or changing the old release's license declaration.

1. Ship `LICENSE`, `NOTICE`, `LICENSES/MIT-legacy.txt` and `TRADEMARKS.md` in
   desktop packages and portable archives. The Tauri resource mapping covers
   installed bundles; the Windows portable wrappers copy these explicitly;
   Linux packages also carry them in `usr/share/doc/localbooru`.
2. Settings > General exposes the full application license, historical notice
   and branding policy in the frontend, including the APK and web UI. These are
   imported from the authoritative root files rather than maintained as copies.
3. Publish the complete corresponding source of **each exact shipping revision**
   alongside its binaries at no additional charge, with an explicit source link
   in the release/download listing. Include lockfiles, build scripts, patches,
   generated platform source required to build, and required installation
   information. If Android or another platform uses a different source commit,
   identify and provide that exact revision separately. Do not substitute the
   moving `main` branch or a desktop-only tag for Android's source.
4. Preserve notices and matching source for all bundled third-party components.
   The existing native runtime source archive alone does not supply the entire
   application's corresponding source. See [native licensing](NATIVE_VIDEO_LICENSES.md)
   and [Linux runtime notices](../release/linux/THIRD_PARTY_NOTICES.md).
5. Verify package metadata says `GPL-3.0-only`. The legacy AUR recipe remains
   pinned to MIT-licensed 2.0.0; update its version, URLs, hashes and license
   together when moving it to a future GPL release.

## Transition review and limits

The main history records contributions as DonutsDelivery and User. Existing
contributions were distributed under the root MIT grant; that grant permits
incorporation into a GPL distribution when its notices are retained. This
transition does not claim exclusive ownership of every contribution.

A read-only review of the current lockfile metadata found permissive licenses
for 44 production npm packages. Locally cached Cargo sources covered 647
registry packages, with permissive or MPL-2.0 declarations; 137 registry
packages were not cached. This is a preliminary compatibility review, not a
complete per-platform binary dependency audit. Review unresolved components and
the exact bundled native configurations before the next release. No dependencies
or already published artifacts were changed by this license transition.

References: [GPLv3](https://opensource.org/license/gpl-3.0),
[MIT](https://opensource.org/license/mit), and
[OSI trademark guidance](https://opensource.org/faq).
