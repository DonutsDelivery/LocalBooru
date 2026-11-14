# Local Flatpak candidate

`scripts/build-flatpak-local.sh` creates an **experimental local candidate** from
an existing Linux x86-64 DMC AppImage and an independently verified SHA-256.
It does not build from source or create a Flathub submission. Use the AppImage
from the exact reviewed release candidate, and keep its corresponding native
runtime source archive and licensing notices available. An older AppImage does
not include later sandbox fixes.

The output directory must be an absolute path that does not already exist:

```sh
scripts/build-flatpak-local.sh /absolute/path/DonutMediaCenter-Linux.AppImage SHA256 /absolute/new/output-directory
```

To opt into the experimental selected-folder profile instead:

```sh
scripts/build-flatpak-local.sh /absolute/path/DonutMediaCenter-Linux.AppImage SHA256 /absolute/new/output-directory selected-folder
```

The optional fourth argument is `host` or `selected-folder`; omission selects
`host` for compatibility. The output's `permission-profile.txt` records the
choice. Both profiles retain network, IPC, display, audio, GPU and the D-Bus
name below. Only `host` grants `--filesystem=host`; selected-folder adds no
filesystem mount and requests GTK's native chooser portal via `GTK_USE_PORTAL=1`.
Flatpak's app-owned XDG data remains available. Contained Linux uses the app
 data directory rather than the legacy `~/.localbooru` directory; existing host
libraries are not automatically migrated or granted access. If a stored library
path is inaccessible, use the folder chooser to select it again and check the
result before relying on it; selection may return a different portal path, so
reconnecting or reconfiguring the library may be necessary. Do not delete the
original library as a recovery step.

These packaging checks cannot establish persistent portal grants. Acceptance on
an actual installed package must exercise the real chooser, restart, revoked
access and reacquisition, and library/helper behavior with synthetic data.

The script requires Flatpak, unsquashfs, and the x86-64 `org.freedesktop.Sdk`
and `org.freedesktop.Platform` runtimes at branch 25.08. It checks the input
hash before extracting it, preserves the bundled patched WebKit and media
runtime, and writes the build tree, a local OSTree repository, a `.flatpak`
bundle, and the input hash. Building and testing this bundle does not publish it.
The launcher preserves the runtime's automatically inherited `codecs-extra`
library and GStreamer paths ahead of its base codec libraries. Freedesktop 25.08
provides H.264 decoding in that extension; selecting the base libraries first
breaks FFmpeg video thumbnails and transcoding even when WebView playback works.
It is not a substitute for a source-based, reproducible Flathub package; the
Flathub manifest and submission are maintained by Emil.

The legacy host profile requests full host filesystem access because existing DMC libraries
can reside outside the home directory, along with network, display, audio and GPU
access. It also permits DMC to own its existing `com.localbooru.app` D-Bus
name. This broad filesystem permission allows access to the user's files; review and narrow
it only with verified portal or other persistent folder access. The build does not
establish sandbox compatibility, helper availability, GPU playback, SVP operation,
portal persistence, or media survival after cache eviction. Use the repository's
isolated desktop test launcher and owned synthetic media; never use the real
profile for acceptance. An actual installed Flatpak, clean restart, file selection,
helper behavior, LAN playback and filesystem recovery must be tested separately
before claiming support. Runtime-compiled Python addons and access to an external
SVP Manager have additional sandbox and runtime dependency constraints.

The patched WebKit runs its video process in a nested Flatpak sandbox. DMC
creates a private per-process bridge directory inside the container's XDG data
directory and exposes only that directory to the nested process. Resize requests
and decoded-frame geometry must be readable/writable on both sides. This is
transient bridge state, removed when its last application owner releases it;
it is not library media or a persistent download. Native non-Flatpak storage
keeps its existing layout. Do not disable WebKit's sandbox to repair playback.

The AppImage conversion must retain a consistent WebKit process set. The
linuxdeploy SONAMEs point at DMC's patched WebKit and JavaScriptCore. The native
runtime includes matching Web, Network, and (when enabled) GPU process helpers;
launchers set their private helper directory. A missing bundled helper fails
within that directory rather than falling back to an unrelated host version.
