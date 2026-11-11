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

The script requires Flatpak, unsquashfs, and the x86-64 `org.freedesktop.Sdk`
and `org.freedesktop.Platform` runtimes at branch 25.08. It checks the input
hash before extracting it, preserves the bundled patched WebKit and media
runtime, and writes the build tree, a local OSTree repository, a `.flatpak`
bundle, and the input hash. Building and testing this bundle does not publish it.
It is not a substitute for a source-based, reproducible Flathub package; the
Flathub manifest and submission are maintained by Emil.

The candidate requests full host filesystem access because existing DMC libraries
can reside outside the home directory, along with network, display, audio and GPU
access. This broad permission allows access to the user's files; review and narrow
it only with verified portal or other persistent folder access. The build does not
establish sandbox compatibility, helper availability, GPU playback, SVP operation,
portal persistence, or media survival after cache eviction. Use the repository's
isolated desktop test launcher and owned synthetic media; never use the real
profile for acceptance. An actual installed Flatpak, clean restart, file selection,
helper behavior, LAN playback and filesystem recovery must be tested separately
before claiming support. Runtime-compiled Python addons and access to an external
SVP Manager have additional sandbox and runtime dependency constraints.
