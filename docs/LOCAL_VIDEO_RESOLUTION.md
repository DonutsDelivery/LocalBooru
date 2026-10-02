# Local decoded video resolution

Linux desktop playback from DMC's selected embedded backend keeps the original
media URL. A lower resolution composes `videoscale → capsfilter → optional SVP`
inside the existing WebKit GStreamer video filter. Audio and the HTML controls
remain on the same player. No encoder or HLS producer is started for this change.
The compressed source still needs decoding at its original resolution.

Bounds belong to a physical video ID, document epoch and resize revision. The
frontend waits for the configuration acknowledgement before attaching the source.
Resolution changes withdraw the previous Manager graph and remount using the
existing playback handoff, retaining time, pause intent, volume and mute. A failed
withdrawal keeps the current player usable and reports the failure.

Negotiated source caps determine the output dimensions. They fit the requested
rectangle without upscaling; resized dimensions are rounded down to even values.
FPS and pixel aspect are retained. Original inserts no scaler. The scaler writes
its measured source and output geometry under the same private video-host lease.
Manager activation for a non-Original contract requires this matching geometry;
missing geometry gets bounded readiness retries and a visible error. Original
source metadata remains separate for the resolution menu.

SVP Off also verifies the current scaler's geometry after media metadata loads.
Stock or older WebKit that cannot produce this proof gets a visible error and an
owned handoff back to Original; a file-write acknowledgement alone does not count
as successful resizing.

Android and remote server playback retain their existing quality routes. Original
can serve the original network file, while lower streamed qualities and remote
SVP use encoded streams. The local gate checks backend selection, so a remote
server reached through an authenticated loopback proxy is still remote.

## Runtime source and focused checks

The canonical WebKit patch is `patches/webkitgtk/2.52.3-playbin-video-filter.patch`.
The recognized legacy upgrade updates exactly the existing two C++ files through
`scripts/prepare-webkit-mpv-isolation.py`; source hashes reject unknown or mixed
states. Runtime installation and application restart are separate operational
steps. Applying product source alone does not update a running WebProcess.

Focused tests use synthetic data only:

- `node --test frontend/src/utils/localVideoResolution.test.js frontend/src/utils/svpVideoHost.test.js`
- The `localResolution.component.test.jsx` Lightbox fixture covers seek, pause,
  audio, rapid choices, navigation, withdrawal failure, Original acknowledgement,
  and SVP toggle/graph handoff.
- The `localResolutionRouting.component.test.jsx` hook fixture covers local
  resize with SVP off/on and retained Android/remote transport behavior.
- `scripts/run-cargo.sh test --manifest-path src-tauri/Cargo.toml --lib svp_`
- `python3 scripts/test-local-video-resize.py --source <recognized-source-root>`
  negotiates actual GStreamer caps with raw `videotestsrc` frames. The optional
  opaque graph is represented by identity so no installed Manager is required.
- `python3 scripts/test-webkit-mpv-isolation.py --source <recognized-source-root>`
  retains registration, relay ownership and socket coexistence checks.

Compiler fixtures use the host heavy-build gate. No profile, real media, user
settings, running WebKit instance or real Manager socket is used by these tests.
