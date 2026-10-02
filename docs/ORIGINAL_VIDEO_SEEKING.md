# Original video seeks

Original playback keeps its existing HTTP/Range source. Relative input is
accumulated from the latest requested position and merged within one display
frame. A later frame can submit a newer target while a native seek is underway;
there is no interpolation startup wait, buffer threshold, or seek cooldown.
Seeking preserves the HTML video element's playing/paused state, so it does not
call `play()` after each jump. Precise timeline input still uses `currentTime`.

The patched Linux WebKit runtime sends ordinary HTTP seek events through its
existing GStreamer asynchronous dispatch path. It uses the existing overlapping
seek slot to retain the newest target while one flush is active. A pending
worker prevents an early PAUSED notification from completing the seek before
FLUSH is sent. Rate changes wait for this dispatch to finish before applying
the requested rate. SVP filters (including nested `localbooruvs`), looping,
HLS playlist URLs, non-HTTP media, and rate-change events retain their previous dispatch policy.

A rejected current HTTP seek reports a media decode error. The lightbox stops
its loading overlay and offers **Close video**; reopen the item to retry. An old
worker cannot clear a replacement pipeline or a newer dispatched seek. Source
reloads create a new MediaPlayer through WebKit's normal HTML media load path.
The frontend also rejects removed-element/source errors and clears pending
input on navigation, source replacement, and unmount.

## Focused synthetic verification

From `frontend`, with the host build gate:

```sh
host-heavy-build run --project dmc-seek-tests --worktree "$PWD/.." --wait 21600 -- \
  env NODE_ENV=test node_modules/.bin/vitest run --config vitest.config.js \
  src/components/Lightbox/hooks/useVideoPlayback.component.test.jsx \
  src/components/Lightbox/localResolution.component.test.jsx \
  src/components/Lightbox/hooks/localResolutionRouting.component.test.jsx
```

`python3 scripts/test-webkit-http-seek.py --source /path/to/patched-source`
compiles the production seek functions, rate function, and seek completion
state branch against real GStreamer in a disposable directory. Compilation
uses the host gate. Synthetic seek callbacks delay before FLUSH and during
its upstream handling; assertions cover main-loop heartbeat, latest target,
no simultaneous seek sends, early PAUSED, rejection, no-op input, queued
supersession, rate ordering, and replacement ownership.

The canonical pristine-source patch and exact legacy upgrade patch must produce
identical C++ files. Use `scripts/test-webkit-mpv-isolation.py` and
`scripts/test-local-video-resize.py` to preserve the existing runtime contracts.

These checks prove the blocking mechanism and source behavior, not attribution
to a particular user media file. PAUSED state changes still follow WebKit's
existing main-thread policy; the synthetic fixture measures that stage too.
A matching app/runtime build and real playback acceptance remain deployment
checks. A paused phone original stream has no shared playback producer or
transport controls over desktop playback; active transfer can still use LAN
and storage bandwidth.
