#!/usr/bin/env python3
"""
Standalone WebKitGTK video playback test with frame drop monitoring.
Uses the same env vars as LocalBooru's lib.rs to replicate the pipeline.

Usage:
    python3 test-webkitgtk-video.py /path/to/video.mp4
    GST_DEBUG=*videodecoder*:5 python3 test-webkitgtk-video.py /path/to/video.mp4
"""

import sys
import os
import signal
import threading
import time
import subprocess

# ── Set the same env vars as lib.rs BEFORE importing GI ──

# 1. VA-API decoders preferred
if "GST_PLUGIN_FEATURE_RANK" not in os.environ:
    os.environ["GST_PLUGIN_FEATURE_RANK"] = (
        "vah264dec:MAX,vah265dec:MAX,vaav1dec:MAX,vavp9dec:MAX,"
        "nvh264dec:PRIMARY+1,nvh265dec:PRIMARY+1,nvav1dec:PRIMARY+1,nvvp9dec:PRIMARY+1,"
        "nvh264sldec:PRIMARY,nvh265sldec:PRIMARY,"
        "avdec_h264:NONE,avdec_h265:NONE"
    )

# 2. nvidia-vaapi-driver
if "GST_VA_ALL_DRIVERS" not in os.environ:
    os.environ["GST_VA_ALL_DRIVERS"] = "1"
if "LIBVA_DRIVER_NAME" not in os.environ:
    os.environ["LIBVA_DRIVER_NAME"] = "nvidia"

# 3. DMA-BUF + native Wayland
os.environ.pop("WEBKIT_DISABLE_DMABUF_RENDERER", None)
os.environ.pop("WEBKIT_DISABLE_COMPOSITING_MODE", None)
os.environ["GDK_BACKEND"] = "wayland"
if "__NV_DISABLE_EXPLICIT_SYNC" not in os.environ:
    os.environ["__NV_DISABLE_EXPLICIT_SYNC"] = "1"

# 4. EGL for DMA-BUF texture import
if "GST_GL_PLATFORM" not in os.environ:
    os.environ["GST_GL_PLATFORM"] = "egl"

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.1')
from gi.repository import Gtk, WebKit2, GLib


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; height: 100vh; }
  video { max-width: 100%%; max-height: 100%%; object-fit: contain; }
  #stats {
    position: fixed; top: 10px; left: 10px;
    background: rgba(0,0,0,0.8); color: #0f0; font: 14px monospace;
    padding: 10px; border-radius: 4px; z-index: 9999;
    white-space: pre; pointer-events: none;
  }
</style>
</head>
<body>
<div id="stats">Loading...</div>
<video id="v" autoplay muted></video>
<script>
const v = document.getElementById('v');
const stats = document.getElementById('stats');

v.src = '%s';

let lastPresentedFrames = 0;
let lastDroppedFrames = 0;
let lastStatsTime = performance.now();
let fpsHistory = [];
let totalJanks = 0;
let jankFrames = [];

function updateStats() {
  const q = v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality() : null;
  const now = performance.now();
  const dt = (now - lastStatsTime) / 1000;

  let lines = [];
  lines.push('Time: ' + v.currentTime.toFixed(1) + 's / ' + (v.duration ? v.duration.toFixed(1) : '?') + 's');
  lines.push('Resolution: ' + v.videoWidth + 'x' + v.videoHeight);

  if (q) {
    const newPresented = q.totalVideoFrames - lastPresentedFrames;
    const newDropped = q.droppedVideoFrames - lastDroppedFrames;
    const fps = dt > 0 ? (newPresented / dt).toFixed(1) : '?';

    fpsHistory.push(parseFloat(fps));
    if (fpsHistory.length > 60) fpsHistory.shift();
    const avgFps = (fpsHistory.reduce((a,b) => a+b, 0) / fpsHistory.length).toFixed(1);

    lines.push('FPS: ' + fps + ' (avg: ' + avgFps + ')');
    lines.push('Presented: ' + q.totalVideoFrames);
    lines.push('Dropped: ' + q.droppedVideoFrames + ' (+' + newDropped + ')');
    lines.push('Drop rate: ' + (q.totalVideoFrames > 0 ? (100 * q.droppedVideoFrames / q.totalVideoFrames).toFixed(2) : 0) + '%%');
    lines.push('Janks detected: ' + totalJanks);

    if (newDropped > 0) {
      lines.push('!! DROPS THIS SECOND: ' + newDropped);
    }

    // Relay stats to Python via document.title
    document.title = 'STATS|fps=' + fps + '|avg=' + avgFps + '|presented=' + q.totalVideoFrames + '|dropped=' + q.droppedVideoFrames + '|newdrop=' + newDropped + '|janks=' + totalJanks + '|time=' + v.currentTime.toFixed(1);

    lastPresentedFrames = q.totalVideoFrames;
    lastDroppedFrames = q.droppedVideoFrames;
  } else {
    lines.push('(getVideoPlaybackQuality not available)');
    document.title = 'STATS|no_quality_api|time=' + v.currentTime.toFixed(1);
  }

  if (v.buffered.length > 0) {
    const buffEnd = v.buffered.end(v.buffered.length - 1);
    const ahead = buffEnd - v.currentTime;
    lines.push('Buffer ahead: ' + ahead.toFixed(1) + 's');
  }

  lines.push('readyState: ' + v.readyState);

  stats.textContent = lines.join('\\n');
  lastStatsTime = now;
}

setInterval(updateStats, 1000);

// Log events via title
['play','pause','waiting','stalled','error','canplay','canplaythrough'].forEach(function(evt) {
  v.addEventListener(evt, function() {
    document.title = 'EVENT|' + evt + '|time=' + v.currentTime.toFixed(2);
  });
});

// requestVideoFrameCallback for jank detection
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
  var lastFrameTime = 0;
  function onFrame(now, metadata) {
    if (lastFrameTime > 0) {
      var delta = now - lastFrameTime;
      var expected = 1000 / 60;
      if (delta > expected * 2.5) {
        totalJanks++;
        jankFrames.push({delta: delta, time: v.currentTime});
      }
    }
    lastFrameTime = now;
    v.requestVideoFrameCallback(onFrame);
  }
  v.requestVideoFrameCallback(onFrame);
  document.title = 'EVENT|rvfc_enabled';
} else {
  document.title = 'EVENT|rvfc_not_available';
}
</script>
</body>
</html>"""


class VideoTestWindow(Gtk.Window):
    def __init__(self, video_path):
        super().__init__(title="WebKitGTK Video Test")
        self.set_default_size(1920, 1080)
        self.video_path = os.path.abspath(video_path)

        ctx = WebKit2.WebContext.get_default()
        self.webview = WebKit2.WebView.new_with_context(ctx)

        settings = self.webview.get_settings()
        settings.set_hardware_acceleration_policy(
            WebKit2.HardwareAccelerationPolicy.ALWAYS
        )
        settings.set_enable_mediasource(True)
        settings.set_enable_media_stream(True)
        settings.set_enable_webaudio(True)
        settings.set_allow_file_access_from_file_urls(True)
        settings.set_allow_universal_access_from_file_urls(True)
        settings.set_enable_developer_extras(True)
        self.webview.set_settings(settings)

        # Capture JS stats via document.title changes
        self.webview.connect('notify::title', self._on_title_change)

        self.add(self.webview)

        video_url = f"file://{self.video_path}"
        html = HTML_TEMPLATE % video_url
        self.webview.load_html(html, f"file://{os.path.dirname(self.video_path)}/")

        print(f"[TEST] Video: {self.video_path}")
        print(f"[TEST] HW Accel: ALWAYS")
        print(f"[TEST] GDK_BACKEND={os.environ.get('GDK_BACKEND', 'unset')}")
        print(f"[TEST] GST_VA_ALL_DRIVERS={os.environ.get('GST_VA_ALL_DRIVERS', 'unset')}")
        print(f"[TEST] LIBVA_DRIVER_NAME={os.environ.get('LIBVA_DRIVER_NAME', 'unset')}")
        print(f"[TEST] GST_GL_PLATFORM={os.environ.get('GST_GL_PLATFORM', 'unset')}")
        print(f"[TEST] __NV_DISABLE_EXPLICIT_SYNC={os.environ.get('__NV_DISABLE_EXPLICIT_SYNC', 'unset')}")
        print(f"[TEST] WEBKIT_DISABLE_DMABUF_RENDERER={os.environ.get('WEBKIT_DISABLE_DMABUF_RENDERER', 'unset')}")
        print()
        sys.stdout.flush()

        self._monitor = True
        self._monitor_thread = threading.Thread(target=self._gpu_monitor, daemon=True)
        self._monitor_thread.start()

    def _on_title_change(self, webview, param):
        title = webview.get_title()
        if title and (title.startswith("STATS|") or title.startswith("EVENT|")):
            print(f"[JS] {title}")
            sys.stdout.flush()

    def _gpu_monitor(self):
        time.sleep(3)
        while self._monitor:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.decoder,utilization.encoder,memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 4:
                        gpu, dec, enc, mem = [p.strip() for p in parts]
                        print(f"[GPU] sm:{gpu}% dec:{dec}% enc:{enc}% mem:{mem}MB")
                        sys.stdout.flush()
            except Exception:
                pass
            time.sleep(3)

    def cleanup(self):
        self._monitor = False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        sys.exit(1)

    print("=" * 60)
    print("WebKitGTK Video Playback Test")
    print("=" * 60)

    win = VideoTestWindow(video_path)
    win.connect("destroy", lambda w: Gtk.main_quit())
    win.show_all()

    signal.signal(signal.SIGINT, lambda *a: Gtk.main_quit())
    GLib.timeout_add(500, lambda: None)

    try:
        Gtk.main()
    finally:
        win.cleanup()
        print("\n[TEST] Done.")


if __name__ == "__main__":
    main()
