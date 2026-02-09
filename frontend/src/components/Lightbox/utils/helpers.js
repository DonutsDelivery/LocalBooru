// Check if filename is a video
export const isVideo = (filename) => {
  if (!filename) return false
  const ext = filename.toLowerCase().split('.').pop()
  return ['webm', 'mp4', 'mov', 'mkv'].includes(ext)
}

// Check if video format needs transcoding (not natively playable in browsers)
// Note: MKV removed — Tauri/WebKitGTK plays MKV natively via GStreamer,
// and users can select a quality preset to trigger transcoding if needed.
export const needsTranscode = (filename) => {
  return false
}

// Format time as MM:SS
// showPlaceholder: if true, shows "--:--" when time is 0 or invalid (useful for duration display while loading)
export const formatTime = (seconds, showPlaceholder = false) => {
  if (!seconds || !isFinite(seconds)) {
    return showPlaceholder ? '--:--' : '0:00'
  }
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}
