// Check if filename is a video
export const isVideo = (filename) => {
  if (!filename) return false
  const ext = filename.toLowerCase().split('.').pop()
  return ['webm', 'mp4', 'mov'].includes(ext)
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
