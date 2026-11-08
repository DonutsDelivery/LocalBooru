import { classifyTapZone } from '../hooks/useVideoGestures.js'

// Mirrors handleNavClick: images navigate on the outer 40% of each side, and a
// zoomed desktop image never navigates, so a double-click can reset the zoom
// anywhere. Videos seek on the outer thirds and toggle playback in the middle.
export function isDoubleClickZone({ ratio, width, isVideo, zoomed, curation }) {
  if (curation) return false
  if (isVideo) return classifyTapZone(ratio) === 'toggle'
  if (zoomed && width > 768) return true
  return ratio >= 0.4 && ratio <= 0.6
}
