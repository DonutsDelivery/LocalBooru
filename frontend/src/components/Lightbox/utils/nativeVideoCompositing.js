const NATIVE_VIDEO_COMPOSITING_CLASS = 'native-video-compositing'

export function setNativeVideoCompositing(documentElement, active) {
  if (!documentElement?.classList) return
  documentElement.classList.toggle(NATIVE_VIDEO_COMPOSITING_CLASS, Boolean(active))
}

export function clearNativeVideoCompositing(documentElement) {
  documentElement?.classList?.remove(NATIVE_VIDEO_COMPOSITING_CLASS)
}
