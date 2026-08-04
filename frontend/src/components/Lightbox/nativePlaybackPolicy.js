export function selectPlaybackTarget({
  desktopTauri,
  nativeRendererAvailable,
  safeMode,
  desktopPlayerMode,
  isVideo,
  isCasting,
  castProtocol,
  mobileWebView,
  localFileAvailable,
  directRemote = false,
}) {
  if (!isVideo) return 'react-image'
  if (isCasting) return castProtocol === 'dlna' ? 'dlna-receiver' : 'cast-receiver'
  if (mobileWebView) return 'mobile-webview'
  if (directRemote) return 'direct-remote'
  const nativeRequested = desktopPlayerMode === 'native' || desktopPlayerMode === 'native_svp'
  if (nativeRequested && desktopTauri && localFileAvailable && nativeRendererAvailable && !safeMode) {
    return 'native-local'
  }
  return 'web-hls'
}

export function nativeViewportForLightbox(bounds, visible) {
  return {
    x: Math.round(bounds.left ?? bounds.x),
    y: Math.round(bounds.top ?? bounds.y),
    width: Math.max(1, Math.round(bounds.width)),
    height: Math.max(1, Math.round(bounds.height)),
    visible: Boolean(visible),
  }
}
