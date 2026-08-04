export const initialNativeVideoView = Object.freeze({
  generation: 0,
  presentation: 'react_image',
  position: 0,
  itemId: null,
  playback: null,
  error: null,
})

export function reduceNativeVideoView(current, action) {
  if (action.type === 'target-native') {
    return { ...current, presentation: 'preparing_native', error: null }
  }
  if (action.type === 'target-web') {
    return {
      ...current,
      presentation: action.isVideo ? 'web_fallback' : 'react_image',
      error: null,
    }
  }
  if (action.type === 'runtime-error') {
    if (action.generation != null && action.generation !== current.generation) return current
    return { ...current, presentation: 'web_fallback', error: action.error || 'Native renderer failed' }
  }

  if (action.type !== 'snapshot' || !action.snapshot) return current
  const snapshot = action.snapshot
  if (snapshot.generation < current.generation) return current
  if (
    snapshot.generation === current.generation &&
    current.presentation === 'web_fallback' &&
    (snapshot.presentation === 'preparing_native' || snapshot.presentation === 'native_video')
  ) return current

  return {
    generation: snapshot.generation,
    presentation: snapshot.presentation,
    position: Number.isFinite(snapshot.position) ? Math.max(0, snapshot.position) : 0,
    itemId: snapshot.item_id ?? null,
    playback: snapshot.playback ?? current.playback,
    error: null,
  }
}

export function nativeViewFlags(view) {
  return {
    useNative: view.presentation === 'preparing_native' || view.presentation === 'native_video',
    preparing: view.presentation === 'preparing_native',
    visible: view.presentation === 'native_video',
    fallbackPosition: view.presentation === 'web_fallback' ? view.position : 0,
    playback: view.playback,
  }
}

export function nativeNavigationAction(event, generation) {
  if (!event || event.generation !== generation) return null
  if (event.type === 'navigate_previous') return -1
  if (event.type === 'navigate_next') return 1
  if (event.type === 'close_requested') return 'close'
  return null
}

export function snapshotMatchesTarget(snapshot, target) {
  if (!snapshot) return false
  const ownsNative = snapshot.presentation === 'preparing_native' || snapshot.presentation === 'native_video'
  if (target === 'native-local') return snapshot.presentation !== 'react_image'
  return !ownsNative
}

export function captureWebPlaybackHandoff(media, fallbackPosition = 0) {
  if (!media) return null
  return {
    position: Number.isFinite(media.currentTime) ? Math.max(0, media.currentTime) : Math.max(0, fallbackPosition),
    paused: Boolean(media.paused),
    volume: Number.isFinite(media.volume) ? Math.max(0, Math.min(1, media.volume)) : 1,
    muted: Boolean(media.muted),
    speed: Number.isFinite(media.playbackRate) ? Math.max(0.5, Math.min(2, media.playbackRate)) : 1,
  }
}
