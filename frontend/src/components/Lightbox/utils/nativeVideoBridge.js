const NATIVE_VIDEO_EVENT = 'localbooru-native-video'

function nativeBridge() {
  return typeof window !== 'undefined' ? window.AndroidNativeVideo : null
}

export function isAndroidNativeVideoAvailable() {
  const bridge = nativeBridge()
  if (!bridge) return false
  try {
    return bridge.available() === true
  } catch {
    return false
  }
}

export function nativeReductionHeight(quality) {
  switch (quality) {
    case '1440p': return 1440
    case '1080p_enhanced':
    case '1080p': return 1080
    case '720p': return 720
    case '480p': return 480
    default: return 0
  }
}

function call(method, ...args) {
  const bridge = nativeBridge()
  if (!bridge || typeof bridge[method] !== 'function') return false
  bridge[method](...args)
  return true
}

export function createNativeVideoAdapter(generation, initial = {}) {
  const listeners = new Map()
  let currentTime = Number(initial.currentTime) || 0
  let duration = Number(initial.duration) || 0
  let bufferedEnd = 0
  let paused = initial.paused !== false
  let volume = Number.isFinite(initial.volume) ? initial.volume : 1
  let muted = Boolean(initial.muted)
  let playbackRate = Number(initial.playbackRate) || 1
  let ended = false
  let videoWidth = 0
  let videoHeight = 0
  let error = null
  let loop = false

  const dispatch = (type) => {
    const event = { type, target: adapter, currentTarget: adapter }
    listeners.get(type)?.forEach(listener => listener(event))
  }

  const onNativeEvent = (event) => {
    const detail = event.detail
    if (!detail || Number(detail.generation) !== generation) return
    const value = detail.value
    switch (detail.type) {
      case 'position':
        currentTime = Number(value?.position) || 0
        duration = Number(value?.duration) || duration
        bufferedEnd = Number(value?.buffered) || 0
        dispatch('timeupdate')
        dispatch('progress')
        break
      case 'ready':
        duration = Number(value?.duration) || duration
        dispatch('loadedmetadata')
        dispatch('durationchange')
        dispatch('canplay')
        break
      case 'first-frame':
        dispatch('firstframe')
        dispatch('playing')
        break
      case 'playing':
        paused = !value
        dispatch(value ? 'play' : 'pause')
        if (value) dispatch('playing')
        break
      case 'buffering':
        dispatch(value ? 'waiting' : 'canplay')
        break
      case 'video-size':
        videoWidth = Number(value?.sourceWidth) || videoWidth
        videoHeight = Number(value?.sourceHeight) || videoHeight
        adapter.nativeVideoDiagnostics = value
        console.info('[LocalBooru Native Video]', value)
        dispatch('resize')
        break
      case 'ended':
        ended = true
        paused = true
        dispatch('ended')
        break
      case 'error':
        error = { code: 4, message: String(value || 'Android playback failed') }
        dispatch('error')
        break
    }
  }
  window.addEventListener(NATIVE_VIDEO_EVENT, onNativeEvent)

  const adapter = {
    __nativeVideo: true,
    nativeVideoDiagnostics: null,
    generation,
    buffered: {
      get length() { return bufferedEnd > 0 ? 1 : 0 },
      start() { return 0 },
      end() { return bufferedEnd },
    },
    textTracks: [],
    get currentTime() { return currentTime },
    set currentTime(value) {
      currentTime = Math.max(0, Number(value) || 0)
      ended = false
      call('seek', generation, currentTime)
    },
    get duration() { return duration },
    get paused() { return paused },
    get ended() { return ended },
    get volume() { return volume },
    set volume(value) {
      volume = Math.max(0, Math.min(1, Number(value) || 0))
      call('setVolume', generation, volume)
    },
    get muted() { return muted },
    set muted(value) {
      muted = Boolean(value)
      call('setMuted', generation, muted)
    },
    get playbackRate() { return playbackRate },
    set playbackRate(value) {
      playbackRate = Math.max(0.25, Math.min(4, Number(value) || 1))
      call('setSpeed', generation, playbackRate)
    },
    get loop() { return loop },
    set loop(value) { loop = Boolean(value) },
    get videoWidth() { return videoWidth },
    get videoHeight() { return videoHeight },
    get readyState() { return duration > 0 ? 4 : 0 },
    get networkState() { return error ? 3 : 1 },
    get error() { return error },
    play() {
      paused = false
      ended = false
      call('play', generation)
      dispatch('play')
      return Promise.resolve()
    },
    pause() {
      if (paused) return
      paused = true
      call('pause', generation)
      dispatch('pause')
    },
    fastSeek(value) { this.currentTime = value },
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set())
      listeners.get(type).add(listener)
    },
    removeEventListener(type, listener) {
      listeners.get(type)?.delete(listener)
    },
    getBoundingClientRect() {
      return initial.viewport?.getBoundingClientRect?.() || new DOMRect()
    },
    querySelectorAll() { return [] },
    addTextTrack() { return null },
    removeAttribute() {},
    load() {},
    destroy() {
      window.removeEventListener(NATIVE_VIDEO_EVENT, onNativeEvent)
      listeners.clear()
    },
  }

  return adapter
}

export function openNativeVideo(generation, url, options = {}) {
  const rect = options.viewport?.getBoundingClientRect?.()
  if (!rect || rect.width <= 0 || rect.height <= 0) return false
  return call(
    'open',
    generation,
    url,
    Number(options.startPosition) || 0,
    options.autoplay !== false,
    nativeReductionHeight(options.quality),
    rect.left,
    rect.top,
    rect.width,
    rect.height,
    window.devicePixelRatio || 1,
  )
}

export function closeNativeVideo(generation) {
  return call('close', generation)
}

export function setNativeVideoViewport(generation, element) {
  if (!element) return false
  const rect = element.getBoundingClientRect()
  return call(
    'setViewport',
    generation,
    rect.left,
    rect.top,
    rect.width,
    rect.height,
    window.devicePixelRatio || 1,
  )
}

export function setNativeVideoReduction(generation, quality) {
  return call('setReduction', generation, nativeReductionHeight(quality))
}

export function setNativeVideoDisplayMode(generation, mode) {
  return call('setDisplayMode', generation, mode)
}
