const VR_MARKERS = {
  180: /(?:hequirect|vr[\s._-]*180|180[\s._-]*vr|180[\s._-]*(?:deg|degree|degrees)|(?:^|[^0-9])180(?:[^0-9]|$))/i,
  360: /(?:(?:^|[^a-z])equirect|vr[\s._-]*360|360[\s._-]*vr|360[\s._-]*(?:deg|degree|degrees)|(?:^|[^0-9])360(?:[^0-9]|$))/i,
}

const VR_STEREO_MARKERS = {
  sbs: /(?:^|[\s._-])(?:sbs|3dh|side[\s._-]*by[\s._-]*side)(?=[\s._-]|$)/i,
  tb: /(?:^|[\s._-])(?:tb|3dv|top[\s._-]*(?:bottom|and[\s._-]*bottom)|over[\s._-]*under)(?=[\s._-]|$)/i,
}

const VR_INPUT_PROJECTION_MARKERS = {
  fisheye: /(?:^|[\s._-])(?:dual[\s._-]*)?fish[\s._-]*eye(?=[\s._-]|$)/i,
}

export const DEFAULT_VR_CAMERA = Object.freeze({
  yaw: 0,
  pitch: 0,
  roll: 0,
})

export const DEFAULT_VR_FOV = 100
export const MIN_VR_FOV = 30
export const MAX_VR_FOV = 120
export const VR_SETTINGS_STORAGE_KEY = 'localbooru_vr_view_settings'
export const DEFAULT_VR_CONFIG = Object.freeze({
  inputProjection: 'fisheye',
  projection: '180',
  stereo: 'sbs',
  eye: 'left',
  fov: DEFAULT_VR_FOV,
})

export function loadVRConfig(storage = globalThis.localStorage) {
  if (!storage) return { ...DEFAULT_VR_CONFIG }
  try {
    const saved = JSON.parse(storage.getItem(VR_SETTINGS_STORAGE_KEY) || '{}')
    return {
      inputProjection: ['equirectangular', 'fisheye'].includes(saved.inputProjection)
        ? saved.inputProjection : DEFAULT_VR_CONFIG.inputProjection,
      projection: ['180', '360'].includes(saved.projection)
        ? saved.projection : DEFAULT_VR_CONFIG.projection,
      stereo: ['mono', 'sbs', 'tb'].includes(saved.stereo)
        ? saved.stereo : DEFAULT_VR_CONFIG.stereo,
      eye: ['left', 'right'].includes(saved.eye) ? saved.eye : DEFAULT_VR_CONFIG.eye,
      fov: Number.isFinite(saved.fov)
        ? clamp(saved.fov, MIN_VR_FOV, MAX_VR_FOV) : DEFAULT_VR_CONFIG.fov,
    }
  } catch {
    return { ...DEFAULT_VR_CONFIG }
  }
}

export function saveVRConfig(config, storage = globalThis.localStorage) {
  if (!storage) return
  try {
    storage.setItem(VR_SETTINGS_STORAGE_KEY, JSON.stringify(config))
  } catch {
    // Playback should continue when storage is unavailable or full.
  }
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

export function normalizeYaw(yaw) {
  let normalized = yaw
  while (normalized > 180) normalized -= 360
  while (normalized < -180) normalized += 360
  return normalized
}

/**
 * Detect common VR projection markers without treating dimensions such as 3840x1920
 * as VR. If both explicit degree markers occur, 360 wins because 360x180 is a
 * conventional equirectangular description.
 */
export function detectVRProjection(filename) {
  if (!filename) return null
  const has180 = VR_MARKERS[180].test(filename)
  const has360 = VR_MARKERS[360].test(filename)
  if (has360) return '360'
  if (has180) return '180'
  return null
}

export function detectVRStereo(filename) {
  if (!filename) return 'mono'
  if (VR_STEREO_MARKERS.sbs.test(filename)) return 'sbs'
  if (VR_STEREO_MARKERS.tb.test(filename)) return 'tb'
  return 'mono'
}

export function detectVRInputProjection(filename) {
  if (filename && VR_INPUT_PROJECTION_MARKERS.fisheye.test(filename)) return 'fisheye'
  return 'equirectangular'
}

export function fitVRTextureSize(width, height, maxTextureSize) {
  if (!Number.isFinite(width) || !Number.isFinite(height)
      || width <= 0 || height <= 0 || !Number.isFinite(maxTextureSize)
      || maxTextureSize <= 0) {
    return { width: 0, height: 0, scaled: false }
  }
  const scale = Math.min(1, maxTextureSize / width, maxTextureSize / height)
  return {
    width: Math.max(1, Math.floor(width * scale)),
    height: Math.max(1, Math.floor(height * scale)),
    scaled: scale < 1,
  }
}

export function shouldStageVRTexture(_platform, scaled) {
  // Only downscale when the decoded frame exceeds MAX_TEXTURE_SIZE.
  // Staging through a 2D canvas every frame bypasses WebKit's GPU video
  // copy (IOSurface on macOS, dmabuf on Linux) and reintroduces the lag
  // the original 4K/8K viewer did not have.
  return Boolean(scaled)
}

export function updateVRCamera(camera, deltaX, deltaY, projection = '360', sensitivity = 0.18) {
  const nextYaw = camera.yaw - deltaX * sensitivity
  return {
    ...camera,
    yaw: projection === '180'
      ? clamp(nextYaw, -90, 90)
      : normalizeYaw(nextYaw),
    pitch: clamp(camera.pitch + deltaY * sensitivity, -89, 89),
  }
}

export function updateVRFov(fov, delta) {
  return clamp(fov + delta, MIN_VR_FOV, MAX_VR_FOV)
}

export function vrPointerDelta(previous, clientX, clientY) {
  return {
    deltaX: clientX - previous.x,
    deltaY: clientY - previous.y,
  }
}

export function uploadVRVideoFrame(gl, video) {
  // WebKitGTK only offers its video-frame GPU copy path for a full texImage2D
  // DOM-source upload. Preallocating and using texSubImage2D forces the slow
  // software conversion path on Linux.
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    video,
  )
}
