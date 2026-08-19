const VR_MARKERS = {
  180: /(?:hequirect|vr[\s._-]*180|180[\s._-]*vr|180[\s._-]*(?:deg|degree|degrees)|(?:^|[^0-9])180(?:[^0-9]|$))/i,
  360: /(?:(?:^|[^a-z])equirect|vr[\s._-]*360|360[\s._-]*vr|360[\s._-]*(?:deg|degree|degrees)|(?:^|[^0-9])360(?:[^0-9]|$))/i,
}

const VR_STEREO_MARKERS = {
  sbs: /(?:^|[\s._-])(?:sbs|side[\s._-]*by[\s._-]*side)(?=[\s._-]|$)/i,
  tb: /(?:^|[\s._-])(?:tb|top[\s._-]*(?:bottom|and[\s._-]*bottom)|over[\s._-]*under)(?=[\s._-]|$)/i,
}

export const DEFAULT_VR_CAMERA = Object.freeze({
  yaw: 0,
  pitch: 0,
  roll: 0,
})

export const DEFAULT_VR_FOV = 100
export const MIN_VR_FOV = 30
export const MAX_VR_FOV = 120

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
