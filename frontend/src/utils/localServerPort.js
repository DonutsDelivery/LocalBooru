// SPDX-License-Identifier: GPL-3.0-only
function validPort(value) {
  if (typeof value !== 'number' && (typeof value !== 'string' || !/^\d+$/.test(value))) return null
  const port = Number(value)
  return Number.isInteger(port) && port > 0 && port <= 65535 ? String(port) : null
}

// Native initialization supplies the running process's port before app modules
// load. Browser pages keep their build configuration and ignore this property.
export function getLocalServerPort(nativeWindow = globalThis.window, buildPort = import.meta.env?.VITE_LOCALBOORU_PORT) {
  const nativeOrigin = nativeWindow?.location?.protocol === 'tauri:' || nativeWindow?.location?.hostname === 'tauri.localhost'
  if (nativeWindow?.__TAURI_INTERNALS__ !== undefined || nativeOrigin) {
    const nativePort = validPort(nativeWindow.__LOCALBOORU_PORT__)
    if (nativePort) return nativePort
    throw new Error('Native backend port is unavailable; refusing to connect to another application instance.')
  }
  return validPort(buildPort) || '8790'
}
