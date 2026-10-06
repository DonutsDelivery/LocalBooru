// SPDX-License-Identifier: GPL-3.0-only
export function getIsolatedInstanceId(nativeWindow = globalThis.window, buildId = import.meta.env?.VITE_LOCALBOORU_ISOLATED_INSTANCE_ID) {
  const native = nativeWindow?.__TAURI_INTERNALS__ !== undefined || nativeWindow?.location?.protocol === 'tauri:' || nativeWindow?.location?.hostname === 'tauri.localhost'
  const value = native ? nativeWindow.__LOCALBOORU_ISOLATED_INSTANCE_ID__ : buildId
  if (value == null || value === '') return null
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) throw new Error('Invalid isolated backend identity')
  return value
}

export function verifyIsolatedHealth(data, expected) {
  if (!data || data.status !== 'healthy' || data.isolated_instance_id !== expected) {
    throw new Error('Backend does not belong to this isolated launch; refusing access')
  }
  return data
}
