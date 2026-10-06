// SPDX-License-Identifier: GPL-3.0-only
export function getDevProxyTarget(port) {
  if (typeof port !== 'string' || !/^\d+$/.test(port) || Number(port) < 1024 || Number(port) > 65535) {
    throw new Error('VITE_LOCALBOORU_PORT must explicitly select the backend. Use npm run app:test-isolated for testing.')
  }
  return `http://127.0.0.1:${Number(port)}`
}
