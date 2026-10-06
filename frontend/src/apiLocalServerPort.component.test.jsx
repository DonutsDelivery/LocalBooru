import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(async () => null) }))

afterEach(() => {
  delete window.__TAURI_INTERNALS__
  delete window.__LOCALBOORU_PORT__
  localStorage.clear()
  vi.resetModules()
})

it('prebuilt native API, media, and local server selection use the isolated process port', async () => {
  window.__TAURI_INTERNALS__ = {}
  window.__LOCALBOORU_PORT__ = 18790
  vi.resetModules()
  const api = await import('./api.js')
  const servers = await import('./serverManager.js')
  expect(api.getApiUrl()).toBe('http://127.0.0.1:18790/api')
  expect(api.apiClient.defaults.baseURL).toBe('http://127.0.0.1:18790/api')
  expect(api.getMediaUrl('/api/images/42/file')).toBe('http://127.0.0.1:18790/api/images/42/file')
  expect(await servers.getApiBaseUrl()).toBe('http://127.0.0.1:18790/api')
  const adapter = vi.fn(async config => ({ data: {}, status: 200, headers: {}, config }))
  await api.apiClient.get('/library/stats', { adapter })
  expect(adapter.mock.calls[0][0].baseURL).toBe('http://127.0.0.1:18790/api')
})

it('normal native initialization retains the existing default backend port', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.resetModules()
  const api = await import('./api.js')
  expect(api.getApiUrl()).toBe('http://127.0.0.1:8790/api')
  expect(api.getMediaUrl('/api/images/42/file')).toBe('http://127.0.0.1:8790/api/images/42/file')
})

it('browser pages ignore spoofed native port data and keep same-origin API connections', async () => {
  window.__LOCALBOORU_PORT__ = 18790
  vi.resetModules()
  const api = await import('./api.js')
  expect(api.getApiUrl()).toBe('/api')
})
