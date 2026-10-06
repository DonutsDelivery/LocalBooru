import { afterEach, expect, it, vi } from 'vitest'
import axios from 'axios'

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(async () => null) }))

afterEach(() => {
  delete window.__TAURI_INTERNALS__
  delete window.__LOCALBOORU_PORT__
  delete window.__LOCALBOORU_ISOLATED_INSTANCE_ID__
  localStorage.clear()
  vi.resetModules()
  vi.unstubAllGlobals()
})

it.each(['5173', '5174', '5175', '5210'])('browser server selection remains same-origin on preview port %s', async port => {
  vi.stubGlobal('window', { location: { port } })
  vi.resetModules()
  const servers = await import('./serverManager.js')
  expect(await servers.getApiBaseUrl()).toBe('/api')
})

it('foreign health identity blocks native metadata requests before their adapter runs', async () => {
  window.__TAURI_INTERNALS__ = {}
  window.__LOCALBOORU_PORT__ = 18793
  window.__LOCALBOORU_ISOLATED_INSTANCE_ID__ = 'a'.repeat(64)
  vi.resetModules()
  const api = await import('./api.js')
  const probe = vi.spyOn(axios, 'get').mockResolvedValue({ data: { status: 'healthy', isolated_instance_id: 'b'.repeat(64) } })
  const events = vi.fn(function () { this.close = vi.fn() })
  vi.stubGlobal('EventSource', events)
  expect(() => api.getMediaUrl('/api/images/1/file')).toThrow('refusing access')
  expect(() => api.subscribeToLibraryEvents(() => {})).toThrow('refusing access')
  expect(() => api.subscribeToMigrationEvents(() => {})).toThrow('refusing access')
  expect(events).not.toHaveBeenCalled()
  const metadata = vi.fn(async config => ({ data: {}, status: 200, headers: {}, config }))
  await expect(api.apiClient.get('/images', { adapter: metadata })).rejects.toThrow('refusing access')
  expect(metadata).not.toHaveBeenCalled()
  expect(probe).toHaveBeenCalledWith('http://127.0.0.1:18793/health', { timeout: 2000 })
  probe.mockResolvedValue({ data: { status: 'healthy', isolated_instance_id: 'a'.repeat(64) } })
  await api.apiClient.get('/images', { adapter: metadata })
  expect(metadata).toHaveBeenCalledTimes(1)
  expect(api.getMediaUrl('/api/images/1/file')).toBe('http://127.0.0.1:18793/api/images/1/file')
  const stopEvents = api.subscribeToLibraryEvents(() => {})
  expect(events).toHaveBeenCalledTimes(1)
  stopEvents()
  probe.mockResolvedValue({ data: { status: 'healthy' } })
  await expect(api.healthCheck()).rejects.toThrow('refusing access')
  expect(() => api.getMediaUrl('/api/images/1/file')).toThrow('refusing access')
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
  window.__LOCALBOORU_PORT__ = 8790
  vi.resetModules()
  const api = await import('./api.js')
  expect(api.getApiUrl()).toBe('http://127.0.0.1:8790/api')
  expect(api.getMediaUrl('/api/images/42/file')).toBe('http://127.0.0.1:8790/api/images/42/file')
})

it('missing native runtime port rejects API initialization before issuing requests', async () => {
  window.__TAURI_INTERNALS__ = {}
  const request = vi.fn()
  vi.stubGlobal('fetch', request)
  vi.resetModules()
  await expect(import('./api.js')).rejects.toThrow('refusing to connect')
  expect(request).not.toHaveBeenCalled()
})

it('browser pages ignore spoofed native port data and keep same-origin API connections', async () => {
  window.__LOCALBOORU_PORT__ = 18790
  vi.resetModules()
  const api = await import('./api.js')
  expect(api.getApiUrl()).toBe('/api')
})
