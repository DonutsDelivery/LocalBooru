import { afterEach, expect, test, vi } from 'vitest'

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke }))

afterEach(() => {
  localStorage.clear()
  delete window.__TAURI_INTERNALS__
  vi.unstubAllGlobals()
  vi.resetModules()
  invoke.mockReset()
})

// AC: @library-management — local selection survives paired remote libraries.
test('This Device selection survives a restart with a saved paired server and clears its proxy', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.stubGlobal('navigator', { userAgent: 'Android' })
  localStorage.setItem('localbooru_servers', JSON.stringify([{ id: 'paired-fixture', name: 'Synthetic PC', url: 'http://192.0.2.1:8790' }]))
  invoke.mockResolvedValue({})
  let manager = await import('./serverManager')
  await manager.setActiveServerId(manager.LOCAL_SERVER.id)
  vi.resetModules()
  manager = await import('./serverManager')
  expect(await manager.getActiveServer()).toEqual(manager.LOCAL_SERVER)
  const api = await import('./api')
  await api.updateServerConfig()
  expect(invoke).toHaveBeenCalledWith('set_remote_proxy', { url: null, fallbackUrl: null, token: null })
  expect(api.getApiUrl()).toBe('http://127.0.0.1:8790/api')
  expect(api.isServerConfigured()).toBe(true)
})

test('a failed proxy reset cannot silently present a remote library as This Device', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.stubGlobal('navigator', { userAgent: 'Android' })
  const manager = await import('./serverManager')
  await manager.setActiveServerId(manager.LOCAL_SERVER.id)
  invoke.mockRejectedValue(new Error('Native connection unavailable'))
  const api = await import('./api')
  await expect(api.updateServerConfig()).rejects.toThrow('Native connection unavailable')
})
