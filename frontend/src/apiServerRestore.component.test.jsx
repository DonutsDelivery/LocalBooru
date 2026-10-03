import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { apiClient, updateServerConfig } from './api'

const { invoke, server, probeServer } = vi.hoisted(() => ({
  invoke: vi.fn(),
  server: { id: 'fixture', url: 'http://192.168.1.10:8790', fallbackUrl: 'http://100.64.1.10:8790' },
  probeServer: vi.fn(),
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke }))
vi.mock('./serverManager', async importOriginal => ({
  ...await importOriginal(), getActiveServer: async () => server, probeServer,
}))
beforeEach(() => {
  window.__TAURI_INTERNALS__ = {}
  invoke.mockReset()
  probeServer.mockReset()
})
afterEach(() => { delete window.__TAURI_INTERNALS__ })

it('configures the verified fallback as primary without probing the failed LAN again', async () => {
  invoke.mockResolvedValue(undefined)
  await updateServerConfig(server.fallbackUrl)
  expect(invoke).toHaveBeenCalledWith('set_remote_proxy', {
    url: server.fallbackUrl, fallbackUrl: server.url, token: null,
  })
  expect(probeServer).not.toHaveBeenCalled()
  expect(apiClient.defaults.baseURL).toMatch(/\/remote\/api$/)
})

it('reports proxy configuration failure to the startup caller', async () => {
  invoke.mockRejectedValue(new Error('Synthetic IPC failure'))
  await expect(updateServerConfig(server.fallbackUrl)).rejects.toThrow('Synthetic IPC failure')
})
