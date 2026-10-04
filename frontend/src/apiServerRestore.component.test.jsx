import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import axios from 'axios'
import { apiClient, updateServerConfig, connectToServer, refreshConnectedServerAddresses } from './api'

const { invoke, server, probeServer, selection, learnServerAddresses } = vi.hoisted(() => ({
  invoke: vi.fn(),
  server: { id: 'fixture', url: 'http://192.168.1.10:8790', fallbackUrl: 'http://100.64.1.10:8790' },
  probeServer: vi.fn(),
  learnServerAddresses: vi.fn(),
  selection: { id: 'fixture' },
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke }))
vi.mock('./serverManager', async importOriginal => ({
  ...await importOriginal(), getActiveServer: async () => selection.id === 'fixture' ? server : { id: '__local__', isLocal: true },
  getActiveServerId: async () => selection.id,
  setActiveServerId: async id => { selection.id = id }, probeServer, learnServerAddresses,
}))
beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(axios, 'get').mockResolvedValue({ data: {} })
  learnServerAddresses.mockReset()
  window.__TAURI_INTERNALS__ = {}
  invoke.mockReset()
  probeServer.mockReset()
  selection.id = 'fixture'
  delete server.username
  delete server.password
})
afterEach(async () => {
  selection.id = '__local__'
  invoke.mockResolvedValue(undefined)
  await updateServerConfig()
  delete window.__TAURI_INTERNALS__
  vi.restoreAllMocks()
  vi.useRealTimers()
})

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

it('does not select an unreachable server', async () => {
  selection.id = '__local__'
  probeServer.mockResolvedValue({ success: false, error: 'Synthetic offline' })
  await expect(connectToServer(server)).rejects.toThrow('Synthetic offline')
  expect(selection.id).toBe('__local__')
  expect(invoke).not.toHaveBeenCalled()
})

it('selects and configures the verified fallback', async () => {
  selection.id = '__local__'
  probeServer.mockResolvedValue({ success: true, url: server.fallbackUrl })
  await connectToServer(server)
  expect(selection.id).toBe(server.id)
  expect(invoke).toHaveBeenCalledWith('set_remote_proxy', {
    url: server.fallbackUrl, fallbackUrl: server.url, token: null,
  })
})

it('restores the previous library and proxy after setup failure', async () => {
  selection.id = '__local__'
  probeServer.mockResolvedValue({ success: true, url: server.fallbackUrl })
  invoke.mockRejectedValueOnce(new Error('Synthetic IPC failure')).mockResolvedValue(undefined)
  await expect(connectToServer(server)).rejects.toThrow('Synthetic IPC failure')
  expect(selection.id).toBe('__local__')
  expect(invoke).toHaveBeenLastCalledWith('set_remote_proxy', { url: null, fallbackUrl: null, token: null })
  expect(apiClient.defaults.baseURL).not.toContain('/remote/')
})

it('sends desktop Basic credentials through the configured proxy', async () => {
  server.username = 'fixture'
  server.password = 'synthetic-password'
  await updateServerConfig(server.fallbackUrl)
  const adapter = vi.fn(async config => ({ data: {}, status: 200, headers: {}, config }))
  await apiClient.get('/library/stats', { adapter })
  expect(adapter.mock.calls[0][0].headers.Authorization).toBe('Basic ' + btoa('fixture:synthetic-password'))
})

it('learns addresses asynchronously after connect and refreshes on focus and interval', async () => {
  learnServerAddresses.mockResolvedValue({ ...server, tailscaleUrl: server.fallbackUrl })
  await updateServerConfig(server.url)
  await vi.waitFor(() => expect(learnServerAddresses).toHaveBeenCalledTimes(1))
  expect(axios.get).toHaveBeenCalledWith(expect.stringMatching(/remote\/api\/network\/addresses$/), expect.objectContaining({ timeout: 5000 }))
  expect(invoke).toHaveBeenCalledWith('set_remote_proxy', {
    url: server.url, fallbackUrl: server.fallbackUrl, token: null, expectedUrl: server.url,
  })
  window.dispatchEvent(new Event('focus'))
  await vi.waitFor(() => expect(learnServerAddresses).toHaveBeenCalledTimes(2))
  await vi.advanceTimersByTimeAsync(300_000)
  expect(learnServerAddresses).toHaveBeenCalledTimes(3)
})

it('discards an old discovery response after switching to this device', async () => {
  let reply
  axios.get.mockImplementationOnce(() => new Promise(resolve => { reply = resolve }))
  await updateServerConfig(server.url)
  await vi.waitFor(() => expect(reply).toBeTypeOf('function'))
  const signal = axios.get.mock.calls[0][1].signal
  selection.id = '__local__'
  await updateServerConfig()
  reply({ data: { tailscale_url: server.fallbackUrl } })
  await refreshConnectedServerAddresses(server.id, server.url)
  expect(signal.aborted).toBe(true)
  expect(learnServerAddresses).not.toHaveBeenCalled()
  expect(invoke).toHaveBeenLastCalledWith('set_remote_proxy', { url: null, fallbackUrl: null, token: null })
})

it('optional discovery failures leave the connected proxy untouched and produce no error toast', async () => {
  axios.get.mockRejectedValue(new Error('Synthetic offline peer'))
  const dispatch = vi.spyOn(window, 'dispatchEvent')
  await updateServerConfig(server.url)
  await vi.waitFor(() => expect(axios.get).toHaveBeenCalledTimes(1))
  await refreshConnectedServerAddresses(server.id, server.url)
  expect(learnServerAddresses).not.toHaveBeenCalled()
  expect(dispatch.mock.calls.some(([event]) => event.type === 'api-error')).toBe(false)
  expect(invoke.mock.calls.filter(([command]) => command === 'set_remote_proxy')).toHaveLength(1)
})
