import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { probeServer, testServerConnection, saveServers, getServers, learnServerAddresses } from './serverManager'

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke }))

beforeEach(() => {
  window.__TAURI_INTERNALS__ = {}
  invoke.mockReset()
})
afterEach(() => { delete window.__TAURI_INTERNALS__ })

describe('native paired connection probes', () => {
  it('uses native IPC and retains authorization when LAN is unreachable', async () => {
    invoke.mockResolvedValueOnce({ success: false, error: 'Offline', networkFailure: true })
      .mockResolvedValueOnce({ success: true, networkFailure: false })
    const server = { url: 'http://192.168.1.10:8790', fallbackUrl: 'http://100.64.1.10:8790', token: 'synthetic-token' }
    expect(await probeServer(server)).toMatchObject({ success: true, url: server.fallbackUrl, usedFallback: true })
    expect(invoke.mock.calls).toEqual([
      ['test_remote_server', { url: server.url, username: null, password: null, token: server.token }],
      ['test_remote_server', { url: server.fallbackUrl, username: null, password: null, token: server.token }],
    ])
  })

  it('normalizes legacy URLs and preserves Basic credentials', async () => {
    invoke.mockResolvedValue({ success: true, networkFailure: false })
    await testServerConnection('example.test:8790', 'fixture-user', 'fixture-password')
    expect(invoke).toHaveBeenCalledWith('test_remote_server', {
      url: 'http://example.test:8790', username: 'fixture-user', password: 'fixture-password', token: null,
    })
  })

  it('does not try another address after native auth rejection', async () => {
    invoke.mockResolvedValue({ success: false, error: 'Authentication required', networkFailure: false })
    expect(await probeServer({ url: 'http://example.test', fallbackUrl: 'http://fallback.test', token: 'fixture' }))
      .toMatchObject({ success: false, networkFailure: false })
    expect(invoke).toHaveBeenCalledTimes(1)
  })
})

it('persists learned metadata while keeping native credentials out of public storage', async () => {
  localStorage.clear()
  let credentials = {}
  invoke.mockImplementation(async (command, args) => {
    if (command === 'load_paired_server_credentials') return credentials
    if (command === 'store_paired_server_credentials') credentials = args.credentials
  })
  const server = { id: 'discovery-fixture', url: 'http://192.168.1.10:8790', token: 'synthetic-token', password: 'synthetic-password' }
  await saveServers([server])
  await learnServerAddresses(server.id, { server_id: 'discovery-fixture', server_port: 8790, tailscale_url: 'http://100.64.1.10:8790' }, server.url)
  const metadata = JSON.parse(localStorage.getItem('localbooru_servers'))[0]
  expect(metadata.fallbackUrl).toBe('http://100.64.1.10:8790')
  expect(metadata.token).toBeUndefined()
  expect(metadata.password).toBeUndefined()
  expect((await getServers())[0]).toMatchObject({ token: server.token, password: server.password, tailscaleUrl: metadata.fallbackUrl })
  expect(credentials[server.id]).toEqual({ token: server.token, password: server.password })
})
