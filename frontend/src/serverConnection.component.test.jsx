import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { probeServer, testServerConnection } from './serverManager'

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
