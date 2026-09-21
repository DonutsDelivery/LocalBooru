import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { fireEvent } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PhonePairingApproval from './PhonePairingApproval'
import { authorizeServerForDesktop, confirmDeviceAuthorization } from '../devicePairing'
import { getServers } from '../serverManager'

vi.mock('../devicePairing', () => ({
  authorizeServerForDesktop: vi.fn(),
  confirmDeviceAuthorization: vi.fn(),
  formatPairingFingerprint: value => value,
  // Mirror the real rule: token session + HTTPS or private-LAN HTTP.
  isEligiblePairingServerUrl: value => {
    try {
      const url = new URL(value)
      if (url.protocol !== 'https:' && url.protocol !== 'http:') return false
      if (url.protocol === 'https:') return true
      return /^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.|localhost$)/.test(url.hostname)
    } catch {
      return false
    }
  },
}))

vi.mock('../serverManager', () => ({
  getServers: vi.fn(),
}))

globalThis.IS_REACT_ACT_ENVIRONMENT = true

const request = {
  displayName: 'Bedroom Desktop',
  publicKeyFingerprint: 'abcd1234',
  callbackUrls: ['http://192.168.1.25:8790'],
  expiresAt: 4_000_000_000,
}

const servers = [
  { id: 'one', name: 'Server One', url: 'https://one.example', token: 'phone-token-one' },
  { id: 'two', name: 'Server Two', url: 'https://two.example', token: 'phone-token-two' },
  { id: 'lan', name: 'LAN HTTP Server', url: 'http://192.168.1.50:8790', token: 'phone-token-lan' },
  { id: 'public-http', name: 'Public HTTP', url: 'http://example.com', token: 'phone-token-public' },
  { id: 'legacy', name: 'Legacy Password', url: 'https://legacy.example' },
]

describe('PhonePairingApproval', () => {
  let container
  let root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    confirmDeviceAuthorization.mockResolvedValue()
    authorizeServerForDesktop.mockImplementation(async server => ({ serverId: server.id }))
    getServers.mockResolvedValue(servers)
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.clearAllMocks()
  })

  it('authorizes selected servers after device approval without a typed name gate', async () => {
    const onComplete = vi.fn()
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={servers} onComplete={onComplete} onClose={vi.fn()} />,
    ))

    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    expect(choices).toHaveLength(3)
    expect(container.textContent).not.toContain('Public HTTP')
    expect(container.textContent).not.toContain('Legacy Password')
    expect(container.textContent).toContain('Desktop destination: http://192.168.1.25:8790')
    expect(container.textContent).not.toContain('To confirm the requesting device')

    const authorizeButton = container.querySelector('.pairing-authorize')
    expect(authorizeButton.disabled).toBe(true)

    await act(async () => choices[0].click())
    expect(authorizeButton.disabled).toBe(false)

    await act(async () => authorizeButton.click())

    expect(confirmDeviceAuthorization).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop).toHaveBeenCalledWith(servers[0], request)
    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(container.textContent).toContain('Server One: authorized')
  })

  it('authorizes each selected eligible session, including private-LAN HTTP', async () => {
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={servers} onClose={vi.fn()} />,
    ))

    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    await act(async () => choices[0].click())
    await act(async () => choices[1].click())
    await act(async () => choices[2].click())
    await act(async () => container.querySelector('.pairing-authorize').click())

    expect(confirmDeviceAuthorization).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop.mock.calls.map(([server]) => server.id).sort()).toEqual(['lan', 'one', 'two'])
  })

  it('re-reads the live server store instead of trusting a stale snapshot', async () => {
    // Regression: the phone paired to a server earlier in the same session, so
    // the caller still holds the pre-pairing (token-less) list. The approval
    // dialog must fetch the live store and offer the newly paired server.
    const stale = [{ id: 'one', name: 'Server One', url: 'https://one.example', token: null }]
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={stale} onClose={vi.fn()} />,
    ))

    await act(async () => {}) // let the live-store fetch settle
    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    expect(choices).toHaveLength(3)
    expect(container.textContent).toContain('Server Two')
    expect(container.textContent).toContain('LAN HTTP Server')
  })
})
