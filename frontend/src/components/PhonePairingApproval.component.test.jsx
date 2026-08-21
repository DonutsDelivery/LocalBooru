import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PhonePairingApproval from './PhonePairingApproval'
import { authorizeServerForDesktop, confirmDeviceAuthorization } from '../devicePairing'

vi.mock('../devicePairing', () => ({
  authorizeServerForDesktop: vi.fn(),
  confirmDeviceAuthorization: vi.fn(),
  formatPairingFingerprint: value => value,
  isEligiblePairingServerUrl: value => value.startsWith('https://') || value.startsWith('http://192.168.'),
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
  { id: 'lan', name: 'LAN Server', url: 'http://192.168.1.20:8790', token: 'phone-token-lan' },
  { id: 'http', name: 'Public HTTP', url: 'http://lan.example', token: 'phone-token-http' },
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
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.clearAllMocks()
  })

  it('auto-selects one eligible server and relies on native device approval', async () => {
    const onComplete = vi.fn()
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={[servers[2]]} onComplete={onComplete} onClose={vi.fn()} />,
    ))

    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    expect(choices).toHaveLength(1)
    expect(choices[0].checked).toBe(true)
    expect(container.textContent).toContain('Desktop destination: http://192.168.1.25:8790')
    expect(container.querySelector('.pairing-name-confirmation')).toBeNull()

    const authorizeButton = container.querySelector('.pairing-authorize')
    expect(authorizeButton.disabled).toBe(false)

    await act(async () => authorizeButton.click())

    expect(confirmDeviceAuthorization).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop).toHaveBeenCalledWith(servers[2], request)
    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(container.textContent).toContain('LAN Server: authorized')
  })

  it('authorizes each explicitly selected HTTPS or private-LAN token session independently', async () => {
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={servers} onClose={vi.fn()} />,
    ))

    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    expect(choices).toHaveLength(3)
    expect(container.textContent).not.toContain('Public HTTP')
    expect(container.textContent).not.toContain('Legacy Password')
    expect(container.querySelector('.pairing-authorize').disabled).toBe(true)
    await act(async () => choices[0].click())
    await act(async () => choices[2].click())
    await act(async () => container.querySelector('.pairing-authorize').click())

    expect(confirmDeviceAuthorization).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop.mock.calls.map(([server]) => server.id).sort()).toEqual(['lan', 'one'])
  })
})
