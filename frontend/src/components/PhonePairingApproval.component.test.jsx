import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { fireEvent } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PhonePairingApproval from './PhonePairingApproval'
import { authorizeServerForDesktop, confirmDeviceAuthorization } from '../devicePairing'

vi.mock('../devicePairing', () => ({
  authorizeServerForDesktop: vi.fn(),
  confirmDeviceAuthorization: vi.fn(),
  formatPairingFingerprint: value => value,
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
  { id: 'http', name: 'Unsafe HTTP', url: 'http://lan.example', token: 'phone-token-http' },
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

  it('requires explicit server selection, exact device name, and device approval', async () => {
    const onComplete = vi.fn()
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={servers} onComplete={onComplete} onClose={vi.fn()} />,
    ))

    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    expect(choices).toHaveLength(2)
    expect(container.textContent).not.toContain('Unsafe HTTP')
    expect(container.textContent).not.toContain('Legacy Password')
    expect(container.textContent).toContain('Desktop destination: http://192.168.1.25:8790')

    const authorizeButton = container.querySelector('.pairing-authorize')
    expect(authorizeButton.disabled).toBe(true)

    await act(async () => choices[0].click())
    expect(authorizeButton.disabled).toBe(true)

    const confirmation = container.querySelector('.pairing-name-confirmation input')
    await act(async () => {
      fireEvent.change(confirmation, { target: { value: 'Bedroom Desktop' } })
    })
    expect(authorizeButton.disabled).toBe(false)

    await act(async () => authorizeButton.click())

    expect(confirmDeviceAuthorization).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop).toHaveBeenCalledWith(servers[0], request)
    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(container.textContent).toContain('Server One: authorized')
  })

  it('authorizes each selected HTTPS token session independently', async () => {
    await act(async () => root.render(
      <PhonePairingApproval request={request} servers={servers} onClose={vi.fn()} />,
    ))

    const choices = [...container.querySelectorAll('input[type="checkbox"]')]
    await act(async () => choices[0].click())
    await act(async () => choices[1].click())
    const confirmation = container.querySelector('.pairing-name-confirmation input')
    await act(async () => {
      fireEvent.change(confirmation, { target: { value: 'Bedroom Desktop' } })
    })
    await act(async () => container.querySelector('.pairing-authorize').click())

    expect(confirmDeviceAuthorization).toHaveBeenCalledTimes(1)
    expect(authorizeServerForDesktop.mock.calls.map(([server]) => server.id).sort()).toEqual(['one', 'two'])
  })
})
