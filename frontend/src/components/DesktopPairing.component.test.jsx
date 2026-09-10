import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import DesktopPairing from './DesktopPairing'
import {
  cancelDesktopPairingSession,
  createDesktopPairingSession,
} from '../devicePairing'

vi.mock('qrcode.react', () => ({
  QRCodeSVG: ({ size, level, boostLevel, marginSize }) => (
    <div
      data-testid="qr"
      data-size={size}
      data-level={level}
      data-boost-level={String(boostLevel)}
      data-margin-size={marginSize}
    />
  ),
}))
vi.mock('../api', () => ({ updateServerConfig: vi.fn() }))
vi.mock('../serverManager', () => ({ setActiveServerId: vi.fn() }))
vi.mock('../devicePairing', () => ({
  cancelDesktopPairingSession: vi.fn().mockResolvedValue(),
  createDesktopPairingSession: vi.fn(),
  formatPairingFingerprint: value => value,
  inspectDesktopDelivery: vi.fn(),
  pollDesktopPairingSession: vi.fn().mockResolvedValue({ deliveries: [] }),
  redeemDesktopPayload: vi.fn(),
}))

globalThis.IS_REACT_ACT_ENVIRONMENT = true

const pairing = {
  session: {
    sessionId: 'session-one',
    sessionSecret: 'secret-one',
    expiresAt: 4_000_000_000,
    displayName: 'Bedroom Desktop',
    publicKeyFingerprint: 'abcd1234',
  },
}

describe('DesktopPairing', () => {
  let container
  let root

  beforeEach(() => {
    const store = new Map()
    globalThis.localStorage = {
      getItem: key => (store.has(key) ? store.get(key) : null),
      setItem: (key, value) => { store.set(key, String(value)) },
      removeItem: key => { store.delete(key) },
      clear: () => { store.clear() },
    }
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    localStorage.clear()
    createDesktopPairingSession.mockResolvedValue(pairing)
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.clearAllMocks()
  })

  it('creates a session only after an explicit click and cancels it when the tab becomes inactive', async () => {
    await act(async () => root.render(<DesktopPairing active />))
    expect(createDesktopPairingSession).not.toHaveBeenCalled()

    await act(async () => container.querySelector('.pairing-primary').click())
    expect(createDesktopPairingSession).toHaveBeenCalledTimes(1)
    const qr = container.querySelector('[data-testid="qr"]')
    expect(qr).not.toBeNull()
    expect(qr.dataset.size).toBe('480')
    expect(qr.dataset.level).toBe('L')
    expect(qr.dataset.boostLevel).toBe('false')
    expect(qr.dataset.marginSize).toBe('4')

    await act(async () => root.render(<DesktopPairing active={false} />))
    expect(cancelDesktopPairingSession).toHaveBeenCalledWith('session-one', 'secret-one')
    expect(container.querySelector('[data-testid="qr"]')).toBeNull()
  })
})
