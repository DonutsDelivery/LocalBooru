import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import QRConnect from './QRConnect'
import { getQRData } from '../api'

vi.mock('qrcode.react', () => ({ QRCodeSVG: () => <div data-testid="qr" /> }))
vi.mock('../api', () => ({ getQRData: vi.fn() }))
vi.mock('../serverManager', () => ({ isMobileApp: () => false }))
vi.mock('./DesktopPairing', () => ({ default: () => <div data-testid="desktop-pairing">Show authorization QR</div> }))
vi.mock('./PairedDevices', () => ({ default: () => <div data-testid="paired-devices" /> }))

globalThis.IS_REACT_ACT_ENVIRONMENT = true

describe('QRConnect', () => {
  let container
  let root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.clearAllMocks()
  })

  it('keeps reverse desktop authorization visible when legacy QR data is unavailable', async () => {
    getQRData.mockRejectedValue(new Error('local network disabled'))

    await act(async () => root.render(<QRConnect active />))

    expect(container.querySelector('[data-testid="desktop-pairing"]')).not.toBeNull()
    expect(container.textContent).toContain('Show authorization QR')
    expect(container.textContent).toContain('Could not fetch server info')
  })
})