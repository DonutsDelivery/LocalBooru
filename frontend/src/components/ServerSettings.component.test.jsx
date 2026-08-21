import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ServerSettings from './ServerSettings'
import {
  getActiveServerId,
  getServers,
  isMobileApp,
  isTauriApp,
  testServerConnection,
} from '../serverManager'

vi.mock('../serverManager', () => ({
  LOCAL_SERVER: { id: '__local__', name: 'This Device', url: null, isLocal: true },
  getServers: vi.fn(),
  addServer: vi.fn(),
  addOrUpdateServer: vi.fn(),
  updateServer: vi.fn(),
  removeServer: vi.fn(),
  getActiveServerId: vi.fn(),
  setActiveServerId: vi.fn(),
  testServerConnection: vi.fn(),
  serverFromQrHandshake: vi.fn(),
  isMobileApp: vi.fn(),
  isTauriApp: vi.fn(),
}))
vi.mock('../api', () => ({ updateServerConfig: vi.fn(), verifyHandshake: vi.fn() }))
vi.mock('../devicePairing', () => ({ validateDesktopPairingRequest: vi.fn() }))
vi.mock('../qrScanner', () => ({ scanQrCode: vi.fn() }))
vi.mock('./PhonePairingApproval', () => ({ default: () => null }))

globalThis.IS_REACT_ACT_ENVIRONMENT = true

const remoteServer = {
  id: 'linux-server',
  name: 'Linux Library',
  url: 'http://192.168.1.20:8790',
  token: 'protected-by-mock',
}

describe('ServerSettings native desktop catalog', () => {
  let container
  let root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    isMobileApp.mockReturnValue(false)
    isTauriApp.mockReturnValue(true)
    getServers.mockResolvedValue([remoteServer])
    getActiveServerId.mockResolvedValue(remoteServer.id)
    testServerConnection.mockResolvedValue({ success: true })
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.clearAllMocks()
  })

  it('shows This Device and saved remote servers without mobile-only scanning', async () => {
    await act(async () => root.render(<ServerSettings />))

    expect(container.textContent).toContain('This Device')
    expect(container.textContent).toContain('Linux Library')
    expect(container.textContent).toContain('Embedded local backend')
    expect(container.textContent).not.toContain('Scan QR')
    expect(container.querySelector('.server-card.local-server .connect-btn')?.textContent).toBe('Use server')
  })

  it('keeps hosted browsers bound to the server that served the page', async () => {
    isTauriApp.mockReturnValue(false)
    await act(async () => root.render(<ServerSettings />))

    expect(container.textContent).toContain('Running as web app')
    expect(container.textContent).not.toContain('This Device')
  })
})
