import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ServerSettings from './ServerSettings'

const { connectToServer, probeServer, testServerConnection } = vi.hoisted(() => ({
  connectToServer: vi.fn(), probeServer: vi.fn(), testServerConnection: vi.fn(),
}))
const server = { id: 'fixture', name: 'Fixture server', url: 'http://192.168.1.10:8790', fallbackUrl: 'http://100.64.1.10:8790', token: 'synthetic-token' }
vi.mock('../serverManager', async importOriginal => ({
  ...await importOriginal(), isTauriApp: () => true, isMobileApp: () => false,
  getServers: async () => [server], getActiveServerId: async () => '__local__',
  probeServer, testServerConnection,
}))
vi.mock('../api', () => ({ connectToServer, updateServerConfig: vi.fn(), verifyHandshake: vi.fn() }))
beforeEach(() => {
  vi.resetAllMocks()
  probeServer.mockResolvedValue({ success: true, url: server.fallbackUrl })
})
afterEach(cleanup)

it('switches the desktop library through the verified connection without reloading the window', async () => {
  const onServerChange = vi.fn()
  render(<ServerSettings onServerChange={onServerChange} />)
  fireEvent.click(await screen.findByText(server.name))
  await waitFor(() => expect(onServerChange).toHaveBeenCalledOnce())
  expect(connectToServer).toHaveBeenCalledWith(server)
  expect(screen.getByText('This Device')).toBeTruthy()
})

it('keeps settings usable and local active after a failed connection', async () => {
  connectToServer.mockRejectedValue(new Error('Synthetic network failure'))
  const onServerChange = vi.fn()
  render(<ServerSettings onServerChange={onServerChange} />)
  fireEvent.click(await screen.findByText(server.name))
  expect((await screen.findByRole('alert')).textContent).toContain('Synthetic network failure')
  expect(onServerChange).not.toHaveBeenCalled()
  expect(screen.queryByRole('status')).toBeNull()
  expect(screen.getByText('Active').closest('.server-card').textContent).toContain('This Device')
})

it('checks the fallback when displaying status and retains pairing credentials in the editor', async () => {
  testServerConnection.mockResolvedValueOnce({ success: false, networkFailure: true })
    .mockResolvedValueOnce({ success: true })
  render(<ServerSettings />)
  fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
  fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))
  await screen.findByText('Connection successful (via fallback URL)!')
  expect(probeServer).toHaveBeenCalledWith(server)
  expect(testServerConnection).toHaveBeenNthCalledWith(1, server.url, '', '', server.token)
  expect(testServerConnection).toHaveBeenNthCalledWith(2, server.fallbackUrl, '', '', server.token)
})
