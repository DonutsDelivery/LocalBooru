import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ServerSelectScreen from './ServerSelectScreen'

const { connectToServer, testServerConnection } = vi.hoisted(() => ({
  connectToServer: vi.fn(), testServerConnection: vi.fn(),
}))
vi.mock('../serverManager', async importOriginal => ({
  ...await importOriginal(), testServerConnection,
}))
vi.mock('../api', () => ({ connectToServer, verifyHandshake: vi.fn() }))
beforeEach(() => { vi.resetAllMocks() })
afterEach(() => cleanup())

const server = { id: 'fixture', name: 'Fixture server', url: 'http://192.168.1.10:8790', fallbackUrl: 'http://100.64.1.10:8790' }

it('opens a paired server through the verified fallback', async () => {
  connectToServer.mockResolvedValue(undefined)
  const onConnect = vi.fn()
  render(<ServerSelectScreen servers={[server]} onConnect={onConnect} />)
  fireEvent.click(screen.getByText(server.name))
  await waitFor(() => expect(onConnect).toHaveBeenCalledOnce())
  expect(connectToServer).toHaveBeenCalledWith(server)
})

it('leaves selection available after proxy setup fails', async () => {
  connectToServer.mockResolvedValue(undefined)
  connectToServer.mockRejectedValue(new Error('Synthetic IPC failure'))
  const onConnect = vi.fn()
  render(<ServerSelectScreen servers={[server]} onConnect={onConnect} />)
  fireEvent.click(screen.getByText(server.name))
  await screen.findByText('Could not open Fixture server: Synthetic IPC failure')
  expect(screen.queryByText('Connecting...')).toBeNull()
  expect(onConnect).not.toHaveBeenCalled()
  expect(screen.getAllByRole('button', { name: 'Connect' })).toHaveLength(2)
})

it('tests edited paired credentials on both primary and fallback addresses', async () => {
  testServerConnection.mockResolvedValueOnce({ success: false, networkFailure: true })
    .mockResolvedValueOnce({ success: true })
  render(<ServerSelectScreen servers={[{ ...server, token: 'synthetic-token' }]} />)
  fireEvent.click(screen.getByTitle('Edit server').querySelector('path'))
  expect(connectToServer).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Test Connection' }))
  await screen.findByText('Connection successful (via fallback URL)!')
  expect(testServerConnection).toHaveBeenNthCalledWith(1, server.url, '', '', 'synthetic-token')
  expect(testServerConnection).toHaveBeenNthCalledWith(2, server.fallbackUrl, '', '', 'synthetic-token')
})
