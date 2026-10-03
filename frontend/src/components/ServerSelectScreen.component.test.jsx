import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ServerSelectScreen from './ServerSelectScreen'

const { probeServer, updateServerConfig, setActiveServerId } = vi.hoisted(() => ({
  probeServer: vi.fn(), updateServerConfig: vi.fn(), setActiveServerId: vi.fn(),
}))
vi.mock('../serverManager', async importOriginal => ({
  ...await importOriginal(), probeServer, setActiveServerId,
}))
vi.mock('../api', () => ({ updateServerConfig, verifyHandshake: vi.fn() }))
beforeEach(() => { vi.resetAllMocks() })
afterEach(() => cleanup())

const server = { id: 'fixture', name: 'Fixture server', url: 'http://192.168.1.10:8790', fallbackUrl: 'http://100.64.1.10:8790' }

it('opens a paired server through the verified fallback', async () => {
  probeServer.mockResolvedValue({ success: true, url: server.fallbackUrl })
  const onConnect = vi.fn()
  render(<ServerSelectScreen servers={[server]} onConnect={onConnect} />)
  fireEvent.click(screen.getByText(server.name))
  await waitFor(() => expect(onConnect).toHaveBeenCalledOnce())
  expect(updateServerConfig).toHaveBeenCalledWith(server.fallbackUrl)
  expect(setActiveServerId).toHaveBeenCalledWith(server.id)
})

it('leaves selection available after proxy setup fails', async () => {
  probeServer.mockResolvedValue({ success: true, url: server.fallbackUrl })
  updateServerConfig.mockRejectedValue(new Error('Synthetic IPC failure'))
  const onConnect = vi.fn()
  render(<ServerSelectScreen servers={[server]} onConnect={onConnect} />)
  fireEvent.click(screen.getByText(server.name))
  await screen.findByText('Could not open Fixture server: Synthetic IPC failure')
  expect(screen.queryByText('Connecting...')).toBeNull()
  expect(onConnect).not.toHaveBeenCalled()
  expect(screen.getAllByRole('button', { name: 'Connect' })).toHaveLength(2)
})
