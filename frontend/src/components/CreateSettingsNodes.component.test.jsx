import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import CreateSettings from './CreateSettings'
import { getAddon } from '../api'
import { getCreateStatus, updateCreateNodes, restartCreateBackend } from '../services/donutCreate'
vi.mock('../api', () => ({ getAddon: vi.fn(), installAddon: vi.fn(), startAddon: vi.fn() }))
vi.mock('../services/donutCreate', () => ({
  getCreateStatus: vi.fn(), updateCreateNodes: vi.fn(), restartCreateBackend: vi.fn(),
  startCreateBackend: vi.fn(), stopCreateBackend: vi.fn(), saveCreateConfig: vi.fn(),
  setupCreateBackend: vi.fn(), cancelCreateSetup: vi.fn(),
  createErrorMessage: error => error.message, formatCreateBytes: bytes => `${bytes} bytes`,
}))
vi.mock('./Toast', () => ({ toast: { error: vi.fn() } }))
let status
beforeEach(() => {
  vi.clearAllMocks()
  status = { mode: 'local', backend_url: 'http://127.0.0.1:8188',
    backend: { ready: false, running: false, owned: false,
      face_node: { registered: false, updateable: true, restart_needed: false } }, setup: {} }
  getAddon.mockResolvedValue({ addon: { installed: true, status: 'running' } })
  getCreateStatus.mockImplementation(async () => status)
  updateCreateNodes.mockImplementation(async () => {
    status = { ...status, setup: { operation: 'nodes', running: true, state: 'installing' } }
    return status
  })
})
afterEach(cleanup)
// AC: @create-multiple-face-prompts ac-existing-backend-node
test('updates DonutNodes in the stopped selected backend without silently restarting it', async () => {
  const onStatusChange = vi.fn()
  render(<CreateSettings backendOnly onStatusChange={onStatusChange} />)
  const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
  summary.closest('details').open = true
  const update = await screen.findByRole('button', { name: 'Update DonutNodes' })
  await waitFor(() => expect(update.disabled).toBe(false))
  fireEvent.click(update)
  await waitFor(() => expect(updateCreateNodes).toHaveBeenCalledTimes(1))
  expect(updateCreateNodes).toHaveBeenCalledWith(expect.any(AbortSignal), expect.any(Function))
  expect(await screen.findByRole('button', { name: 'Updating DonutNodes…' })).toBeTruthy()
  expect(restartCreateBackend).not.toHaveBeenCalled()
  expect(onStatusChange).toHaveBeenCalledWith(status)
})
// AC: @create-multiple-face-prompts ac-existing-backend-node
test('requires stopping the connected backend before source updates', async () => {
  status.backend = { ...status.backend, running: true, owned: true }
  render(<CreateSettings backendOnly />)
  const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
  summary.closest('details').open = true
  const update = await screen.findByRole('button', { name: 'Update DonutNodes' })
  expect(update.disabled).toBe(true)
  expect(screen.getByText(/Stop ComfyUI before updating DonutNodes/)).toBeTruthy()
  fireEvent.click(update)
  expect(updateCreateNodes).not.toHaveBeenCalled()
})
// AC: @create-multiple-face-prompts ac-existing-backend-node
test('keeps retry available after a protected source update failure', async () => {
  updateCreateNodes.mockRejectedValueOnce(new Error('Modified node source was preserved.'))
  render(<CreateSettings backendOnly />)
  const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
  summary.closest('details').open = true
  const update = await screen.findByRole('button', { name: 'Update DonutNodes' })
  await waitFor(() => expect(update.disabled).toBe(false))
  fireEvent.click(update)
  expect(await screen.findByText('Modified node source was preserved.')).toBeTruthy()
  await waitFor(() => expect(update.disabled).toBe(false))
  expect(restartCreateBackend).not.toHaveBeenCalled()
})
// AC: @create-multiple-face-prompts ac-existing-backend-node
test('shows terminal worker failure after the update operation has finished', async () => {
  status.setup = { operation: null, running: false, node_update: { state: 'error' } }
  status.backend.face_node.error = 'DonutNodes has local changes. They were preserved.'
  render(<CreateSettings backendOnly />)
  const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
  summary.closest('details').open = true
  expect(await screen.findByText('DonutNodes has local changes. They were preserved.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Update DonutNodes' }).disabled).toBe(false)
})
// AC: @create-multiple-face-prompts ac-existing-backend-node
test.each(['http://127.0.0.1:8188', 'https://synthetic-backend.test'])('does not offer filesystem updates for URL-only %s', async backend_url => {
  status = { ...status, mode: 'existing', backend_url,
    backend: { ...status.backend, running: true,
      face_node: { registered: false, updateable: false, restart_needed: false } } }
  render(<CreateSettings backendOnly />)
  const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
  summary.closest('details').open = true
  await screen.findByRole('button', { name: 'Start ComfyUI' })
  expect(screen.queryByRole('button', { name: 'Update DonutNodes' })).toBeNull()
  expect(screen.getByText(/Update DonutNodes on the backend host/)).toBeTruthy()
  expect(updateCreateNodes).not.toHaveBeenCalled()
})
