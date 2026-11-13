import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const { desktop, api, notifications } = vi.hoisted(() => ({
  desktop: { getFolderAccessStatus: vi.fn(), addDirectory: vi.fn() },
  api: {
    fetchDirectories: vi.fn(), fetchLibraries: vi.fn(), listParentDirectories: vi.fn(),
    addDirectory: vi.fn(), addLibrary: vi.fn(), removeLibrary: vi.fn(),
    updateDirectory: vi.fn(), tagUntagged: vi.fn(), clearDirectoryTagQueue: vi.fn(),
    mountLibrary: vi.fn(), unmountLibrary: vi.fn(), removeParentDirectory: vi.fn(),
    subscribeToLibraryEvents: vi.fn(() => () => {}),
  },
  notifications: { error: vi.fn(), success: vi.fn(), warning: vi.fn() },
}))
vi.mock('../api', () => api)
vi.mock('../tauriAPI', () => ({ getDesktopAPI: () => desktop, isDesktopApp: () => true }))
vi.mock('../components/Toast', () => ({ toast: notifications }))
vi.mock('../hooks/useAddonStatus', () => ({ useAddonStatus: () => ({ installed: false }) }))
vi.mock('../components/ComfyUIConfigModal', () => ({ default: () => null }))

import DirectoriesPage from './DirectoriesPage'

beforeEach(() => {
  vi.clearAllMocks()
  api.fetchDirectories.mockResolvedValue({ directories: [] })
  api.fetchLibraries.mockResolvedValue({ libraries: [{ uuid: 'fixture', is_primary: true, mounted: true, name: 'Fixture' }] })
  api.listParentDirectories.mockResolvedValue({ parents: [] })
  desktop.getFolderAccessStatus.mockResolvedValue({ required: true, failed: 1, storage_error: false })
})
afterEach(cleanup)

test('reauthorization clears recovery without adding or removing library references', async () => {
  desktop.addDirectory.mockResolvedValue('/synthetic/Library æ')
  desktop.getFolderAccessStatus
    .mockResolvedValueOnce({ required: true, failed: 1, storage_error: false })
    .mockResolvedValue({ required: true, failed: 0, storage_error: false })
  render(<DirectoriesPage />)
  fireEvent.click(await screen.findByRole('button', { name: 'Restore folder access' }))
  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  expect(notifications.success).toHaveBeenCalledWith('Folder access saved for future launches')
  expect(api.addDirectory).not.toHaveBeenCalled()
  expect(api.removeLibrary).not.toHaveBeenCalled()
})

test('a failed grant save stays visible and leaves recovery available', async () => {
  desktop.addDirectory.mockRejectedValue(new Error('Folder permission could not be saved'))
  render(<DirectoriesPage />)
  fireEvent.click(await screen.findByRole('button', { name: 'Restore folder access' }))
  await waitFor(() => expect(notifications.error).toHaveBeenCalledWith('Folder permission could not be saved'))
  expect(screen.getByRole('alert').textContent).toContain('library references are preserved')
  expect(api.removeLibrary).not.toHaveBeenCalled()
})

test('full desktop status does not display sandbox recovery', async () => {
  desktop.getFolderAccessStatus.mockResolvedValue({ required: false, failed: 0, storage_error: false })
  render(<DirectoriesPage />)
  await waitFor(() => expect(api.fetchDirectories).toHaveBeenCalled())
  expect(screen.queryByRole('alert')).toBeNull()
})
