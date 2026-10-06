// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
const api = vi.hoisted(() => ({ getConnectedHubs: vi.fn(), fetchCollections: vi.fn(), fetchDirectories: vi.fn(), getSavedSearches: vi.fn(), createCollection: vi.fn(), deleteCollection: vi.fn(), getMediaUrl: path => path, getFileDimensions: vi.fn(), createSavedSearch: vi.fn(), deleteSavedSearch: vi.fn() }))
vi.mock('../api', () => api)
vi.mock('../components/FamilyModeLock', () => ({ default: () => null }))
vi.mock('../components/Music/PersistentMusicPlayer', () => ({ default: () => null }))
import CollectionsPage from './CollectionsPage'
beforeEach(() => {
  vi.resetAllMocks(); localStorage.clear(); sessionStorage.clear()
  api.getConnectedHubs.mockResolvedValue({ hubs: [{ source_id: 'main', display_name: 'DonutBooru', enabled: true, transport_mode: 'relayed' }] })
  api.fetchCollections.mockResolvedValue({ collections: [{ id: 7, name: 'Local images', item_count: 0 }] })
  api.fetchDirectories.mockResolvedValue({ directories: [] })
  api.getSavedSearches.mockResolvedValue({ searches: [] })
})
afterEach(cleanup)
// AC: @collections ac-collection-sidebar
test('collection index puts shared navigation in the sidebar without a separate main-pane card strip', async () => {
  render(<MemoryRouter initialEntries={['/collections?media_type=image']}><CollectionsPage /></MemoryRouter>)
  const shared = await screen.findByRole('link', { name: 'DonutBooru', exact: true })
  expect(shared.closest('aside')).not.toBeNull()
  expect(shared.getAttribute('href')).toBe('/shared/main?media_type=image')
  expect(within(document.querySelector('main')).queryByRole('region', { name: 'Shared collections' })).toBeNull()
  expect(within(document.querySelector('aside')).getByRole('link', { name: /Local images/ })).toBeTruthy()
  expect((await within(document.querySelector('main')).findByText('Local images')).closest('[role=button]')).not.toBeNull()
  expect(api.getConnectedHubs).toHaveBeenCalledOnce()
})
