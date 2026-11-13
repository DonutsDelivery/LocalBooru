import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { CapabilityGate, EditionProvider } from './EditionContext'
import { addonAvailableInEdition } from './edition'
import SharedCollectionsPanel from '../components/SharedCollectionsPanel'
import HubLinkHandler from '../components/HubLinkHandler'

const mocks = vi.hoisted(() => ({ hubs: vi.fn(), authorize: vi.fn(), listen: vi.fn(), invoke: vi.fn(), error: vi.fn() }))
vi.mock('../api', () => ({ getConnectedHubs: mocks.hubs, completeHubAuthorization: mocks.authorize }))
vi.mock('../components/Toast', () => ({ toast: { error: mocks.error } }))
vi.mock('@tauri-apps/api/core', () => ({ invoke: mocks.invoke }))
vi.mock('@tauri-apps/api/event', () => ({ listen: mocks.listen }))

const lite = Object.freeze({ edition: 'store-lite', community_hubs: false, downloaded_executable_addons: false, lan_library: true, self_updates: false })
beforeEach(() => { vi.resetAllMocks(); window.__DMC_EDITION_CAPABILITIES__ = lite })
afterEach(() => { cleanup(); delete window.__DMC_EDITION_CAPABILITIES__; delete window.__TAURI_INTERNALS__; delete window.__dmcPendingHubLink; vi.unstubAllGlobals() })

test('native profile wins without fetching capabilities from a paired full server', () => {
  const fetch = vi.fn(); vi.stubGlobal('fetch', fetch)
  render(<EditionProvider><CapabilityGate capability="community_hubs"><span>Communities</span></CapabilityGate><CapabilityGate capability="lan_library"><span>Paired library</span></CapabilityGate></EditionProvider>)
  expect(screen.queryByText('Communities')).toBeNull()
  expect(screen.getByText('Paired library')).toBeTruthy()
  expect(fetch).not.toHaveBeenCalled()
})

test('lite shared sidebar never loads or exposes persisted communities', () => {
  render(<MemoryRouter><EditionProvider><SharedCollectionsPanel mediaType="image" /></EditionProvider></MemoryRouter>)
  expect(mocks.hubs).not.toHaveBeenCalled()
  expect(screen.queryByRole('region', { name: 'Shared collections' })).toBeNull()
})

test('hub and OAuth deep links cannot start enrollment or complete authorization in lite', async () => {
  window.__TAURI_INTERNALS__ = {}
  mocks.listen.mockResolvedValue(() => {})
  mocks.invoke.mockResolvedValue(['donutmediacenter://hub/add?url=https%3A%2F%2Fsynthetic.invalid', 'donutmediacenter://auth/callback?state=synthetic&code=synthetic'])
  render(<MemoryRouter><EditionProvider><HubLinkHandler /></EditionProvider></MemoryRouter>)
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2))
  expect(mocks.authorize).not.toHaveBeenCalled()
  expect(window.__dmcPendingHubLink).toBeUndefined()
})

test('browser discovery fails closed for optional features while retaining library browsing', async () => {
  delete window.__DMC_EDITION_CAPABILITIES__
  const fetch = vi.fn().mockRejectedValue(new Error('offline')); vi.stubGlobal('fetch', fetch)
  render(<EditionProvider><CapabilityGate capability="downloaded_executable_addons"><span>Executable installer</span></CapabilityGate><CapabilityGate capability="lan_library"><span>Library</span></CapabilityGate></EditionProvider>)
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/edition', expect.objectContaining({ credentials: 'same-origin' })))
  expect(screen.queryByText('Executable installer')).toBeNull()
  expect(screen.getByText('Library')).toBeTruthy()
})

test('lite hides executable tools reported by a full remote server and retains builtin curation', () => {
  expect(addonAvailableInEdition({ id: 'donut-create', installation: 'python_venv' }, lite)).toBe(false)
  expect(addonAvailableInEdition({ id: 'curation-game', installation: 'builtin' }, lite)).toBe(true)
  expect(addonAvailableInEdition({ id: 'donut-create' }, { downloaded_executable_addons: true })).toBe(true)
})
