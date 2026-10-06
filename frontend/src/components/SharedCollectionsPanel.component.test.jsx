// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { getConnectedHubs } from '../api'
import SharedCollectionsPanel from './SharedCollectionsPanel'

vi.mock('../api', () => ({ getConnectedHubs: vi.fn() }))
afterEach(() => { cleanup(); vi.resetAllMocks() })

// AC: @creator-community-replication ac-4
test('disabled connections retain a discoverable personal offline gallery in Collections', async () => {
  getConnectedHubs.mockResolvedValue({ hubs: [
    { source_id: 'disabled/source', display_name: 'Offline community', enabled: false },
    { source_id: 'live', display_name: 'Live community', enabled: true, transport_mode: 'relayed' },
  ] })
  render(<MemoryRouter><SharedCollectionsPanel mediaType="video" galleryReturnTo="/videos" /></MemoryRouter>)
  const offline = await screen.findByRole('link', { name: /Offline community.*Browse offline copies/ })
  expect(offline.getAttribute('href')).toBe('/shared/disabled%2Fsource?media_type=video&creator_view=offline&offline_only=true')
  expect(screen.getByRole('link', { name: /Live community/ }).getAttribute('href')).toBe('/shared/live?media_type=video')
})
