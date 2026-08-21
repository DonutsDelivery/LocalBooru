import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import OnlinePage from './OnlinePage'
import * as api from '../api'

globalThis.IS_REACT_ACT_ENVIRONMENT = true

vi.mock('../api', () => ({
  browseRemoteSource: vi.fn(),
  createPublication: vi.fn(),
  createRemoteConnection: vi.fn(),
  createRemoteSource: vi.fn(),
  deleteRemoteConnection: vi.fn(),
  deleteRemoteSource: vi.fn(),
  fetchDirectories: vi.fn(),
  getPublications: vi.fn(),
  getRemoteConnections: vi.fn(),
  getRemoteSources: vi.fn(),
  importRemoteItem: vi.fn(),
  probeRemoteSource: vi.fn(),
  simulatePublication: vi.fn(),
}))

describe('OnlinePage remote/local boundary', () => {
  let container
  let root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    api.getRemoteSources.mockResolvedValue({ sources: [{
      source_id: 'remote-a',
      display_name: 'Fixture Booru',
      provider_family: 'donutbooru',
      enabled: true,
      policy_profile: 'safe',
      capabilities: { browse: true, search: true, upload: false },
    }] })
    api.fetchDirectories.mockResolvedValue({ directories: [{ id: 9, name: 'Imports', path: '/synthetic/imports' }] })
    api.getPublications.mockResolvedValue({ publications: [] })
    api.getRemoteConnections.mockResolvedValue({ connections: [] })
    api.browseRemoteSource.mockResolvedValue({ stale: false, items: [{
      source_id: 'remote-a',
      remote_post_id: '42',
      canonical_url: 'https://fixture.test/images/42',
      title: null,
      tags: ['synthetic', 'safe'],
      media: [{ kind: 'thumbnail' }, { kind: 'original' }],
    }] })
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.clearAllMocks()
  })

  it('labels remote media and exposes import without local mutation controls', async () => {
    await act(async () => {
      root.render(<MemoryRouter initialEntries={['/online']}><OnlinePage /></MemoryRouter>)
      await Promise.resolve()
    })

    const form = container.querySelector('.online-toolbar')
    await act(async () => {
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
      await Promise.resolve()
    })
    const card = container.querySelector('.remote-card')
    expect(card).not.toBeNull()
    expect(card.textContent).toContain('Remote · Fixture Booru')
    for (const forbidden of ['Favorite', 'Delete', 'Move', 'Retag', 'Collection']) {
      expect(card.textContent).not.toContain(forbidden)
    }

    await act(async () => card.dispatchEvent(new MouseEvent('click', { bubbles: true })))
    expect(container.querySelector('[role="dialog"]').textContent).toContain('Import a local copy')
    expect(container.querySelector('[role="dialog"]').textContent).toContain('Browsing does not add this item')
  })

  it('collects account secrets only in a password field and never renders stored secrets', async () => {
    await act(async () => {
      root.render(<MemoryRouter initialEntries={['/online']}><OnlinePage /></MemoryRouter>)
      await Promise.resolve()
    })
    const sourcesTab = [...container.querySelectorAll('nav button')].find(button => button.textContent === 'Sources')
    await act(async () => sourcesTab.click())
    expect(container.querySelector('input[type="password"]')).not.toBeNull()
    expect(container.textContent).toContain('never returned to this page')
  })
})
