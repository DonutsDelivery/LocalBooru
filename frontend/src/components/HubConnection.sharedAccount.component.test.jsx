// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { getHubConnection, getRemoteConnections, getSharedHubAccount, reuseSharedHubAccount, authorizeHub } from '../api'
vi.mock('../api', () => ({ getHubConnection: vi.fn(), getRemoteConnections: vi.fn(), getSharedHubAccount: vi.fn(), reuseSharedHubAccount: vi.fn(), authorizeHub: vi.fn(), createHub: vi.fn(), previewHub: vi.fn() }))
import HubConnection from './HubConnection'
const hub = source => ({ transport_mode: 'relayed', bootstrap: { peer_id: 'a'.repeat(64), relay_url: 'https://relay.example' }, descriptor: { instance: { id: source, name: `Community ${source}`, origin: `https://${source}.example` }, access: { catalog: 'members', registration: 'closed', submission: 'members', moderation: 'required' }, authentication: { methods: ['bearer'] }, capabilities: { native_authorization: true } } })
const candidate = (id = 'donor', account = 'Test account') => ({ connection_id: id, source_id: 'a', community_name: 'Community a', account_display_name: account })
const connected = () => ({ connection_id: 'alias', trust_state: 'verified', has_credential: true, shared_account: true })
const deferred = () => { let resolve; const promise = new Promise(yes => { resolve = yes }); return { promise, resolve } }
const show = (sourceId = 'b') => render(<MemoryRouter><HubConnection sourceId={sourceId} /></MemoryRouter>)
beforeEach(() => {
  vi.resetAllMocks()
  getHubConnection.mockImplementation(async source => hub(source))
  getRemoteConnections.mockResolvedValue({ connections: [] })
  getSharedHubAccount.mockResolvedValue({ state: 'available', candidates: [candidate()] })
  reuseSharedHubAccount.mockResolvedValue({ state: 'attached', source_id: 'b', connection_id: 'alias' })
})
afterEach(cleanup)

// AC: @verified-shared-hub-account ac-2
test('legacy empty community offers deliberate existing-account reuse without browser login or automatic attachment', async () => {
  show()
  const reuse = await screen.findByRole('button', { name: 'Use existing account' })
  expect(reuse.disabled).toBe(false)
  expect(reuseSharedHubAccount).not.toHaveBeenCalled()
  getRemoteConnections.mockResolvedValue({ connections: [connected()] })
  fireEvent.click(reuse)
  await waitFor(() => expect(reuseSharedHubAccount).toHaveBeenCalledWith('b', 'donor'))
  await screen.findByText('Account connected')
  await screen.findByText('This community uses your existing connected account.')
  expect(authorizeHub).not.toHaveBeenCalled()
})

// AC: @verified-shared-hub-account ac-6
test('ambiguous connected accounts require a selection before reuse', async () => {
  getSharedHubAccount.mockResolvedValue({ state: 'selection_required', candidates: [candidate('one', 'One'), candidate('two', 'Two')] })
  show()
  const reuse = await screen.findByRole('button', { name: 'Use existing account' })
  expect(reuse.disabled).toBe(true)
  fireEvent.click(reuse)
  expect(reuseSharedHubAccount).not.toHaveBeenCalled()
  fireEvent.change(screen.getByRole('combobox', { name: 'Existing account' }), { target: { value: 'two' } })
  getRemoteConnections.mockResolvedValue({ connections: [connected()] })
  fireEvent.click(reuse)
  await waitFor(() => expect(reuseSharedHubAccount).toHaveBeenCalledWith('b', 'two'))
})

// AC: @verified-shared-hub-account ac-6
test('explicit expired account is preserved and never offered replacement through shared sign-in', async () => {
  getRemoteConnections.mockResolvedValue({ connections: [{ connection_id: 'explicit', trust_state: 'expired', has_credential: false, expires_at: 1 }] })
  show()
  await screen.findByText('Sign in required to browse this community.')
  expect(getSharedHubAccount).not.toHaveBeenCalled()
  expect(screen.queryByRole('button', { name: 'Use existing account' })).toBeNull()
  expect(reuseSharedHubAccount).not.toHaveBeenCalled()
})

// AC: @verified-shared-hub-account ac-4
test('rejected catalog verification reports failure without claiming an attached account', async () => {
  reuseSharedHubAccount.mockRejectedValue({ response: { status: 403, data: { detail: 'Membership denied by this community' } } })
  show()
  fireEvent.click(await screen.findByRole('button', { name: 'Use existing account' }))
  await screen.findByText('Membership denied by this community')
  expect(screen.queryByText('Account connected')).toBeNull()
  expect(authorizeHub).not.toHaveBeenCalled()
})

// AC: @verified-shared-hub-account ac-5
test('late candidate discovery for a former source cannot replace the current community choice', async () => {
  const old = deferred()
  getSharedHubAccount.mockImplementation(source => source === 'b' ? old.promise : Promise.resolve({ state: 'unavailable', candidates: [] }))
  const view = show('b')
  await waitFor(() => expect(getSharedHubAccount).toHaveBeenCalledWith('b'))
  view.rerender(<MemoryRouter><HubConnection sourceId="c" /></MemoryRouter>)
  await screen.findByRole('heading', { name: 'Community c' })
  await act(async () => old.resolve({ state: 'available', candidates: [candidate()] }))
  expect(screen.queryByRole('button', { name: 'Use existing account' })).toBeNull()
  expect(reuseSharedHubAccount).not.toHaveBeenCalled()
})

// AC: @verified-shared-hub-account ac-5
test('switching source during reuse leaves the next source usable and ignores the old completion', async () => {
  const pending = deferred()
  reuseSharedHubAccount.mockReturnValue(pending.promise)
  const view = show('b')
  fireEvent.click(await screen.findByRole('button', { name: 'Use existing account' }))
  await waitFor(() => expect(reuseSharedHubAccount).toHaveBeenCalledWith('b', 'donor'))
  view.rerender(<MemoryRouter><HubConnection sourceId="c" /></MemoryRouter>)
  const reuse = await screen.findByRole('button', { name: 'Use existing account' })
  expect(reuse.disabled).toBe(false)
  await act(async () => pending.resolve({ state: 'attached', source_id: 'b' }))
  expect(screen.queryByText('Account connected')).toBeNull()
  expect(screen.getByRole('heading', { name: 'Community c' })).toBeTruthy()
  expect(reuse.disabled).toBe(false)
})
