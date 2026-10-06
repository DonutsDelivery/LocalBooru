// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { getRemoteCreators, getRemoteOffline, removeRemoteOffline, updateRemoteCreatorPreferences, updateRemoteOffline } from '../api'
vi.mock('../api', () => ({ getRemoteCreators: vi.fn(), getRemoteOffline: vi.fn(), removeRemoteOffline: vi.fn(), updateRemoteCreatorPreferences: vi.fn(), updateRemoteOffline: vi.fn() }))
import RemoteCreatorCollections from './RemoteCreatorCollections'
const GiB = 1024 ** 3
const creator = (id = 'alice', name = 'Alice') => ({ id, name, item_count: 2, availability: 'available', following: false, offline: { enabled: false, state: 'disabled', downloaded_items: 0, total_items: 2, bytes: 0, error: null } })
const offline = () => ({ enabled: false, sync_all_creators: false, storage_limit_bytes: 0, downloaded_bytes: 0, state: 'disabled', downloaded_items: 0, total_items: 0, peer_hosting_enabled: false, peer_hosting_state: 'disabled' })
let row, settings
const directory = (rows = [row], page = 1, has_more = false) => ({ creators: rows, page, has_more })
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }
const expandSettings = () => fireEvent.click(screen.getByText('Offline sync and hosting'))
const props = () => ({ sourceId: 'one', mediaType: 'image', onDirectoryChange: vi.fn(), onSelectCreator: vi.fn() })
beforeEach(() => {
  vi.resetAllMocks(); row = creator(); settings = offline()
  getRemoteCreators.mockImplementation(async () => directory())
  getRemoteOffline.mockImplementation(async () => settings)
  updateRemoteCreatorPreferences.mockImplementation(async (_source, _id, preferences) => { row = { ...row, following: preferences.following ?? row.following, offline: { ...row.offline, enabled: preferences.offline_enabled ?? row.offline.enabled, state: preferences.offline_enabled ? 'queued' : row.offline.state } }; return { creator: row, offline: settings } })
  updateRemoteOffline.mockImplementation(async (_source, values) => { settings = { ...settings, ...values }; return settings })
  removeRemoteOffline.mockImplementation(async () => { settings = offline(); return settings })
})
afterEach(() => { cleanup(); vi.useRealTimers() })

// AC: @creator-community-replication ac-4
test('removing retained copies requires the explicit action and preserves following choices', async () => {
  settings = { ...offline(), downloaded_bytes: GiB, downloaded_items: 2 }
  row = { ...creator(), following: true }
  render(<RemoteCreatorCollections {...props()} />)
  expandSettings()
  fireEvent.click(await screen.findByRole('button', { name: 'Remove offline copies' }))
  expect(screen.getByRole('dialog').textContent).toContain('following choices')
  expect(removeRemoteOffline).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(removeRemoteOffline).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Remove offline copies' }))
  fireEvent.click(screen.getByRole('button', { name: 'Remove copies' }))
  await waitFor(() => expect(removeRemoteOffline).toHaveBeenCalledWith('one'))
  await screen.findByRole('button', { name: 'Unfollow Alice' })
  expect(updateRemoteCreatorPreferences).not.toHaveBeenCalled()
})

// AC: @creator-community-replication ac-2
test('following is a separate preference and starts no downloads', async () => {
  render(<RemoteCreatorCollections {...props()} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Follow Alice' }))
  await screen.findByRole('button', { name: 'Unfollow Alice' })
  expect(updateRemoteCreatorPreferences).toHaveBeenCalledWith('one', 'alice', { following: true })
  expect(updateRemoteOffline).not.toHaveBeenCalled()
  expect(screen.queryByRole('dialog')).toBeNull()
})

// AC: @creator-community-replication ac-3
test('individual offline sync starts only from reviewed storage controls and uses an atomic cap', async () => {
  render(<RemoteCreatorCollections {...props()} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Keep available offline Alice' }))
  expect(updateRemoteCreatorPreferences).not.toHaveBeenCalled()
  expect(screen.getByRole('spinbutton', { name: 'Community storage limit (GiB)' }).value).toBe('5')
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '2.5' } })
  fireEvent.click(screen.getByRole('button', { name: 'Start downloads' }))
  await screen.findByRole('button', { name: 'Stop offline sync Alice' })
  expect(updateRemoteCreatorPreferences).toHaveBeenCalledWith('one', 'alice', { offline_enabled: true, storage_limit_bytes: 2.5 * GiB, confirmed_download: true })
  expect(row.following).toBe(false)
})

// AC: @creator-community-replication ac-3
test('sync all includes future creators and requires the chosen budget on the actual action', async () => {
  render(<RemoteCreatorCollections {...props()} />)
  expandSettings()
  await waitFor(() => expect(screen.getByRole('button', { name: 'Sync all creators' }).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Sync all creators' }))
  expect(screen.getByText(/including future creators/)).toBeTruthy()
  expect(updateRemoteOffline).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Start downloads' }))
  await waitFor(() => expect(updateRemoteOffline).toHaveBeenCalledWith('one', { sync_all_creators: true, storage_limit_bytes: 5 * GiB, confirmed_download: true }))
  await screen.findByRole('button', { name: 'Stop syncing all creators' })
})

// AC: @creator-community-replication ac-3
// AC: @creator-community-replication ac-5
test('peer hosting requires its own explicit action and renders backend status', async () => {
  updateRemoteOffline.mockResolvedValue({ ...offline(), peer_hosting_enabled: true, peer_hosting_state: 'hosting' })
  render(<RemoteCreatorCollections {...props()} />)
  expandSettings()
  fireEvent.click(await screen.findByRole('button', { name: 'Host local copies for peers' }))
  expect(screen.getByText(/Credentials and personal library files stay private/)).toBeTruthy()
  expect(updateRemoteOffline).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Start hosting' }))
  await screen.findByText('Hosting local copies')
  expect(updateRemoteOffline).toHaveBeenCalledWith('one', { peer_hosting_enabled: true, confirmed_hosting: true })
  expect(updateRemoteCreatorPreferences).not.toHaveBeenCalled()
})

// AC: @creator-community-replication ac-1
test('creator directory filters are server scoped and late previous lists are discarded', async () => {
  const old = deferred()
  getRemoteCreators.mockReturnValueOnce(old.promise).mockResolvedValue(directory([creator('bob', 'Bob')]))
  const callbacks = props()
  const rendered = render(<RemoteCreatorCollections {...callbacks} />)
  fireEvent.click(screen.getByRole('button', { name: 'Following', exact: true }))
  expect(callbacks.onDirectoryChange).toHaveBeenCalledWith('following', '')
  rendered.rerender(<RemoteCreatorCollections {...callbacks} view="following" query="bob" mediaType="video" />)
  await screen.findByRole('button', { name: 'Bob 2 videos' })
  await act(async () => old.resolve(directory([creator()])))
  expect(screen.queryByRole('button', { name: 'Alice 2 videos' })).toBeNull()
  expect(getRemoteCreators).toHaveBeenLastCalledWith('one', { query: 'bob', media_type: 'video', view: 'following', page: 1, per_page: 30 })
  expect(callbacks.onDirectoryChange).toHaveBeenCalledTimes(1)
})

test('an old source preference failure cannot appear on the new source', async () => {
  const old = deferred()
  updateRemoteCreatorPreferences.mockReturnValueOnce(old.promise)
  const callbacks = props()
  const rendered = render(<RemoteCreatorCollections {...callbacks} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Follow Alice' }))
  row = creator('bob', 'Bob')
  rendered.rerender(<RemoteCreatorCollections {...callbacks} sourceId="two" />)
  await screen.findByRole('button', { name: 'Follow Bob' })
  await act(async () => old.reject(new Error('Previous source failed')))
  expect(screen.queryByText('Previous source failed')).toBeNull()
  expect(screen.getByRole('button', { name: 'Follow Bob' }).disabled).toBe(false)
})

test('the storage dialog supports keyboard cancel and restores the triggering control', async () => {
  render(<RemoteCreatorCollections {...props()} />)
  const trigger = await screen.findByRole('button', { name: 'Keep available offline Alice' })
  trigger.focus(); fireEvent.click(trigger)
  expect(document.activeElement).toBe(screen.getByRole('spinbutton'))
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(document.activeElement).toBe(trigger)
  expect(updateRemoteCreatorPreferences).not.toHaveBeenCalled()
})

// AC: @creator-community-replication ac-3
// AC: @creator-community-replication ac-6
test('partial downloads and storage failure display real progress without claiming readiness', async () => {
  row = { ...creator(), availability: 'offline', offline: { enabled: true, state: 'budget_exceeded', downloaded_items: 1, total_items: 2, bytes: GiB, error: 'Not enough budget' } }
  render(<RemoteCreatorCollections {...props()} view="offline" localCopies />)
  await screen.findByText('Storage limit reached · 1/2 items · 1.00 GiB')
  expect(screen.getByText('Community unavailable')).toBeTruthy()
  expect(screen.getByText('Not enough budget')).toBeTruthy()
  expect(screen.getByText('Local copies')).toBeTruthy()
  expect(screen.queryByText('Available offline')).toBeNull()
})

test('status polls preserve expanded creator pages and refresh current query and view', async () => {
  vi.useFakeTimers()
  getRemoteCreators.mockImplementation(async (_source, params) => directory([params.page === 1 ? creator() : creator('bob', 'Bob')], params.page, params.page === 1))
  const callbacks = props()
  render(<RemoteCreatorCollections {...callbacks} view="all" query="active" />)
  await act(async () => { await vi.advanceTimersByTimeAsync(0) })
  fireEvent.click(screen.getByRole('button', { name: 'More creators' }))
  await act(async () => { await vi.advanceTimersByTimeAsync(0) })
  expect(screen.getByRole('button', { name: 'Bob 2 images' })).toBeTruthy()
  await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
  expect(screen.getByRole('button', { name: 'Alice 2 images' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Bob 2 images' })).toBeTruthy()
  expect(getRemoteCreators).toHaveBeenLastCalledWith('one', { query: 'active', media_type: 'image', view: 'all', page: 2, per_page: 30 })
})

// AC: @creator-community-replication ac-1
test('creator search debounces typed names and does not send media tag filters as directory search', async () => {
  vi.useFakeTimers()
  const callbacks = props()
  render(<RemoteCreatorCollections {...callbacks} view="following" />)
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search creators' }), { target: { value: 'ali' } })
  expect(callbacks.onDirectoryChange).not.toHaveBeenCalled()
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  expect(callbacks.onDirectoryChange).toHaveBeenCalledWith('following', 'ali')
  expect(getRemoteCreators).toHaveBeenCalledWith('one', { query: '', media_type: 'image', view: 'following', page: 1, per_page: 30 })
})

// AC: @creator-community-replication ac-3
test('invalid storage cap cannot start downloads and cancel does not enable sync', async () => {
  render(<RemoteCreatorCollections {...props()} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Keep available offline Alice' }))
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '0' } })
  expect(screen.getByRole('button', { name: 'Start downloads' }).disabled).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(updateRemoteCreatorPreferences).not.toHaveBeenCalled()
  expect(updateRemoteOffline).not.toHaveBeenCalled()
})

test('storage review traps keyboard focus in both directions', async () => {
  render(<RemoteCreatorCollections {...props()} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Keep available offline Alice' }))
  const input = screen.getByRole('spinbutton')
  const start = screen.getByRole('button', { name: 'Start downloads' })
  expect(document.activeElement).toBe(input)
  fireEvent.keyDown(input, { key: 'Tab', shiftKey: true })
  expect(document.activeElement).toBe(start)
  fireEvent.keyDown(start, { key: 'Tab' })
  expect(document.activeElement).toBe(input)
})

// AC: @creator-community-replication ac-6
test.each([
  [{ central_readiness: 'partial', hub_hosted: true }, 'Partially available'],
  [{ availability: 'partial' }, 'Partially available'],
  [{ central_readiness: 'ready', hub_hosted: true }, 'Hosted by community'],
  [{ central_readiness: 'unavailable', hub_hosted: false, peer_availability: 'recently_reported' }, 'Recently shared · reachability unknown'],
  [{ central_readiness: 'unavailable', hub_hosted: false, provider_availability: { peer_status: 'recently_reported' } }, 'Recently shared · reachability unknown'],
])('availability remains honest for %j', async (availability, label) => {
  row = { ...creator(), ...availability }
  render(<RemoteCreatorCollections {...props()} view="all" />)
  await screen.findByText(label)
  expect(screen.getByRole('button', { name: 'Show all' })).toBeTruthy()
})

// AC: @creator-community-replication ac-2
// AC: @creator-community-replication ac-6
test('followed creators remain selectable when their community is unavailable without enabling downloads', async () => {
  row = { ...creator(), availability: 'offline', following: true }
  const callbacks = props()
  render(<RemoteCreatorCollections {...callbacks} view="following" />)
  fireEvent.click(await screen.findByRole('button', { name: 'Alice 2 images' }))
  expect(callbacks.onSelectCreator).toHaveBeenCalledWith('alice', false)
  expect(screen.getByRole('button', { name: 'Unfollow Alice' })).toBeTruthy()
  expect(screen.getByText('Community unavailable')).toBeTruthy()
  expect(updateRemoteCreatorPreferences).not.toHaveBeenCalled()
  expect(updateRemoteOffline).not.toHaveBeenCalled()
})
