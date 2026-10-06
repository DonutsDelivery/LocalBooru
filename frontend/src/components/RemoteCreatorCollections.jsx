// SPDX-License-Identifier: GPL-3.0-only
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { getRemoteCreators, getRemoteOffline, removeRemoteOffline, updateRemoteCreatorPreferences, updateRemoteOffline } from '../api'
import useDebounce from './Sidebar/hooks/useDebounce'
import './RemoteCreatorCollections.css'

const GiB = 1024 ** 3
const views = [['available', 'Available'], ['following', 'Following'], ['offline', 'Offline'], ['all', 'Show all']]
const errorMessage = error => error?.response?.data?.detail || error?.response?.data?.error || error?.message || 'Could not load creator collections.'
const bytesLabel = value => `${((Number(value) || 0) / GiB).toFixed(2)} GiB`
const availabilityLabel = creator => {
  if (creator.availability === 'withdrawn') return 'Withdrawn from community'
  if (creator.availability === 'partial' || creator.central_readiness === 'partial') return 'Partially available'
  if (creator.central_readiness === 'ready' || creator.hub_hosted === true) return 'Hosted by community'
  if (creator.peer_availability === 'recently_reported' || creator.provider_availability?.peer_status === 'recently_reported') return 'Recently shared · reachability unknown'
  if (creator.availability === 'offline') return 'Community unavailable'
  if (creator.availability === 'available' && creator.central_readiness !== 'unavailable') return 'Available from community'
  return 'Availability unknown'
}
const states = { disabled: 'Not kept offline', queued: 'Queued', syncing: 'Downloading', ready: 'Available offline', paused: 'Paused', budget_exceeded: 'Storage limit reached', auth_required: 'Sign in required', error: 'Download failed' }
const offlineLabel = offline => {
  if (!offline) return 'Offline status unavailable'
  const progress = Number(offline.total_items) > 0 ? ` · ${offline.downloaded_items || 0}/${offline.total_items} items` : ''
  return `${states[offline.state] || offline.state || 'Status unavailable'}${progress}`
}

export default function RemoteCreatorCollections({ sourceId, mediaType, selectedCreatorId, localCopies = false, view = 'available', query = '', onDirectoryChange, onSelectCreator }) {
  const [search, setSearch] = useState(query)
  const debouncedSearch = useDebounce(search, 200)
  const editingSearch = useRef(false)
  const scope = JSON.stringify([sourceId, mediaType, view, query])
  const activeScope = useRef(scope); activeScope.current = scope
  const sourceScope = `${sourceId}:${mediaType}`
  const activeSource = useRef(sourceScope); activeSource.current = sourceScope
  const generation = useRef(0)
  const busy = useRef(false)
  const mounted = useRef(true)
  const [directory, setDirectory] = useState({ scope, creators: [], page: 0, hasMore: false, loading: true, error: '' })
  const directoryRef = useRef(directory)
  const [settings, setSettings] = useState(null)
  const settingsGeneration = useRef(0)
  const [notice, setNotice] = useState('')
  const [mutating, setMutating] = useState(false)
  const mutationBusy = useRef(false)
  const [review, setReview] = useState(null)
  const [budget, setBudget] = useState('5')
  const dialogRef = useRef(null)
  const triggerRef = useRef(null)
  const current = directory.scope === scope ? directory : { creators: [], page: 0, hasMore: false, loading: true, error: '' }

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; generation.current += 1 } }, [])
  useEffect(() => { editingSearch.current = false; setSearch(query) }, [query, sourceId, mediaType])
  useEffect(() => { if (editingSearch.current && debouncedSearch === search && debouncedSearch !== query) onDirectoryChange(view, debouncedSearch) }, [debouncedSearch, search, query, view, onDirectoryChange])

  const load = useCallback(async (page = 1, append = false, refresh = false) => {
    if (busy.current) return
    busy.current = true
    const request = ++generation.current
    const previous = directoryRef.current.scope === scope ? directoryRef.current : { scope, creators: [], page: 0, hasMore: false }
    setDirectory({ ...previous, loading: true, error: '' })
    try {
      let data
      const refreshed = []
      const lastPage = refresh ? Math.max(1, previous.page) : page
      for (let nextPage = refresh ? 1 : page; nextPage <= lastPage; nextPage += 1) {
        data = await getRemoteCreators(sourceId, { query, media_type: mediaType, view, page: nextPage, per_page: 30 })
        if (!mounted.current || request !== generation.current || activeScope.current !== scope) return
        refreshed.push(...(data.creators || []))
        if (!data.has_more) break
      }
      const creators = refresh ? refreshed : append ? [...previous.creators, ...(data.creators || []).filter(creator => !previous.creators.some(existing => String(existing.id) === String(creator.id)))] : data.creators || []
      const next = { scope, creators, page: data.page || page, hasMore: Boolean(data.has_more), loading: false, error: '' }
      directoryRef.current = next; setDirectory(next)
    } catch (error) {
      if (mounted.current && request === generation.current && activeScope.current === scope) { const next = { ...previous, loading: false, error: errorMessage(error) }; directoryRef.current = next; setDirectory(next) }
    } finally { if (request === generation.current) busy.current = false }
  }, [sourceId, mediaType, query, view, scope])

  const loadRef = useRef(load); loadRef.current = load

  useEffect(() => {
    generation.current += 1; busy.current = false
    load()
    return () => { generation.current += 1; busy.current = false }
  }, [load])

  useEffect(() => {
    let active = true
    setSettings(null); setReview(null); setNotice(''); setMutating(false); mutationBusy.current = false
    const refresh = () => {
      const request = ++settingsGeneration.current
      return getRemoteOffline(sourceId).then(data => { if (active && request === settingsGeneration.current && !mutationBusy.current) setSettings(data) }).catch(error => { if (active && request === settingsGeneration.current && !mutationBusy.current) setNotice(errorMessage(error)) })
    }
    refresh()
    const timer = setInterval(() => { if (!mutationBusy.current) { refresh(); loadRef.current(1, false, true) } }, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [sourceId, mediaType]) // Directory fetches have their own query/view generation.

  useEffect(() => {
    if (!review) return
    dialogRef.current?.querySelector('input,button')?.focus()
    return () => triggerRef.current?.focus()
  }, [review])

  const openReview = action => {
    triggerRef.current = document.activeElement
    setBudget(String(settings?.storage_limit_bytes > 0 ? settings.storage_limit_bytes / GiB : 5))
    setNotice(''); setReview(action)
  }
  const mutate = async (action, preference = null) => {
    if (mutationBusy.current) return
    const requestSource = sourceScope
    const requestDirectory = scope
    settingsGeneration.current += 1
    mutationBusy.current = true; setMutating(true); setNotice('')
    try {
      if (preference) {
        const data = await updateRemoteCreatorPreferences(sourceId, action.id, preference)
        if (!mounted.current || activeSource.current !== requestSource) return
        if (data.offline) setSettings(data.offline)
        if (activeScope.current === requestDirectory && data.creator) {
          const next = { ...directoryRef.current, creators: directoryRef.current.creators.map(creator => String(creator.id) === String(action.id) ? data.creator : creator) }
          directoryRef.current = next; setDirectory(next)
        }
      } else {
        const data = action.remove_copies ? await removeRemoteOffline(sourceId) : await updateRemoteOffline(sourceId, action)
        if (!mounted.current || activeSource.current !== requestSource) return
        setSettings(data)
      }
      if (mounted.current && activeSource.current === requestSource) {
        setReview(null)
        if (activeScope.current === requestDirectory) { generation.current += 1; busy.current = false; load() }
      }
    } catch (error) { if (mounted.current && activeSource.current === requestSource) setNotice(errorMessage(error)) }
    finally { if (mounted.current && activeSource.current === requestSource) { mutationBusy.current = false; setMutating(false) } }
  }
  const cap = Number(budget) * GiB
  const validCap = Number.isFinite(cap) && cap > 0 && Number.isSafeInteger(Math.round(cap))
  const confirmReview = event => {
    event.preventDefault()
    if (review.kind === 'remove') { mutate({ remove_copies: true }); return }
    if (review.kind !== 'hosting' && !validCap) return
    const storage_limit_bytes = Math.round(cap)
    if (review.kind === 'creator') mutate(review.creator, { offline_enabled: true, storage_limit_bytes, confirmed_download: true })
    else if (review.kind === 'all') mutate({ sync_all_creators: true, storage_limit_bytes, confirmed_download: true })
    else if (review.kind === 'hosting') mutate({ peer_hosting_enabled: true, confirmed_hosting: true })
    else mutate({ storage_limit_bytes })
  }
  const trapDialog = event => {
    if (event.key === 'Escape' && !mutating) { event.stopPropagation(); setReview(null) }
    if (event.key !== 'Tab') return
    const focusable = [...dialogRef.current.querySelectorAll('input,button')].filter(element => !element.disabled)
    if (event.shiftKey && document.activeElement === focusable[0]) { event.preventDefault(); focusable.at(-1)?.focus() }
    else if (!event.shiftKey && document.activeElement === focusable.at(-1)) { event.preventDefault(); focusable[0]?.focus() }
  }

  return <section className="creator-collections" aria-label="Creator collections">
    <h4>Creator collections</h4><p className="creator-help">Collections published by community accounts.</p>
    <label className="creator-search-label">Search creators<input type="search" value={search} onChange={event => { editingSearch.current = true; setSearch(event.target.value) }} /></label>
    <div className="creator-views" role="group" aria-label="Creator collection views">{views.map(([value, label]) => <button key={value} type="button" aria-pressed={view === value} onClick={() => onDirectoryChange(value, search)}>{label}</button>)}</div>
    <button type="button" className="creator-select-all" aria-pressed={!selectedCreatorId && localCopies === (view === 'offline')} onClick={() => onSelectCreator(null, view === 'offline')}>{view === 'offline' ? 'All offline copies' : 'All creators'}</button>
    {localCopies && <p className="creator-local-status" role="status">Local copies</p>}
    {current.loading && <p role="status">Loading creators…</p>}
    {current.error && <p role="alert">{current.error} <button type="button" onClick={() => load()}>Retry creator list</button></p>}
    {!current.loading && !current.error && current.creators.length === 0 && <p className="creator-help">No creator collections in this view.</p>}
    <ul className="creator-list" tabIndex={0} aria-label="Creator collection list">{current.creators.map(creator => <li key={creator.id}>
      <button type="button" className="creator-select" aria-label={`${creator.name} ${creator.item_count ?? 0} ${mediaType === 'video' ? 'videos' : 'images'}`} aria-pressed={String(selectedCreatorId) === String(creator.id)} onClick={() => onSelectCreator(creator.id, view === 'offline')}><span>{creator.name}</span><span>{creator.item_count ?? 0} {mediaType === 'video' ? 'videos' : 'images'}</span></button>
      <p className="creator-availability">{availabilityLabel(creator)}</p>
      <p className="creator-offline-progress">{offlineLabel(creator.offline)}{creator.offline?.bytes > 0 ? ` · ${bytesLabel(creator.offline.bytes)}` : ''}</p>
      {creator.offline?.error && <p className="creator-error" role="alert">{creator.offline.error}</p>}
      <div className="creator-row-actions"><button type="button" disabled={mutating} aria-label={`${creator.following ? 'Unfollow' : 'Follow'} ${creator.name}`} onClick={() => mutate(creator, { following: !creator.following })}>{creator.following ? 'Unfollow' : 'Follow'}<span className="sr-only"> {creator.name}</span></button><button type="button" disabled={mutating} aria-label={`${creator.offline?.enabled ? 'Stop offline sync' : 'Keep available offline'} ${creator.name}`} onClick={() => creator.offline?.enabled ? mutate(creator, { offline_enabled: false }) : openReview({ kind: 'creator', creator })}>{creator.offline?.enabled ? 'Stop offline sync' : 'Keep available offline'}<span className="sr-only"> {creator.name}</span></button></div>
    </li>)}</ul>
    {current.hasMore && <button type="button" disabled={current.loading} onClick={() => load(current.page + 1, true)}>More creators</button>}
    <details className="creator-sync-settings"><summary>Offline sync and hosting</summary>
      <h4>Offline storage</h4>
      {settings && <><p>{offlineLabel(settings)}</p><p>{bytesLabel(settings.downloaded_bytes)} used{settings.storage_limit_bytes > 0 ? ` of ${bytesLabel(settings.storage_limit_bytes)}` : ''}</p>{settings.error && <p role="alert">{settings.error}</p>}</>}
      <button type="button" disabled={mutating || !settings} onClick={() => settings?.sync_all_creators ? mutate({ sync_all_creators: false }) : openReview({ kind: 'all' })}>{settings?.sync_all_creators ? 'Stop syncing all creators' : 'Sync all creators'}</button>
      <button type="button" disabled={mutating || !settings} onClick={() => openReview({ kind: 'budget' })}>Storage limit</button>
      {settings?.downloaded_bytes > 0 && <button type="button" disabled={mutating} onClick={() => openReview({ kind: 'remove' })}>Remove offline copies</button>}
      {settings?.enabled && <button type="button" disabled={mutating} onClick={() => mutate({ paused: !settings.paused })}>{settings.paused ? 'Resume downloads' : 'Pause downloads'}</button>}
      {settings && <><h4>Peer hosting</h4><p>{({ disabled: 'Not hosting', idle: 'Ready to host', hosting: 'Hosting local copies', error: 'Hosting failed' })[settings.peer_hosting_state] || 'Hosting status unavailable'}</p>{settings.peer_hosting_error && <p role="alert">{settings.peer_hosting_error}</p>}<button type="button" disabled={mutating} onClick={() => settings.peer_hosting_enabled ? mutate({ peer_hosting_enabled: false }) : openReview({ kind: 'hosting' })}>{settings.peer_hosting_enabled ? 'Stop peer hosting' : 'Host local copies for peers'}</button></>}
    </details>
    {notice && <p role="alert">{notice}</p>}
    {review && createPortal(<div className="creator-review-backdrop"><form ref={dialogRef} className="creator-review-dialog" role="dialog" aria-modal="true" aria-labelledby="creator-review-title" onSubmit={confirmReview} onKeyDown={trapDialog}>
      <h2 id="creator-review-title">{review.kind === 'creator' ? `Keep ${review.creator.name} available offline` : review.kind === 'all' ? 'Sync all creators' : review.kind === 'hosting' ? 'Host local copies for peers' : review.kind === 'remove' ? 'Remove offline copies' : 'Offline storage limit'}</h2>
      {review.kind === 'all' && <p>Download every creator collection in this community, including future creators, within the storage limit.</p>}
      {review.kind === 'creator' && <p>Download this creator’s published collection and keep it up to date within the community storage limit. Following is a separate choice.</p>}
      {review.kind === 'hosting' ? <p>Share verified community downloads with authorized peers while DonutMediaCenter runs. The community checks each recipient’s access. Hosting uses your network connection and can be stopped at any time. Credentials and personal library files stay private.</p> : review.kind === 'remove' ? <p>Remove {bytesLabel(settings?.downloaded_bytes)} of downloaded copies from this device and stop this community’s downloads and hosting. The community connection, following choices, and your personal library files remain. You will need an available host to download these copies again.</p> : <><label>Community storage limit (GiB)<input type="number" min="0.01" step="0.01" value={budget} onChange={event => setBudget(event.target.value)} required /></label><p>{bytesLabel(settings?.downloaded_bytes)} currently downloaded. Downloads pause when the limit is reached.</p></>}
      {notice && <p role="alert">{notice}</p>}
      <div className="creator-review-actions"><button type="button" disabled={mutating} onClick={() => setReview(null)}>Cancel</button><button type="submit" disabled={mutating || (!['hosting', 'remove'].includes(review.kind) && !validCap)}>{mutating ? 'Saving…' : review.kind === 'hosting' ? 'Start hosting' : review.kind === 'remove' ? 'Remove copies' : review.kind === 'budget' ? 'Save limit' : 'Start downloads'}</button></div>
    </form></div>, document.body)}
  </section>
}
