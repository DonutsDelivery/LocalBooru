import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import HubConnection from '../components/HubConnection'
import { openHubBrowser, safeHubHttpsUrl } from '../utils/hubConnection'
import {
  browseRemoteSource,
  createPublication,
  deliverPublication,
  createRemoteConnection,
  createRemoteSource,
  deleteRemoteConnection,
  deleteRemoteSource,
  fetchDirectories,
  getMediaUrl,
  getPublications,
  getRemoteConnections,
  getHubConnection,
  getRemoteSources,
  importRemoteItem,
  probeRemoteSource,
  refreshPublication,
} from '../api'
import './OnlinePage.css'

const assetUrl = (item, kind = 'sample') =>
  getMediaUrl(`/api/online/sources/${encodeURIComponent(item.source_id)}/items/${encodeURIComponent(item.remote_post_id)}/asset/${kind}`)

function errorMessage(error) {
  return error?.response?.data?.detail || error?.response?.data?.error || error?.message || 'The request failed'
}

export default function OnlinePage({ initialTab = 'browse', fixedSourceId = '', mediaType = '' }) {
  const embedded = Boolean(fixedSourceId)
  const [tab, setTab] = useState(initialTab)
  const [sources, setSources] = useState([])
  const [chosenSourceId, setSourceId] = useState(fixedSourceId)
  const sourceId = fixedSourceId || chosenSourceId
  const [query, setQuery] = useState('')
  const [pageResult, setPage] = useState(null)
  const page = pageResult?.source_id === sourceId && pageResult?.media_type === mediaType ? pageResult : null
  const [selected, setSelected] = useState(null)
  const [directories, setDirectories] = useState([])
  const [directoryId, setDirectoryId] = useState('')
  const [publications, setPublications] = useState([])
  const [connectionResult, setConnectionResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [hubResult, setHub] = useState(null)
  const hub = hubResult?.source_id === sourceId ? hubResult : null
  const requestGeneration = useRef(0)
  const submittedQuery = useRef('')
  const currentSource = useRef(sourceId)
  const sourceGeneration = useRef(0)
  const connectionRequestGeneration = useRef(0)
  if (currentSource.current !== sourceId) {
    sourceGeneration.current += 1
    currentSource.current = sourceId
  }
  const connections = connectionResult?.source_id === sourceId && connectionResult?.generation === sourceGeneration.current ? connectionResult.connections : []
  const [form, setForm] = useState({ display_name: '', base_url: '', provider_family: 'donutbooru', policy_profile: 'unrestricted', allow_local_network: false })
  const [connectionForm, setConnectionForm] = useState({ account_display_name: '', auth_kind: 'bearer', secret: '' })

  const selectedSource = useMemo(() => sources.find(source => source.source_id === sourceId), [sources, sourceId])
  const hasPublishingAccount = connections.some(connection => connection.has_credential && (!connection.expires_at || connection.expires_at * 1000 > Date.now()))

  const selectProvider = provider => setForm(current => ({ ...current, provider_family: provider }))

  const loadSources = useCallback(async () => {
    const result = await getRemoteSources()
    setSources(result.sources || [])
    setSourceId(current => fixedSourceId || (result.sources?.some(source => source.source_id === current) ? current : result.sources?.[0]?.source_id || ''))
  }, [fixedSourceId])

  const loadHistory = useCallback(async () => {
    const result = await getPublications()
    setPublications(result.publications || [])
  }, [])

  useEffect(() => {
    Promise.all([loadSources(), fetchDirectories(true), embedded ? Promise.resolve() : loadHistory()])
      .then(([, result]) => {
        const dirs = result.directories || []
        setDirectories(dirs)
        setDirectoryId(dirs[0]?.id ? String(dirs[0].id) : '')
      })
      .catch(error => setNotice(errorMessage(error)))
  }, [loadHistory, loadSources, embedded])

  const loadConnections = useCallback(async () => {
    if (!sourceId || embedded) return
    const sourceVersion = sourceGeneration.current
    const requestVersion = ++connectionRequestGeneration.current
    try {
      const result = await getRemoteConnections(sourceId)
      if (currentSource.current !== sourceId || sourceGeneration.current !== sourceVersion || connectionRequestGeneration.current !== requestVersion) return
      setConnectionResult({ source_id: sourceId, generation: sourceVersion, connections: result.connections || [] })
    } catch (error) {
      if (currentSource.current === sourceId && sourceGeneration.current === sourceVersion && connectionRequestGeneration.current === requestVersion) setNotice(errorMessage(error))
    }
  }, [sourceId, embedded])

  useEffect(() => {
    let disposed = false
    const sourceVersion = sourceGeneration.current
    setConnectionForm({ account_display_name: '', auth_kind: 'bearer', secret: '' })
    setConnectionResult(null)
    setSelected(null)
    setPage(null)
    setBusy(false)
    setNotice('')
    if (!sourceId) return
    getHubConnection(sourceId).then(result => {
      if (!disposed && sourceGeneration.current === sourceVersion) setHub({ ...result, source_id: sourceId })
    }).catch(() => { if (!disposed && sourceGeneration.current === sourceVersion) setHub(null) })
    loadConnections()
    const authorized = event => {
      if (event.detail?.source_id === sourceId) loadConnections()
    }
    window.addEventListener('dmc-hub-authorized', authorized)
    return () => {
      disposed = true
      connectionRequestGeneration.current += 1
      window.removeEventListener('dmc-hub-authorized', authorized)
    }
  }, [sourceId, loadConnections])

  const browsePage = useCallback(async (requestedPage = 1, requestedQuery = '', cursor = undefined) => {
    if (!sourceId) return
    const generation = ++requestGeneration.current
    setBusy(true)
    setNotice('')
    setSelected(null)
    setPage(null)
    try {
      const result = await browseRemoteSource(sourceId, { q: requestedQuery, page: requestedPage, per_page: 60, ...(cursor ? { cursor } : {}), ...(mediaType ? { media_type: mediaType } : {}) })
      if (generation !== requestGeneration.current || currentSource.current !== sourceId) return
      submittedQuery.current = requestedQuery
      setPage({ ...result, source_id: sourceId, media_type: mediaType })
      if (result.stale) setNotice('The community is offline. Showing the most recent cached result.')
    } catch (error) {
      if (generation === requestGeneration.current && currentSource.current === sourceId) setNotice(errorMessage(error))
    } finally {
      if (generation === requestGeneration.current) setBusy(false)
    }
  }, [sourceId, mediaType])

  useEffect(() => {
    if (!embedded || !fixedSourceId) return
    setSourceId(fixedSourceId)
  }, [embedded, fixedSourceId])

  useEffect(() => {
    if (embedded && sourceId === fixedSourceId) browsePage(1, '')
    return () => { requestGeneration.current += 1 }
  }, [embedded, sourceId, fixedSourceId, browsePage])

  const browse = event => {
    event?.preventDefault()
    browsePage(1, query)
  }

  const addSource = async event => {
    event.preventDefault()
    setBusy(true)
    try {
      const source = await createRemoteSource(form)
      await loadSources()
      setSourceId(source.source_id)
      setForm({ ...form, display_name: '', base_url: '' })
      setNotice('Source saved. Probe it before relying on its capabilities.')
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setBusy(false)
    }
  }

  const importSelected = async () => {
    if (!selected || !directoryId || selected.source_id !== sourceId) return
    const sourceVersion = sourceGeneration.current
    const current = () => currentSource.current === sourceId && sourceGeneration.current === sourceVersion
    setBusy(true)
    try {
      const result = await importRemoteItem(selected.source_id, selected.remote_post_id, {
        kind: selected.media.some(media => media.kind === 'original') ? 'original' : selected.media[0]?.kind,
        directory_id: Number(directoryId),
      })
      if (current()) setNotice(`Imported as local image #${result.image_id}. The remote item itself was not modified.`)
    } catch (error) {
      if (current()) setNotice(errorMessage(error))
    } finally {
      if (current()) setBusy(false)
    }
  }

  const addConnection = async event => {
    event.preventDefault()
    if (!sourceId) return
    const sourceVersion = sourceGeneration.current
    const current = () => currentSource.current === sourceId && sourceGeneration.current === sourceVersion
    connectionRequestGeneration.current += 1
    setBusy(true)
    try {
      await createRemoteConnection(sourceId, connectionForm)
      if (!current()) return
      await loadConnections()
      if (!current()) return
      setConnectionForm({ account_display_name: '', auth_kind: connectionForm.auth_kind, secret: '' })
      setNotice('Account credential saved in backend-owned storage. Probe the source to verify it.')
    } catch (error) {
      if (current()) setNotice(errorMessage(error))
    } finally {
      if (current()) setBusy(false)
    }
  }

  const disconnectAccount = async connectionId => {
    const sourceVersion = sourceGeneration.current
    const current = () => currentSource.current === sourceId && sourceGeneration.current === sourceVersion
    connectionRequestGeneration.current += 1
    setBusy(true)
    try {
      await deleteRemoteConnection(sourceId, connectionId)
      if (current()) await loadConnections()
    } catch (error) {
      if (current()) setNotice(errorMessage(error))
    } finally {
      if (current()) setBusy(false)
    }
  }

  const probeSource = async targetSourceId => {
    const sourceVersion = sourceGeneration.current
    const current = () => currentSource.current === sourceId && sourceGeneration.current === sourceVersion
    try {
      await probeRemoteSource(targetSourceId)
      await loadSources()
      if (!current()) return
      if (targetSourceId === sourceId) await loadConnections()
      if (current()) setNotice('Probe succeeded.')
    } catch (error) {
      if (current()) setNotice(errorMessage(error))
    }
  }

  const removeSource = async source => {
    if (!window.confirm(`Remove ${source.display_name}?`)) return
    const sourceVersion = sourceGeneration.current
    try {
      await deleteRemoteSource(source.source_id)
      await loadSources()
    } catch (error) {
      if (currentSource.current === sourceId && sourceGeneration.current === sourceVersion) setNotice(errorMessage(error))
    }
  }

  const queuePublication = async () => {
    if (!selectedSource?.capabilities?.upload) return
    const sourceVersion = sourceGeneration.current
    const current = () => currentSource.current === sourceId && sourceGeneration.current === sourceVersion
    setBusy(true)
    try {
      const imageId = window.prompt('Local image ID to publish')
      if (!imageId) return
      if (!/^\d+$/.test(imageId) || Number(imageId) < 1) {
        setNotice('Enter a valid local image ID.')
        return
      }
      const caption = window.prompt('Caption (optional)') || null
      await createPublication({
        snapshot: { library_id: 'primary', directory_id: Number(directoryId || 0), image_id: Number(imageId), content_sha256: null, caption, alt_text: null, content_warning: null, tags: [], rating: null },
        target_source_ids: [selectedSource.source_id],
      })
      await loadHistory()
      if (!current()) return
      setTab('publishing')
      setNotice('Publication queued with an immutable metadata snapshot. Delivery has not been implied.')
    } catch (error) {
      if (current()) setNotice(errorMessage(error))
    } finally {
      if (current()) setBusy(false)
    }
  }

  return <div className="online-page">
    <header className="online-header">
      <div><h2>{embedded ? selectedSource?.display_name || 'Shared community' : 'Communities & Nodes'}</h2><p>{embedded ? `Browse shared ${mediaType === 'video' ? 'videos' : 'images'} from this community. Import saves a local copy.` : 'Add communities, manage accounts and choose how to connect.'}</p></div>
      {!embedded && <nav aria-label="Online sections">
        {['browse', 'sources', 'publishing'].map(name => <button key={name} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>{name === 'publishing' ? 'Publishing' : name[0].toUpperCase() + name.slice(1)}</button>)}
      </nav>}
    </header>
    {notice && <div className="online-notice" role="status">{notice}</div>}

    {(embedded || tab === 'browse') && <section className="online-main">
      <form className="online-toolbar" onSubmit={browse}>
        {!embedded && <select aria-label="Remote source" value={sourceId} onChange={event => { setSourceId(event.target.value); setPage(null); setSelected(null); setHub(null); setConnectionResult(null); setBusy(false) }}>
          <option value="">Choose a source</option>
          {sources.filter(source => source.enabled).map(source => <option key={source.source_id} value={source.source_id}>{source.display_name}</option>)}
        </select>}
        <input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search remote tags…" />
        <button disabled={!sourceId || busy}>Browse</button>
      </form>
      {selectedSource && <div className="capability-strip"><strong>{selectedSource.display_name}</strong><span>{selectedSource.provider_family}</span><span>{selectedSource.capabilities?.upload ? 'Publishing supported' : 'Read-only source'}</span><span>{selectedSource.policy_profile}</span>{hub && <span>{hub.transport_mode}</span>}</div>}
      {embedded && <p><Link to="/settings/fediverse">Community account and connection settings →</Link></p>}
      {hub?.descriptor?.access?.catalog === 'members' && <p>Sign in required. This community controls membership and catalog access.</p>}
      {!page && <div className="online-empty"><h2>Remote media stays remote</h2><p>Select a connected source and browse. No files are copied until you choose Import.</p></div>}
      {page?.items?.length === 0 && <div className="online-empty">No {mediaType || 'remote'} results on this page.</div>}
      <div className="remote-grid">
        {page?.items?.filter(item => item.source_id === sourceId).map(item => {
          const kind = item.media.some(media => media.kind === 'thumbnail') ? 'thumbnail' : item.media.some(media => media.kind === 'sample') ? 'sample' : item.media[0]?.kind
          return <button className="remote-card" key={`${item.source_id}:${item.remote_post_id}`} onClick={() => setSelected(item)}>
            {kind && !item.media.find(media => media.kind === kind)?.mime_type?.startsWith('video/') ? <img src={assetUrl(item, kind)} alt={item.title || item.tags.slice(0, 5).join(', ') || `Remote post ${item.remote_post_id}`} loading="lazy" /> : <div className="remote-no-preview">No preview</div>}
            <span className="remote-badge">Remote · {selectedSource?.display_name}</span>
            <span className="remote-tags">{item.tags.slice(0, 6).join(' ') || `Post ${item.remote_post_id}`}</span>
          </button>
        })}
      </div>
      {page && <nav className="remote-pagination" aria-label="Remote result pages"><button disabled={busy || page.page <= 1} onClick={() => browsePage(page.page - 1, submittedQuery.current)}>Previous</button><span>Page {page.page}{page.stale ? ' · cached' : ''}</span><button disabled={busy || (!page.next_cursor && (page.total != null ? page.page * page.per_page >= page.total : page.items.length < page.per_page))} onClick={() => browsePage(page.page + 1, submittedQuery.current, page.next_cursor)}>Next</button></nav>}
    </section>}

    {!embedded && tab === 'sources' && <section className="online-main online-two-column">
      <HubConnection onConnected={async source => { await loadSources(); setSourceId(source.source_id); setPage(null) }} />
      <section className="online-panel"><h2>Communities</h2>{sources.length === 0 && <p>No communities added yet.</p>}{sources.map(source => <article className="source-row" key={source.source_id}><div><strong>{source.display_name}</strong><small>{source.normalized_base_url}</small><Link to={`/shared/${encodeURIComponent(source.source_id)}?media_type=image`}>Browse community →</Link></div><button onClick={() => { setSourceId(source.source_id); setConnectionResult(null); setHub(null) }}>Account and connection</button></article>)}</section>
      {sourceId && <HubConnection sourceId={sourceId} />}
      <details className="online-advanced"><summary>Advanced provider sources and manual credentials</summary>
      <div className="online-two-column">
      <form className="online-panel" onSubmit={addSource}><h2>Add provider source</h2>
        <label>Name<input required value={form.display_name} onChange={event => setForm({ ...form, display_name: event.target.value })} /></label>
        <label>Provider<select value={form.provider_family} onChange={event => selectProvider(event.target.value)}><option value="danbooru">Danbooru-compatible</option><option value="donutbooru">DonutBooru</option></select></label>
        <label>HTTPS base URL<input required type="url" placeholder="https://example.test" value={form.base_url} onChange={event => setForm({ ...form, base_url: event.target.value })} /></label>
        <label>Content policy<select value={form.policy_profile} onChange={event => setForm({ ...form, policy_profile: event.target.value })}><option value="unrestricted">Unrestricted</option><option value="safe">Safe only</option><option value="adult-hidden">Adult content hidden</option></select></label>
        <details className="online-connection-options">
          <summary>Advanced connection settings</summary>
          <label className="online-checkbox"><input type="checkbox" checked={form.allow_local_network} aria-describedby="source-local-network-help" onChange={event => setForm({ ...form, allow_local_network: event.target.checked })} />Allow connections to private network addresses</label>
          <p id="source-local-network-help">Enable only for a source hosted on this computer or a private network, such as localhost or 192.168.1.10. DMC can then connect to private addresses and use HTTP for that source. This does not share your library or allow incoming connections.</p>
        </details>
        <button disabled={busy}>Save source</button>
      </form>
      <section className="online-panel"><h2>Connected sources</h2>{sources.length === 0 && <p>No sources configured.</p>}{sources.map(source => <article className="source-row" key={source.source_id}><div><strong>{source.display_name}</strong><small>{source.normalized_base_url}</small><small>{source.capabilities.upload ? 'Browse + publish' : 'Browse only'} · {source.last_probe_at ? 'probed' : 'not probed'}</small></div><div><button onClick={() => probeSource(source.source_id)}>Probe</button><button className="danger" onClick={() => removeSource(source)}>Remove</button></div></article>)}</section>
      <form className="online-panel" onSubmit={addConnection}><h2>Connect account</h2>
        <label>Source<select required value={sourceId} onChange={event => setSourceId(event.target.value)}><option value="">Choose a source</option>{sources.map(source => <option key={source.source_id} value={source.source_id}>{source.display_name}</option>)}</select></label>
        <label>Account label<input disabled={busy} required value={connectionForm.account_display_name} onChange={event => setConnectionForm({ ...connectionForm, account_display_name: event.target.value })} /></label>
        <label>Authentication<select disabled={busy} value={connectionForm.auth_kind} onChange={event => setConnectionForm({ ...connectionForm, auth_kind: event.target.value })}><option value="bearer">Bearer token</option><option value="api_key_header">X-API-Key header</option></select></label>
        <label>Credential<input disabled={busy} required type="password" autoComplete="off" value={connectionForm.secret} onChange={event => setConnectionForm({ ...connectionForm, secret: event.target.value })} /></label>
        <button disabled={!sourceId || busy}>Save account</button>
        <small>The credential is sent directly to backend-owned storage and is never returned to this page.</small>
        {connections.map(connection => <article className="source-row" key={connection.connection_id}><div><strong>{connection.account_display_name || 'Connected account'}</strong><small>{connection.trust_state} · {connection.has_credential ? 'credential stored' : 'sign in again'}</small></div><button type="button" className="danger" disabled={busy} onClick={() => disconnectAccount(connection.connection_id)}>Disconnect</button></article>)}
      </form>
      </div></details>
    </section>}

    {!embedded && tab === 'publishing' && <section className="online-main online-panel publishing-panel">
      <div className="publishing-heading"><div><h2>Publication history</h2><p>Each destination has its own durable state and receipt.</p></div><div><select aria-label="Publication source" value={sourceId} onChange={event => setSourceId(event.target.value)}><option value="">Choose a source</option>{sources.filter(source => source.capabilities.upload).map(source => <option key={source.source_id} value={source.source_id}>{source.display_name}</option>)}</select><select aria-label="Local directory" value={directoryId} onChange={event => setDirectoryId(event.target.value)}><option value="">Choose directory</option>{directories.map(directory => <option key={directory.id} value={directory.id}>{directory.name || directory.path}</option>)}</select><button disabled={!selectedSource?.capabilities?.upload || !hasPublishingAccount || !directoryId || busy} title={!hasPublishingAccount ? 'Connect a publishing account in Sources first' : ''} onClick={queuePublication}>New publication</button></div></div>
      {publications.length === 0 && <div className="online-empty">Nothing has been published from this app.</div>}
      {publications.map(entry => <article className="publication-row" key={entry.target.target_id}><div><strong>Local image #{entry.snapshot.image_id}</strong><small>Target {entry.target.source_id}</small><small>{new Date(entry.created_at).toLocaleString()}</small>{entry.target.error?.message && <small role="alert">{entry.target.error.message}</small>}</div><div><span className={`publication-state state-${entry.target.state}`}>{entry.target.state.replaceAll('_', ' ')}</span>{safeHubHttpsUrl(entry.target.remote_url) && <button onClick={async () => { if (!window.confirm('Open the receipt directly in your system browser? This browser connection does not use DMC’s relay.')) return; try { await openHubBrowser(entry.target.remote_url) } catch (error) { setNotice(errorMessage(error)) } }}>Open receipt in browser (direct)</button>}{['submitted', 'pending_moderation'].includes(entry.target.state) && <button disabled={busy} onClick={async () => { setBusy(true); try { await refreshPublication(entry.publication_id); await loadHistory(); setNotice('Publication status refreshed from the community.') } catch (error) { setNotice(errorMessage(error)) } finally { setBusy(false) } }}>Refresh status</button>}{['queued', 'failed_retryable'].includes(entry.target.state) && <button disabled={busy} onClick={async () => { setBusy(true); try { const result = await deliverPublication(entry.publication_id); await loadHistory(); const failed = result.targets?.find(target => target.error); setNotice(failed ? failed.error.message : 'Publication delivery completed.') } catch (error) { setNotice(errorMessage(error)) } finally { setBusy(false) } }}>{entry.target.state === 'queued' ? 'Publish now' : 'Retry'}</button>}</div></article>)}
    </section>}

    {selected?.source_id === sourceId && <div className="remote-viewer" role="dialog" aria-modal="true" aria-label="Remote item"><button className="viewer-close" onClick={() => setSelected(null)}>×</button><div className="viewer-media">{(selected.duration != null || selected.media.some(media => media.mime_type?.startsWith('video/'))) && selected.media.some(media => media.kind === 'original') ? <video src={assetUrl(selected, 'original')} controls /> : selected.media.some(media => media.kind === 'sample') ? <img src={assetUrl(selected, 'sample')} alt={selected.title || selected.tags.join(', ')} /> : <img src={assetUrl(selected, selected.media[0]?.kind)} alt={selected.title || selected.tags.join(', ')} />}</div><aside><span className="remote-badge">Remote item</span><h2>Post {selected.remote_post_id}</h2><p>{selected.tags.join(' ')}</p>{safeHubHttpsUrl(selected.canonical_url) && <button type="button" onClick={async () => { if (hub?.transport_mode?.toLowerCase() === 'relayed' && !window.confirm('Open the source directly in your system browser? This browser connection does not use DMC’s relay.')) return; try { await openHubBrowser(selected.canonical_url) } catch (error) { setNotice(errorMessage(error)) } }}>Open source in browser (direct) ↗</button>}<label>Import destination<select value={directoryId} onChange={event => setDirectoryId(event.target.value)}><option value="">Choose watched directory</option>{directories.map(directory => <option key={directory.id} value={directory.id}>{directory.name || directory.path}</option>)}</select></label><button className="import-button" disabled={!directoryId || busy} onClick={importSelected}>Import a local copy</button><small>Browsing does not add this item. Import downloads and verifies one media file, then adds it through the normal library importer.</small></aside></div>}
  </div>
}
