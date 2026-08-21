import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  browseRemoteSource,
  createPublication,
  createRemoteConnection,
  createRemoteSource,
  deleteRemoteConnection,
  deleteRemoteSource,
  fetchDirectories,
  getPublications,
  getRemoteConnections,
  getRemoteSources,
  importRemoteItem,
  probeRemoteSource,
  simulatePublication,
} from '../api'
import './OnlinePage.css'

const assetUrl = (item, kind = 'sample') =>
  `/api/online/sources/${encodeURIComponent(item.source_id)}/items/${encodeURIComponent(item.remote_post_id)}/asset/${kind}`

function errorMessage(error) {
  return error?.response?.data?.error || error?.message || 'The request failed'
}

export default function OnlinePage() {
  const [tab, setTab] = useState('browse')
  const [sources, setSources] = useState([])
  const [sourceId, setSourceId] = useState('')
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(null)
  const [selected, setSelected] = useState(null)
  const [directories, setDirectories] = useState([])
  const [directoryId, setDirectoryId] = useState('')
  const [publications, setPublications] = useState([])
  const [connections, setConnections] = useState([])
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [form, setForm] = useState({ display_name: '', provider_family: 'danbooru', base_url: '', policy_profile: 'unrestricted' })
  const [connectionForm, setConnectionForm] = useState({ account_display_name: '', auth_kind: 'bearer', secret: '' })

  const selectedSource = useMemo(() => sources.find(source => source.source_id === sourceId), [sources, sourceId])

  const loadSources = useCallback(async () => {
    const result = await getRemoteSources()
    setSources(result.sources || [])
    setSourceId(current => current || result.sources?.[0]?.source_id || '')
  }, [])

  const loadHistory = useCallback(async () => {
    const result = await getPublications()
    setPublications(result.publications || [])
  }, [])

  useEffect(() => {
    Promise.all([loadSources(), fetchDirectories(true), loadHistory()])
      .then(([, result]) => {
        const dirs = result.directories || []
        setDirectories(dirs)
        setDirectoryId(dirs[0]?.id ? String(dirs[0].id) : '')
      })
      .catch(error => setNotice(errorMessage(error)))
  }, [loadHistory, loadSources])

  useEffect(() => {
    if (!sourceId) {
      setConnections([])
      return
    }
    getRemoteConnections(sourceId)
      .then(result => setConnections(result.connections || []))
      .catch(error => setNotice(errorMessage(error)))
  }, [sourceId])

  const browse = async event => {
    event?.preventDefault()
    if (!sourceId) return
    setBusy(true)
    setNotice('')
    try {
      const result = await browseRemoteSource(sourceId, { q: query, page: 1, per_page: 60 })
      setPage(result)
      setSelected(null)
      if (result.stale) setNotice('The source is offline. Showing the most recent cached result.')
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setBusy(false)
    }
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
    if (!selected || !directoryId) return
    setBusy(true)
    try {
      const result = await importRemoteItem(selected.source_id, selected.remote_post_id, {
        kind: selected.media.some(media => media.kind === 'original') ? 'original' : selected.media[0]?.kind,
        directory_id: Number(directoryId),
      })
      setNotice(`Imported as local image #${result.image_id}. The remote item itself was not modified.`)
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setBusy(false)
    }
  }

  const addConnection = async event => {
    event.preventDefault()
    if (!sourceId) return
    setBusy(true)
    try {
      await createRemoteConnection(sourceId, connectionForm)
      const result = await getRemoteConnections(sourceId)
      setConnections(result.connections || [])
      setConnectionForm({ ...connectionForm, account_display_name: '', secret: '' })
      setNotice('Account credential saved in backend-owned storage. Probe the source to verify it.')
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setBusy(false)
    }
  }

  const queuePublication = async () => {
    if (!selectedSource?.capabilities?.upload) return
    setBusy(true)
    try {
      const imageId = window.prompt('Local image ID to publish')
      if (!imageId) return
      const caption = window.prompt('Caption (optional)') || null
      await createPublication({
        snapshot: { library_id: 'primary', directory_id: Number(directoryId || 0), image_id: Number(imageId), content_sha256: null, caption, alt_text: null, content_warning: null, tags: [], rating: null },
        target_source_ids: [selectedSource.source_id],
      })
      await loadHistory()
      setTab('publishing')
      setNotice('Publication queued with an immutable metadata snapshot. Delivery has not been implied.')
    } catch (error) {
      setNotice(errorMessage(error))
    } finally {
      setBusy(false)
    }
  }

  return <div className="online-page">
    <header className="online-header">
      <div><Link to="/" className="online-back">← Local Library</Link><h1>Online</h1><p>Browse remote media without adding it to your library. Import and publish are always explicit actions.</p></div>
      <nav aria-label="Online sections">
        {['browse', 'sources', 'publishing'].map(name => <button key={name} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>{name === 'publishing' ? 'Publishing' : name[0].toUpperCase() + name.slice(1)}</button>)}
      </nav>
    </header>
    {notice && <div className="online-notice" role="status">{notice}</div>}

    {tab === 'browse' && <main>
      <form className="online-toolbar" onSubmit={browse}>
        <select aria-label="Remote source" value={sourceId} onChange={event => { setSourceId(event.target.value); setPage(null) }}>
          <option value="">Choose a source</option>
          {sources.filter(source => source.enabled).map(source => <option key={source.source_id} value={source.source_id}>{source.display_name}</option>)}
        </select>
        <input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search remote tags…" />
        <button disabled={!sourceId || busy}>Browse</button>
      </form>
      {selectedSource && <div className="capability-strip"><strong>{selectedSource.display_name}</strong><span>{selectedSource.provider_family}</span><span>{selectedSource.capabilities.upload ? 'Publishing supported' : 'Read-only source'}</span><span>{selectedSource.policy_profile}</span></div>}
      {!page && <div className="online-empty"><h2>Remote media stays remote</h2><p>Select a connected source and browse. No files are copied until you choose Import.</p></div>}
      {page?.items?.length === 0 && <div className="online-empty">No remote results.</div>}
      <div className="remote-grid">
        {page?.items?.map(item => {
          const kind = item.media.some(media => media.kind === 'thumbnail') ? 'thumbnail' : item.media.some(media => media.kind === 'sample') ? 'sample' : item.media[0]?.kind
          return <button className="remote-card" key={`${item.source_id}:${item.remote_post_id}`} onClick={() => setSelected(item)}>
            {kind ? <img src={assetUrl(item, kind)} alt={item.title || item.tags.slice(0, 5).join(', ') || `Remote post ${item.remote_post_id}`} loading="lazy" /> : <div className="remote-no-preview">No preview</div>}
            <span className="remote-badge">Remote · {selectedSource?.display_name}</span>
            <span className="remote-tags">{item.tags.slice(0, 6).join(' ') || `Post ${item.remote_post_id}`}</span>
          </button>
        })}
      </div>
    </main>}

    {tab === 'sources' && <main className="online-two-column">
      <form className="online-panel" onSubmit={addSource}><h2>Add source</h2>
        <label>Name<input required value={form.display_name} onChange={event => setForm({ ...form, display_name: event.target.value })} /></label>
        <label>Provider<select value={form.provider_family} onChange={event => setForm({ ...form, provider_family: event.target.value })}><option value="danbooru">Danbooru-compatible</option><option value="donutbooru">DonutBooru</option></select></label>
        <label>HTTPS base URL<input required type="url" placeholder="https://example.test" value={form.base_url} onChange={event => setForm({ ...form, base_url: event.target.value })} /></label>
        <label>Content policy<select value={form.policy_profile} onChange={event => setForm({ ...form, policy_profile: event.target.value })}><option value="unrestricted">Unrestricted</option><option value="safe">Safe only</option><option value="adult-hidden">Adult content hidden</option></select></label>
        <button disabled={busy}>Save source</button>
      </form>
      <section className="online-panel"><h2>Connected sources</h2>{sources.length === 0 && <p>No sources configured.</p>}{sources.map(source => <article className="source-row" key={source.source_id}><div><strong>{source.display_name}</strong><small>{source.normalized_base_url}</small><small>{source.capabilities.upload ? 'Browse + publish' : 'Browse only'} · {source.last_probe_at ? 'probed' : 'not probed'}</small></div><div><button onClick={async () => { setSourceId(source.source_id); try { await probeRemoteSource(source.source_id); await loadSources(); const result = await getRemoteConnections(source.source_id); setConnections(result.connections || []); setNotice('Probe succeeded.') } catch (error) { setNotice(errorMessage(error)) } }}>Probe</button><button className="danger" onClick={async () => { if (window.confirm(`Remove ${source.display_name}?`)) { await deleteRemoteSource(source.source_id); await loadSources() } }}>Remove</button></div></article>)}</section>
      <form className="online-panel" onSubmit={addConnection}><h2>Connect account</h2>
        <label>Source<select required value={sourceId} onChange={event => setSourceId(event.target.value)}><option value="">Choose a source</option>{sources.map(source => <option key={source.source_id} value={source.source_id}>{source.display_name}</option>)}</select></label>
        <label>Account label<input required value={connectionForm.account_display_name} onChange={event => setConnectionForm({ ...connectionForm, account_display_name: event.target.value })} /></label>
        <label>Authentication<select value={connectionForm.auth_kind} onChange={event => setConnectionForm({ ...connectionForm, auth_kind: event.target.value })}><option value="bearer">Bearer token</option><option value="api_key_header">X-API-Key header</option></select></label>
        <label>Credential<input required type="password" autoComplete="off" value={connectionForm.secret} onChange={event => setConnectionForm({ ...connectionForm, secret: event.target.value })} /></label>
        <button disabled={!sourceId || busy}>Save account</button>
        <small>The credential is sent directly to backend-owned storage and is never returned to this page.</small>
        {connections.map(connection => <article className="source-row" key={connection.connection_id}><div><strong>{connection.account_display_name || 'Connected account'}</strong><small>{connection.trust_state} · credential stored</small></div><button type="button" className="danger" onClick={async () => { await deleteRemoteConnection(sourceId, connection.connection_id); const result = await getRemoteConnections(sourceId); setConnections(result.connections || []) }}>Disconnect</button></article>)}
      </form>
    </main>}

    {tab === 'publishing' && <main className="online-panel publishing-panel">
      <div className="publishing-heading"><div><h2>Publication history</h2><p>Each destination has its own durable state and receipt.</p></div><button disabled={!selectedSource?.capabilities?.upload || busy} title={selectedSource?.capabilities?.upload ? '' : 'Choose an upload-capable source'} onClick={queuePublication}>New publication</button></div>
      {publications.length === 0 && <div className="online-empty">Nothing has been published from this app.</div>}
      {publications.map(entry => <article className="publication-row" key={entry.target.target_id}><div><strong>Local image #{entry.snapshot.image_id}</strong><small>Target {entry.target.source_id}</small><small>{new Date(entry.created_at).toLocaleString()}</small></div><div><span className={`publication-state state-${entry.target.state}`}>{entry.target.state.replaceAll('_', ' ')}</span>{entry.target.remote_url && <a href={entry.target.remote_url} target="_blank" rel="noreferrer">Open receipt</a>}{entry.target.state === 'queued' && <button onClick={async () => { await simulatePublication(entry.publication_id); await loadHistory() }}>Run fake delivery</button>}</div></article>)}
    </main>}

    {selected && <div className="remote-viewer" role="dialog" aria-modal="true" aria-label="Remote item"><button className="viewer-close" onClick={() => setSelected(null)}>×</button><div className="viewer-media">{selected.media.some(media => media.kind === 'sample') ? <img src={assetUrl(selected, 'sample')} alt={selected.title || selected.tags.join(', ')} /> : <img src={assetUrl(selected, selected.media[0]?.kind)} alt={selected.title || selected.tags.join(', ')} />}</div><aside><span className="remote-badge">Remote item</span><h2>Post {selected.remote_post_id}</h2><p>{selected.tags.join(' ')}</p><a href={selected.canonical_url} target="_blank" rel="noreferrer">Open on source ↗</a><label>Import destination<select value={directoryId} onChange={event => setDirectoryId(event.target.value)}><option value="">Choose watched directory</option>{directories.map(directory => <option key={directory.id} value={directory.id}>{directory.name || directory.path}</option>)}</select></label><button className="import-button" disabled={!directoryId || busy} onClick={importSelected}>Import a local copy</button><small>Browsing does not add this item. Import downloads and verifies one media file, then adds it through the normal library importer.</small></aside></div>}
  </div>
}
