// SPDX-License-Identifier: GPL-3.0-only
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { authorizeHub, createHub, getHubConnection, getRemoteConnections, previewHub } from '../api'
import { openHubBrowser, parseHubEnrollmentLink, safeHubHttpsUrl, validateHubReview } from '../utils/hubConnection'
import './HubConnection.css'

const defaults = { base_url: 'https://donutbooru.com', peer_id: '', relay_url: '', transport_mode: 'relayed', allow_local_network: false }
const message = error => error?.response?.data?.detail || error?.response?.data?.error || error?.message || 'The community request failed.'

function HubDetails({ hub }) {
  const descriptor = hub.descriptor
  return <dl className="hub-details">
    <dt>Community</dt><dd>{descriptor.instance.name}</dd>
    <dt>Hub identity</dt><dd>{descriptor.instance.id}</dd>
    <dt>Address</dt><dd>{hub.normalized_base_url || descriptor.instance.origin}</dd>
    <dt>Connection</dt><dd>{hub.transport_mode}</dd>
    {hub.bootstrap?.peer_id && <><dt>Peer identity</dt><dd>{hub.bootstrap.peer_id}</dd><dt>Relay</dt><dd>{hub.bootstrap.relay_url}</dd></>}
    <dt>Browse access</dt><dd>{descriptor.access.catalog === 'public' ? 'Public · no sign-in required' : 'Sign in required · community membership controls access'}</dd>
    <dt>Sign-in methods</dt><dd>{descriptor.authentication.methods.join(', ') || 'None advertised'}</dd>
    <dt>Registration</dt><dd>{descriptor.access.registration}</dd>
    <dt>Submission</dt><dd>{descriptor.access.submission} · moderation {descriptor.access.moderation}</dd>
  </dl>
}

function SavedHubConnection({ sourceId }) {
  const [hub, setHub] = useState(null)
  const [connections, setConnections] = useState([])
  const [consent, setConsent] = useState(false)
  const [authorizationUrl, setAuthorizationUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const generation = useRef(0)

  useEffect(() => {
    const current = ++generation.current
    let disposed = false
    const load = async () => {
      try {
        const [result, accounts] = await Promise.all([getHubConnection(sourceId), getRemoteConnections(sourceId)])
        if (disposed || current !== generation.current) return
        setHub(validateHubReview(result))
        setConnections(accounts.connections || [])
        setConsent(false)
        setAuthorizationUrl('')
        setNotice('')
      } catch (error) {
        if (disposed || current !== generation.current) return
        setHub(null)
        if (error?.response?.status !== 404) setNotice(message(error))
      }
    }
    load()
    const authorized = event => {
      if (!event.detail?.source_id || event.detail.source_id === sourceId) load()
    }
    window.addEventListener('dmc-hub-authorized', authorized)
    return () => { disposed = true; window.removeEventListener('dmc-hub-authorized', authorized) }
  }, [sourceId])

  const signIn = async () => {
    if (!consent || !hub) return
    const current = generation.current
    setBusy(true)
    setNotice('')
    try {
      const result = await authorizeHub(sourceId, { allow_browser_direct: true })
      if (current !== generation.current) return
      if (!safeHubHttpsUrl(result.authorization_url) || new URL(result.authorization_url).origin !== new URL(hub.descriptor.instance.origin).origin) throw new Error('The sign-in address does not belong to this community.')
      setAuthorizationUrl(result.authorization_url)
      await openHubBrowser(result.authorization_url)
      setNotice('Finish signing in in your browser, then return to DMC. Your media connection keeps its saved route.')
    } catch (error) {
      if (current === generation.current) setNotice(message(error))
    } finally {
      if (current === generation.current) setBusy(false)
    }
  }

  if (!hub) return notice ? <p role="status">{notice}</p> : null
  const signedIn = connections.some(connection => connection.has_credential && connection.trust_state === 'verified' && (!connection.expires_at || connection.expires_at * 1000 > Date.now()))
  return <section className="online-panel hub-account">
    <h2>{hub.descriptor.instance.name}</h2>
    <p>{signedIn ? 'Account connected' : hub.descriptor.access.catalog === 'public' ? 'Public browsing is available without an account.' : 'Sign in required to browse this community.'}</p>
    <details><summary>Saved community identity and access</summary><HubDetails hub={hub} /></details>
    <Link to={`/shared/${encodeURIComponent(sourceId)}?media_type=image`}>Browse community →</Link>
    {hub.descriptor.capabilities.native_authorization && <>
      <p className="hub-browser-boundary">Sign-in opens your system browser. The browser connects directly to the community and its sign-in provider; it does not use DMC’s relay. Catalog and media requests keep using the saved {hub.transport_mode.toLowerCase()} connection.</p>
      <label className="online-checkbox"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} />I allow direct browser access for this sign-in.</label>
      <button type="button" disabled={!consent || busy} onClick={signIn}>{signedIn ? 'Sign in with another account' : 'Sign in in browser'}</button>
      {authorizationUrl && consent && <button type="button" onClick={async () => { try { await openHubBrowser(authorizationUrl) } catch (error) { setNotice(message(error)) } }}>Continue sign-in in browser (direct)</button>}
    </>}
    {notice && <p role="status">{notice}</p>}
  </section>
}

export default function HubConnection({ sourceId, onConnected }) {
  const [form, setForm] = useState(defaults)
  const [link, setLink] = useState('')
  const [review, setReview] = useState(null)
  const [name, setName] = useState('')
  const [policy, setPolicy] = useState('unrestricted')
  const [confirmed, setConfirmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const revision = useRef(0)

  useEffect(() => {
    if (sourceId) return
    const receiveLink = value => {
      try {
        const parsed = parseHubEnrollmentLink(value)
        revision.current += 1
        setLink(value)
        setForm({ ...defaults, ...parsed })
        setReview(null)
        setConfirmed(false)
        setNotice(parsed.collection ? `This link names collection ${parsed.collection}. Review the community before adding it; collection access remains subject to its policy.` : 'Link loaded. Review the community before adding it.')
      } catch (error) { setNotice(message(error)) }
    }
    const receive = event => receiveLink(event.detail)
    window.addEventListener('dmc-hub-link', receive)
    if (window.__dmcPendingHubLink) {
      const pending = window.__dmcPendingHubLink
      delete window.__dmcPendingHubLink
      receiveLink(pending)
    }
    return () => window.removeEventListener('dmc-hub-link', receive)
  }, [sourceId])

  if (sourceId) return <SavedHubConnection key={sourceId} sourceId={sourceId} />

  const change = (key, value) => {
    revision.current += 1
    setForm(current => ({ ...current, [key]: value }))
    setReview(null)
    setConfirmed(false)
    setNotice('')
  }
  const loadLink = () => {
    try {
      const parsed = parseHubEnrollmentLink(link)
      revision.current += 1
      setForm({ ...defaults, ...parsed })
      setReview(null)
      setConfirmed(false)
      setNotice('Link loaded. Review the community before adding it.')
    } catch (error) { setNotice(message(error)) }
  }
  const preview = async event => {
    event.preventDefault()
    if (!form.peer_id || !form.relay_url) {
      setNotice('A community connection needs its published peer identity and relay address. Ask the operator for an Open in DMC link; these values cannot be guessed from the website address.')
      return
    }
    const current = revision.current
    setBusy(true)
    setNotice('')
    try {
      const result = await previewHub(form)
      if (current !== revision.current) return
      setReview(validateHubReview(result))
      setName(result.descriptor.instance.name)
      setConfirmed(false)
    } catch (error) { if (current === revision.current) setNotice(message(error)) }
    finally { setBusy(false) }
  }
  const save = async () => {
    if (!review || !confirmed) return
    setBusy(true)
    setNotice('')
    try {
      const result = await createHub({ ...form, expected_instance_id: review.descriptor.instance.id, display_name: name.trim(), policy_profile: policy })
      setReview(null)
      setConfirmed(false)
      setNotice(result.hub?.descriptor?.access?.catalog === 'members' ? 'Community added. Sign in is required to browse its catalog.' : 'Community added. Public browsing does not require an account.')
      await onConnected?.(result.source)
    } catch (error) { setNotice(message(error)) }
    finally { setBusy(false) }
  }

  return <section className="online-panel hub-connection">
    <h2>Add community</h2>
    <p>Connect DonutBooru or another compatible community. Review its identity and access before adding it.</p>
    <label>Open in DMC link<textarea value={link} disabled={busy} onChange={event => setLink(event.target.value)} placeholder="donutmediacenter://hub/add?url=…&peer_id=…&relay_url=…" /></label>
    <button type="button" disabled={!link.trim() || busy} onClick={loadLink}>Load link</button>
    <form onSubmit={preview}>
      <label>Community address<input type="url" required value={form.base_url} disabled={busy} onChange={event => change('base_url', event.target.value)} /></label>
      <label>Connection<select value={form.transport_mode} disabled={busy} onChange={event => change('transport_mode', event.target.value)}><option value="relayed">Relayed (default)</option><option value="standard">Standard · Iroh direct with relay fallback</option></select></label>
      {form.transport_mode === 'relayed' ? <p>Community traffic uses the published Iroh relay. DMC does not fall back to a direct website connection.</p> : <p>Standard allows a direct Iroh connection to the community, with its published relay as fallback.</p>}
        <label>Published peer identity<input required value={form.peer_id} disabled={busy} onChange={event => change('peer_id', event.target.value)} /></label>
        <label>Published relay address<input type="url" required value={form.relay_url} disabled={busy} onChange={event => change('relay_url', event.target.value)} placeholder="https://relay.example" /></label>
      <button disabled={busy}>Review community</button>
    </form>
    {review && <section className="hub-review">
      <h3>Confirm community</h3>
      <HubDetails hub={review} />
      <label>Name in DMC<input value={name} disabled={busy} onChange={event => setName(event.target.value)} /></label>
      <label>Content policy<select value={policy} disabled={busy} onChange={event => setPolicy(event.target.value)}><option value="unrestricted">Unrestricted</option><option value="safe">Safe only</option><option value="adult-hidden">Adult content hidden</option></select></label>
      <label className="online-checkbox"><input type="checkbox" checked={confirmed} disabled={busy} onChange={event => setConfirmed(event.target.checked)} />I reviewed this identity, name, access policy and sign-in methods.</label>
      <button type="button" disabled={busy || !confirmed || !name.trim()} onClick={save}>Add community</button>
    </section>}
    {notice && <p className="hub-notice" role="status">{notice}</p>}
  </section>
}
