import { useEffect, useMemo, useState } from 'react'
import { authorizeServerForDesktop, confirmDeviceAuthorization, formatPairingFingerprint } from '../devicePairing'

export default function PhonePairingApproval({ request, servers, onClose, onComplete }) {
  const compatibleServers = useMemo(
    () => servers.filter(server => server.token && server.url?.startsWith('https://')),
    [servers],
  )
  const [selected, setSelected] = useState(() => new Set())
  const [confirmName, setConfirmName] = useState('')
  const [authorizing, setAuthorizing] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [nowSeconds, setNowSeconds] = useState(() => Math.floor(Date.now() / 1000))

  useEffect(() => {
    const timer = window.setInterval(() => setNowSeconds(Math.floor(Date.now() / 1000)), 1000)
    return () => window.clearInterval(timer)
  }, [])

  function toggle(serverId) {
    setSelected(previous => {
      const next = new Set(previous)
      if (next.has(serverId)) next.delete(serverId)
      else next.add(serverId)
      return next
    })
  }

  async function authorize() {
    if (confirmName.trim() !== request.displayName || selected.size === 0) return
    if (Number(request.expiresAt) <= Math.floor(Date.now() / 1000)) {
      setError('This authorization request has expired.')
      return
    }
    setAuthorizing(true)
    setError(null)
    try {
      await confirmDeviceAuthorization()
    } catch (approvalError) {
      setAuthorizing(false)
      setError(approvalError.message)
      return
    }
    if (Number(request.expiresAt) <= Math.floor(Date.now() / 1000)) {
      setAuthorizing(false)
      setError('This authorization request expired during device confirmation.')
      return
    }
    const settled = await Promise.all(compatibleServers
      .filter(server => selected.has(server.id))
      .map(async server => {
        try {
          if (Number(request.expiresAt) <= Math.floor(Date.now() / 1000)) {
            throw new Error('Authorization request expired')
          }
          const value = await authorizeServerForDesktop(server, request)
          return { success: true, server, value }
        } catch (approvalError) {
          return { success: false, server, error: approvalError.message }
        }
      }))
    setResults(settled)
    setAuthorizing(false)
    if (settled.some(result => result.success)) onComplete?.(settled)
    if (!settled.some(result => result.success)) setError('No selected server could complete authorization.')
  }

  const isExpired = Number(request.expiresAt) <= nowSeconds
  const finished = results.length > 0

  return (
    <div className="pairing-approval-overlay" role="dialog" aria-modal="true" aria-labelledby="pairing-approval-title">
      <div className="pairing-approval-card">
        <h2 id="pairing-approval-title">Authorize this desktop</h2>
        <div className="pairing-request-identity">
          <strong>{request.displayName}</strong>
          <span>Fingerprint</span>
          <code>{formatPairingFingerprint(request.publicKeyFingerprint)}</code>
          <span>Requested access: browse and curate over local network</span>
          <span>Desktop destination: {(request.callbackUrls || []).join(', ')}</span>
          <span>{isExpired ? 'This request has expired.' : `Expires at ${new Date(request.expiresAt * 1000).toLocaleTimeString()}`}</span>
        </div>

        {!finished && (
          <>
            <p>Choose exactly which authenticated servers this desktop may connect to. Your phone sessions are never copied.</p>
            <div className="pairing-server-choices">
              {compatibleServers.map(server => (
                <label key={server.id} className="pairing-server-choice">
                  <input type="checkbox" checked={selected.has(server.id)} onChange={() => toggle(server.id)} disabled={authorizing || isExpired} />
                  <span><strong>{server.name}</strong><small>{server.url}</small></span>
                </label>
              ))}
              {compatibleServers.length === 0 && <p className="pairing-warning">No saved server has a compatible authenticated token. Reconnect this phone to a current LocalBooru server first.</p>}
              {servers.some(server => !server.token || !server.url?.startsWith('https://')) && <p className="pairing-warning">Servers using legacy password sessions or unencrypted HTTP are not eligible. Connect them over HTTPS with QR authentication first.</p>}
            </div>
            <label className="pairing-name-confirmation">
              To confirm the requesting device, type <strong>{request.displayName}</strong>
              <input autoComplete="off" value={confirmName} onChange={event => setConfirmName(event.target.value)} disabled={authorizing || isExpired} />
            </label>
            <div className="pairing-approval-actions">
              <button className="pairing-cancel" onClick={onClose} disabled={authorizing}>Reject</button>
              <button className="pairing-authorize" onClick={authorize} disabled={authorizing || isExpired || selected.size === 0 || confirmName.trim() !== request.displayName}>
                {authorizing ? 'Authorizing…' : `Authorize ${selected.size || ''} server${selected.size === 1 ? '' : 's'}`}
              </button>
            </div>
          </>
        )}

        {finished && (
          <div className="pairing-phone-results">
            {results.map(result => (
              <div className={result.success ? 'success' : 'failure'} key={result.server.id}>
                <strong>{result.server.name}</strong>: {result.success ? 'authorized' : result.error}
              </div>
            ))}
            <button className="pairing-authorize" onClick={onClose}>Done</button>
          </div>
        )}
        {error && <div className="scan-error">{error}</div>}
      </div>
    </div>
  )
}
