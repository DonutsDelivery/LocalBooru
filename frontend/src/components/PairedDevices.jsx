import { useCallback, useEffect, useState } from 'react'
import { getLocalPairingApiBase } from '../devicePairing'

async function request(path, options = {}) {
  const fetchImpl = window.__TAURI_INTERNALS__
    ? (await import('@tauri-apps/plugin-http')).fetch
    : fetch
  const response = await fetchImpl(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || body.error || `Request failed (${response.status})`)
  return body
}

export default function PairedDevices() {
  const [devices, setDevices] = useState([])
  const [error, setError] = useState(null)
  const [revoking, setRevoking] = useState(null)

  const load = useCallback(async () => {
    try {
      const apiBase = await getLocalPairingApiBase()
      const body = await request(`${apiBase}/devices`)
      setDevices(body.devices || [])
      setError(null)
    } catch (loadError) {
      setError(loadError.message)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  async function revoke(device) {
    if (!window.confirm(`Revoke “${device.displayName}”? It will immediately lose access to this server.`)) return
    setRevoking(device.deviceId)
    try {
      const apiBase = await getLocalPairingApiBase()
      await request(`${apiBase}/devices/${encodeURIComponent(device.deviceId)}`, { method: 'DELETE' })
      await load()
    } catch (revokeError) {
      setError(revokeError.message)
    } finally {
      setRevoking(null)
    }
  }

  if (devices.length === 0 && !error) return null

  return (
    <section className="paired-devices">
      <h3>Authorized desktops</h3>
      {devices.map(device => (
        <div className={`paired-device ${device.revokedAt ? 'revoked' : ''}`} key={device.deviceId}>
          <div>
            <strong>{device.displayName}</strong>
            <small>{device.publicKeyFingerprint.slice(0, 24)}</small>
            <small>{device.revokedAt ? `Revoked ${new Date(`${device.revokedAt}Z`).toLocaleString()}` : `Authorized ${new Date(`${device.createdAt}Z`).toLocaleString()}`}</small>
          </div>
          {!device.revokedAt && <button onClick={() => revoke(device)} disabled={revoking === device.deviceId}>{revoking === device.deviceId ? 'Revoking…' : 'Revoke'}</button>}
        </div>
      ))}
      {error && <div className="qr-error">{error}</div>}
    </section>
  )
}
