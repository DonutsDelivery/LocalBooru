import { useEffect, useRef, useState } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import {
  cancelDesktopPairingSession,
  createDesktopPairingSession,
  formatPairingFingerprint,
  inspectDesktopDelivery,
  pollDesktopPairingSession,
  redeemDesktopPayload,
} from '../devicePairing'
import { setActiveServerId } from '../serverManager'
import { updateServerConfig } from '../api'

const DEFAULT_DEVICE_NAME = `${navigator.platform || 'LocalBooru'} Desktop`

export default function DesktopPairing({ active = true }) {
  const [deviceName, setDeviceName] = useState(() => localStorage.getItem('localbooru_device_name') || DEFAULT_DEVICE_NAME)
  const [pairing, setPairing] = useState(null)
  const [status, setStatus] = useState('idle')
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [secondsLeft, setSecondsLeft] = useState(0)
  const redeemedRef = useRef(new Set())
  const connectingRef = useRef(new Set())
  const [connecting, setConnecting] = useState(() => new Set())

  useEffect(() => {
    if (!pairing) return
    let stopped = false
    const tick = async () => {
      const remaining = Math.max(0, pairing.session.expiresAt - Math.floor(Date.now() / 1000))
      if (!stopped) setSecondsLeft(remaining)
      if (remaining === 0) {
        if (!stopped) setStatus('expired')
        return
      }
      try {
        const latest = await pollDesktopPairingSession(pairing.session.sessionId, pairing.session.sessionSecret)
        for (const delivery of latest.deliveries || []) {
          const key = `${delivery.serverId}:${delivery.encryptedPayload}`
          if (redeemedRef.current.has(key)) continue
          redeemedRef.current.add(key)
          try {
            const payload = await inspectDesktopDelivery(delivery, pairing)
            if (!stopped) setResults(previous => [...previous, { success: null, payload }])
          } catch (deliveryError) {
            if (!stopped) setResults(previous => [...previous, { success: false, serverId: delivery.serverId, error: deliveryError.message }])
          }
        }
      } catch (pollError) {
        if (remaining > 1 && !stopped) setError(pollError.message)
      }
      if (!stopped) window.setTimeout(tick, 1000)
    }
    tick()
    return () => { stopped = true }
  }, [pairing])

  useEffect(() => () => {
    if (pairing?.session?.sessionId) void cancelDesktopPairingSession(pairing.session.sessionId, pairing.session.sessionSecret)
  }, [pairing])

  useEffect(() => {
    if (!active && pairing?.session?.sessionId) {
      void cancelDesktopPairingSession(pairing.session.sessionId, pairing.session.sessionSecret)
      setPairing(null)
      setStatus('idle')
      setSecondsLeft(0)
    }
  }, [active, pairing])

  async function startPairing() {
    try {
      if (pairing?.session?.sessionId) await cancelDesktopPairingSession(pairing.session.sessionId, pairing.session.sessionSecret)
      setError(null)
      setResults([])
      redeemedRef.current.clear()
      setStatus('creating')
      const created = await createDesktopPairingSession(deviceName.trim())
      localStorage.setItem('localbooru_device_name', deviceName.trim())
      setPairing(created)
      setSecondsLeft(created.session.expiresAt - Math.floor(Date.now() / 1000))
      setStatus('waiting')
    } catch (startError) {
      setStatus('idle')
      setError(startError.message)
    }
  }

  async function activateServer(server) {
    await setActiveServerId(server.id)
    await updateServerConfig()
    window.location.reload()
  }

  async function connectPendingServer(pendingResult) {
    const resultId = pendingResult.payload.serverId
    if (connectingRef.current.has(resultId)) return
    connectingRef.current.add(resultId)
    setConnecting(previous => new Set(previous).add(resultId))
    setError(null)
    try {
      const server = await redeemDesktopPayload(pendingResult.payload, pairing)
      setResults(previous => previous.map(result => result.payload?.serverId === resultId ? { success: true, server } : result))
    } catch (connectError) {
      setResults(previous => previous.map(result => result.payload?.serverId === resultId
        ? { ...result, success: false, serverId: resultId, error: connectError.message }
        : result))
    } finally {
      connectingRef.current.delete(resultId)
      setConnecting(previous => {
        const next = new Set(previous)
        next.delete(resultId)
        return next
      })
    }
  }

  const successful = results.filter(result => result.success)
  const failed = results.filter(result => result.success === false)
  const pending = results.filter(result => result.success === null)

  return (
    <section className="desktop-pairing">
      <h3>Connect servers from phone</h3>
      <p className="pairing-lead">Authorize this desktop from a phone that is already connected to your LocalBooru servers.</p>

      <label className="pairing-device-name">
        Desktop name
        <input value={deviceName} maxLength={80} onChange={event => setDeviceName(event.target.value)} disabled={status === 'creating' || status === 'waiting'} />
      </label>

      {status !== 'waiting' && status !== 'expired' && successful.length === 0 && (
        <>
          <button className="pairing-primary" onClick={startPairing} disabled={!deviceName.trim() || status === 'creating'}>
            {status === 'creating' ? 'Creating secure request…' : 'Show authorization QR'}
          </button>
          {error && <div className="qr-error pairing-create-error" role="alert">{error}</div>}
        </>
      )}

      {(status === 'waiting' || status === 'expired') && pairing && (
        <>
          <div className={`qr-container pairing-qr ${status === 'expired' ? 'expired' : ''}`}>
            <QRCodeSVG
              value={JSON.stringify(pairing.session)}
              size={480}
              level="L"
              boostLevel={false}
              marginSize={4}
              bgColor="#ffffff"
              fgColor="#000000"
              title="LocalBooru desktop authorization"
            />
          </div>
          <div className="pairing-summary">
            <strong>{pairing.session.displayName}</strong>
            <span>Fingerprint: {formatPairingFingerprint(pairing.session.publicKeyFingerprint)}</span>
            <span>{status === 'expired' ? 'Request expired' : `Expires in ${secondsLeft}s`}</span>
            <span>Scope: browse and curate over local network</span>
          </div>
          {status === 'waiting' ? (
            <p className="pairing-waiting">On your phone, tap Scan QR and choose exactly which servers to authorize.</p>
          ) : (
            <button className="pairing-primary" onClick={startPairing}>Create a new QR</button>
          )}
        </>
      )}

      {successful.length > 0 && (
        <div className="pairing-results success">
          <strong>{successful.length} server{successful.length === 1 ? '' : 's'} connected to “{deviceName}”</strong>
          {successful.map(({ server }) => (
            <div className="pairing-result" key={server.id}>
              <span>{server.name}</span>
              <button onClick={() => activateServer(server)}>Use server</button>
            </div>
          ))}
          {status === 'waiting' && <button className="pairing-secondary" onClick={startPairing}>Authorize more servers</button>}
        </div>
      )}

      {pending.length > 0 && (
        <div className="pairing-results pending">
          <strong>Confirm servers on this desktop</strong>
          <p>The phone approved these destinations. Check each address before exchanging its one-time grant.</p>
          {pending.map(result => (
            <div className="pairing-result" key={result.payload.serverId}>
              <span><strong>{result.payload.serverName}</strong><small>{result.payload.url}</small></span>
              <button onClick={() => connectPendingServer(result)} disabled={connecting.has(result.payload.serverId)}>
                {connecting.has(result.payload.serverId) ? 'Connecting…' : 'Connect server'}
              </button>
            </div>
          ))}
        </div>
      )}

      {failed.length > 0 && (
        <div className="pairing-results failure">
          {failed.map(result => (
            <div key={result.serverId}>
              {result.serverId}: {result.error}
              {result.payload && <button onClick={() => connectPendingServer(result)} disabled={connecting.has(result.serverId)}>Retry</button>}
            </div>
          ))}
        </div>
      )}
      {error && (status === 'waiting' || status === 'expired' || successful.length > 0) && <div className="qr-error" role="alert">{error}</div>}

      <ul className="pairing-security-notes">
        <li>The QR expires after two minutes and contains no reusable server credential.</li>
        <li>Each server issues this desktop a separate, independently revocable session.</li>
        <li>Approval is bound to the fingerprint shown above; a copied QR cannot redeem the grant on another device.</li>
      </ul>
    </section>
  )
}
