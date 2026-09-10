import { useState, useEffect, useRef } from 'react'
import {
  getServers,
  addServer,
  addOrUpdateServer,
  updateServer,
  removeServer,
  setActiveServerId,
  testServerConnection,
  probeServer,
  pingAllServers,
  serverFromQrHandshake,
  isMobileApp,
  LOCAL_SERVER
} from '../serverManager'
import { updateServerConfig, verifyHandshake } from '../api'
import { validateDesktopPairingRequest } from '../devicePairing'
import { scanQrCode } from '../qrScanner'
import PhonePairingApproval from './PhonePairingApproval'
import './ServerSelectScreen.css'

export default function ServerSelectScreen({ servers: initialServers, serverStatuses: initialStatuses, error: initialError, onConnect, onAddServer }) {
  const [servers, setServers] = useState(initialServers || [])
  const [statuses, setStatuses] = useState(initialStatuses || {})
  const [connecting, setConnecting] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingServer, setEditingServer] = useState(null)
  const [inlineError, setInlineError] = useState(initialError || null)
  const [desktopPairingRequest, setDesktopPairingRequest] = useState(null)

  // Update inline error when prop changes
  useEffect(() => {
    if (initialError) setInlineError(initialError)
  }, [initialError])

  // Load servers if not provided
  useEffect(() => {
    if (!initialServers) {
      loadServers()
    }
  }, [initialServers])

  async function loadServers() {
    const serverList = await getServers()
    setServers(serverList)
    if (serverList.length > 0) {
      setRefreshing(true)
      const newStatuses = await pingAllServers(serverList)
      setStatuses(newStatuses)
      setRefreshing(false)
    }
  }

  async function handleRefresh() {
    setRefreshing(true)
    const serverList = await getServers()
    setServers(serverList)
    if (serverList.length > 0) {
      const newStatuses = await pingAllServers(serverList)
      setStatuses(newStatuses)
    }
    setRefreshing(false)
  }

  async function handleConnectLocal() {
    setConnecting(LOCAL_SERVER.id)
    setInlineError(null)
    await setActiveServerId(LOCAL_SERVER.id)
    await updateServerConfig()
    onConnect?.()
    setConnecting(null)
  }

  async function handleConnect(server) {
    setConnecting(server.id)
    setInlineError(null)
    // Probe primary, then fallback URL on network failure
    const result = await probeServer(server)
    if (result.success) {
      await setActiveServerId(server.id)
      await updateServerConfig(result.url)
      onConnect?.()
    } else {
      // Update status to show it's offline
      setStatuses(prev => ({ ...prev, [server.id]: 'offline' }))
      const isAuthError = result.error?.includes('401') || result.error?.includes('Authentication')
      if (isAuthError) {
        setStatuses(prev => ({ ...prev, [server.id]: 'auth_failed' }))
        setInlineError(`Authentication failed for ${server.name}. Please re-scan QR code to re-pair.`)
      } else {
        setInlineError(`Could not connect to ${server.name}: ${result.error}`)
      }
    }
    setConnecting(null)
  }

  async function handleEditSave(serverData) {
    if (editingServer) {
      await updateServer(editingServer.id, serverData)
    }
    setEditingServer(null)
    await loadServers()
  }

  async function handleDelete(server, e) {
    e.stopPropagation()
    if (!confirm(`Remove "${server.name}"?`)) return
    await removeServer(server.id)
    await loadServers()
  }

  async function handleScanQR() {
    try {
      setScanError(null)
      setScanning(true)
      const decodedText = await scanQrCode({ idPrefix: 'qr-scanner-select' })
        // Parse QR data
        let qrData
        try {
          qrData = JSON.parse(decodedText)
        } catch {
          setScanError('Invalid QR code format')
          return
        }

        // Reverse pairing: this phone authorizes the requesting desktop.
        if (qrData.type === 'localbooru-desktop-pairing') {
          if (!await validateDesktopPairingRequest(qrData)) {
            setScanError('This desktop authorization QR is invalid or expired')
            return
          }
          setDesktopPairingRequest(qrData)
          return
        }

        // Legacy phone-connect QR.
        if (qrData.type !== 'localbooru') {
          setScanError('Not a LocalBooru QR code')
          return
        }

        // Try connecting - local first, then public
        // On Tauri mobile, use IPC to bypass WebView mixed-content restrictions
        const useTauriIPC = window.__TAURI_INTERNALS__ !== undefined
        let invoke
        if (useTauriIPC) {
          invoke = (await import('@tauri-apps/api/core')).invoke
        }

        let workingUrl = null
        let urls = []
        if (qrData.local) urls.push(qrData.local)
        if (qrData.public) urls.push(qrData.public)

        const errors = []
        for (const url of urls) {
          try {
            let testResult
            if (useTauriIPC) {
              testResult = await invoke('test_remote_server', { url })
            } else {
              testResult = await testServerConnection(url)
            }
            if (testResult.success) {
              workingUrl = url
              break
            }
            errors.push(`${url}: ${testResult.error}`)
          } catch (err) {
            errors.push(`${url}: ${err.message}`)
          }
        }

        if (!workingUrl) {
          setScanError(`Could not connect. Tried: ${errors.join('; ')}`)
          return
        }

        // Verify handshake and get JWT token
        let pairedServer = null
        if (qrData.nonce) {
          try {
            let handshakeResult
            if (useTauriIPC) {
              handshakeResult = await invoke('verify_remote_handshake', { url: workingUrl, nonce: qrData.nonce })
            } else {
              handshakeResult = await verifyHandshake(workingUrl, qrData.nonce)
            }
            pairedServer = serverFromQrHandshake(qrData, workingUrl, handshakeResult)
          } catch (err) {
            setScanError(`Pairing authorization failed: ${err.message}`)
            return
          }
        }
        if (!pairedServer) {
          setScanError('This server QR cannot issue a revocable device credential. Create a new QR on the server.')
          return
        }

        // Add or update existing server (avoids duplicates on re-pair)
        const newServer = await addOrUpdateServer(pairedServer)

        // Auto-connect to the new server
        await setActiveServerId(newServer.id)
        await updateServerConfig()
        onConnect?.()
    } catch (err) {
      console.error('Scan error:', err)
      setScanError(err.message || 'Failed to start QR scanner. Check camera permissions.')
    } finally {
      setScanning(false)
    }
  }

  async function handleSaveServer(serverData) {
    const newServer = await addServer(serverData)
    await loadServers()
    setShowAddModal(false)
    // Auto-connect to the new server
    await setActiveServerId(newServer.id)
    await updateServerConfig()
    onConnect?.()
  }

  return (
    <div className="server-select-screen">
      <div className="server-select-header">
        <h1>LocalBooru</h1>
        <p>Select a server to connect</p>
      </div>

      {inlineError && (
        <div className="scan-error">
          {inlineError}
          <button className="dismiss-btn" onClick={() => setInlineError(null)}>Dismiss</button>
        </div>
      )}

      {scanError && (
        <div className="scan-error">
          {scanError}
          <button className="dismiss-btn" onClick={() => setScanError(null)}>Dismiss</button>
        </div>
      )}

      <div className="server-select-list">
        {/* "This Device" — local embedded server, always first, always online */}
        <div
          className={`server-select-card local-server ${connecting === LOCAL_SERVER.id ? 'connecting' : ''}`}
          onClick={handleConnectLocal}
        >
          <div className="server-status-indicator">
            <span className="status-dot connected" title="Online"></span>
          </div>

          <div className="server-select-info">
            <div className="server-select-name">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: 6, verticalAlign: 'text-bottom' }}>
                <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/>
                <line x1="12" y1="18" x2="12.01" y2="18"/>
              </svg>
              This Device
              <span className="local-badge">Local</span>
            </div>
            <div className="server-select-url">Embedded server</div>
          </div>

          <div className="server-select-actions">
            {connecting === LOCAL_SERVER.id ? (
              <span className="connecting-text">Connecting...</span>
            ) : (
              <button
                className="connect-btn"
                onClick={(e) => { e.stopPropagation(); handleConnectLocal() }}
              >
                Connect
              </button>
            )}
          </div>
        </div>

        {/* Remote servers */}
        {servers.map(server => (
          <div
            key={server.id}
            className={`server-select-card ${connecting === server.id ? 'connecting' : ''} ${statuses[server.id] === 'auth_failed' ? 'auth-failed' : ''}`}
            onClick={() => handleConnect(server)}
          >
            <div className="server-status-indicator">
              {refreshing ? (
                <span className="status-dot testing" title="Checking..."></span>
              ) : statuses[server.id] === 'online' ? (
                <span className="status-dot connected" title="Online"></span>
              ) : statuses[server.id] === 'auth_failed' ? (
                <span className="status-dot error" title="Auth Failed"></span>
              ) : (
                <span className="status-dot error" title="Offline"></span>
              )}
            </div>

            <div className="server-select-info">
              <div className="server-select-name">{server.name}</div>
              <div className="server-select-url">{server.url}</div>
              {statuses[server.id] === 'auth_failed' && (
                <div className="server-select-auth-error">Authentication failed — re-scan QR to re-pair</div>
              )}
            </div>

            <div className="server-select-actions">
              {connecting === server.id ? (
                <span className="connecting-text">Connecting...</span>
              ) : (
                <>
                  {statuses[server.id] === 'auth_failed' ? (
                    <button
                      className="connect-btn re-pair-btn"
                      onClick={(e) => { e.stopPropagation(); handleScanQR() }}
                    >
                      Re-pair
                    </button>
                  ) : (
                    <button
                      className="connect-btn"
                      onClick={(e) => { e.stopPropagation(); handleConnect(server) }}
                    >
                      Connect
                    </button>
                  )}
                  <button
                    className="edit-server-btn"
                    onClick={(e) => { e.stopPropagation(); setEditingServer(server) }}
                    title="Edit server"
                  >
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                    </svg>
                  </button>
                  <button
                    className="delete-server-btn"
                    onClick={(e) => handleDelete(server, e)}
                    title="Remove server"
                  >
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"/>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                  </button>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="server-select-footer">
        <button
          className="refresh-btn"
          onClick={handleRefresh}
          disabled={refreshing}
          title="Refresh server status"
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" className={refreshing ? 'spinning' : ''}>
            <path d="M23 4v6h-6"/>
            <path d="M1 20v-6h6"/>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
          </svg>
        </button>
        <button
          className="scan-qr-btn"
          onClick={handleScanQR}
          disabled={scanning}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="7" height="7"/>
            <rect x="14" y="3" width="7" height="7"/>
            <rect x="3" y="14" width="7" height="7"/>
            <rect x="14" y="14" width="3" height="3"/>
            <rect x="18" y="14" width="3" height="3"/>
            <rect x="14" y="18" width="3" height="3"/>
            <rect x="18" y="18" width="3" height="3"/>
          </svg>
          {scanning ? 'Scanning...' : 'Scan QR'}
        </button>
        <button
          className="add-server-btn"
          onClick={() => setShowAddModal(true)}
        >
          + Add Server
        </button>
      </div>

      {showAddModal && (
        <AddServerModal
          onSave={handleSaveServer}
          onClose={() => setShowAddModal(false)}
        />
      )}
      {editingServer && (
        <AddServerModal
          server={editingServer}
          onSave={handleEditSave}
          onClose={() => setEditingServer(null)}
        />
      )}
      {desktopPairingRequest && (
        <PhonePairingApproval
          request={desktopPairingRequest}
          servers={servers}
          onClose={() => setDesktopPairingRequest(null)}
        />
      )}
    </div>
  )
}

function AddServerModal({ server, onSave, onClose }) {
  const [name, setName] = useState(server?.name || '')
  const [url, setUrl] = useState(server?.url || '')
  const [fallbackUrl, setFallbackUrl] = useState(server?.fallbackUrl || '')
  const [username, setUsername] = useState(server?.username || '')
  const [password, setPassword] = useState(server?.password || '')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  function normalizeUrl(value) {
    let v = value.trim()
    if (!v) return ''
    if (!v.startsWith('http://') && !v.startsWith('https://')) {
      v = 'http://' + v
    }
    return v.replace(/\/$/, '')
  }

  async function handleTest() {
    if (!url) return

    setTesting(true)
    setTestResult(null)

    const normalizedUrl = normalizeUrl(url)
    const normalizedFallback = fallbackUrl ? normalizeUrl(fallbackUrl) : ''
    setUrl(normalizedUrl)
    if (normalizedFallback) setFallbackUrl(normalizedFallback)

    const primary = await testServerConnection(normalizedUrl, username, password)
    if (primary.success) {
      setTestResult({ ...primary, usedFallback: false })
      setTesting(false)
      return
    }
    if (normalizedFallback && primary.networkFailure) {
      const fb = await testServerConnection(normalizedFallback, username, password)
      if (fb.success) {
        setTestResult({ success: true, usedFallback: true })
        setTesting(false)
        return
      }
    }
    setTestResult(primary)
    setTesting(false)
  }

  function handleSave() {
    if (!url) return
    // For new servers, require a successful test. When editing, allow save without
    // re-testing — the user may be adding a fallback URL while offline from the primary.
    if (!server && !testResult?.success) return

    // Normalize so URLs always have a scheme — without it, reqwest in the proxy
    // produces "builder error" because the URL parser rejects scheme-less inputs.
    const finalUrl = normalizeUrl(url)
    const finalFallback = fallbackUrl ? normalizeUrl(fallbackUrl) : null

    // Extract hostname for default name
    let defaultName = name
    if (!defaultName) {
      try {
        const urlObj = new URL(finalUrl)
        defaultName = urlObj.hostname
      } catch {
        defaultName = 'LocalBooru Server'
      }
    }

    onSave({
      name: defaultName,
      url: finalUrl,
      fallbackUrl: finalFallback,
      username: username || null,
      password: password || null,
      lastConnected: new Date().toISOString()
    })
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h3>{server ? 'Edit Server' : 'Add Server'}</h3>

        <div className="form-group">
          <label>Server URL</label>
          <input
            type="text"
            placeholder="192.168.1.100:8790"
            value={url}
            onChange={e => setUrl(e.target.value)}
          />
          <small>IP address or hostname with port</small>
        </div>

        <div className="form-group">
          <label>Fallback URL (optional)</label>
          <input
            type="text"
            placeholder="100.x.x.x:8790 (Tailscale)"
            value={fallbackUrl}
            onChange={e => setFallbackUrl(e.target.value)}
          />
          <small>Used when the primary URL is unreachable (e.g. away from LAN)</small>
        </div>

        <div className="form-group">
          <label>Name (optional)</label>
          <input
            type="text"
            placeholder="My Server"
            value={name}
            onChange={e => setName(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Username (optional)</label>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Password (optional)</label>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
          />
        </div>

        {testResult && (
          <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
            {testResult.success
              ? (testResult.usedFallback ? 'Connection successful (via fallback URL)!' : 'Connection successful!')
              : `Error: ${testResult.error}`}
          </div>
        )}

        <div className="modal-actions">
          <button className="cancel-btn" onClick={onClose}>Cancel</button>
          <button
            className="test-btn"
            onClick={handleTest}
            disabled={testing || !url}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button
            className="save-btn"
            onClick={handleSave}
            disabled={!url || (!server && !testResult?.success)}
          >
            {server ? 'Save' : 'Save & Connect'}
          </button>
        </div>
      </div>
    </div>
  )
}
