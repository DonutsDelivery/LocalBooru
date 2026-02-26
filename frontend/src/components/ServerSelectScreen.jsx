import { useState, useEffect, useRef } from 'react'
import {
  getServers,
  addServer,
  removeServer,
  setActiveServerId,
  testServerConnection,
  pingAllServers,
  isMobileApp,
  LOCAL_SERVER
} from '../serverManager'
import { updateServerConfig, verifyHandshake } from '../api'
import './ServerSelectScreen.css'

export default function ServerSelectScreen({ servers: initialServers, serverStatuses: initialStatuses, error: initialError, onConnect, onAddServer }) {
  const [servers, setServers] = useState(initialServers || [])
  const [statuses, setStatuses] = useState(initialStatuses || {})
  const [connecting, setConnecting] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const [inlineError, setInlineError] = useState(initialError || null)

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
    // Test connection first
    const result = await testServerConnection(server.url, server.username, server.password)
    if (result.success) {
      await setActiveServerId(server.id)
      await updateServerConfig()
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

  async function handleDelete(server, e) {
    e.stopPropagation()
    if (!confirm(`Remove "${server.name}"?`)) return
    await removeServer(server.id)
    await loadServers()
  }

  async function handleScanQR() {
    let stream = null
    let scanInterval = null

    try {
      setScanError(null)
      setScanning(true)

      // Get camera stream directly
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      })

      // Create fullscreen overlay with our own video element
      const overlay = document.createElement('div')
      overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;background:#000;display:flex;flex-direction:column;'

      // Video element — we control this directly
      const video = document.createElement('video')
      video.setAttribute('autoplay', '')
      video.setAttribute('playsinline', '')
      video.setAttribute('muted', '')
      video.style.cssText = 'flex:1;width:100%;object-fit:cover;background:#000;'
      video.srcObject = stream
      overlay.appendChild(video)

      // Scan target indicator
      const indicator = document.createElement('div')
      indicator.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:250px;height:250px;border:3px solid rgba(99,102,241,0.8);border-radius:16px;pointer-events:none;'
      overlay.appendChild(indicator)

      // Bottom bar with cancel
      const bottomBar = document.createElement('div')
      bottomBar.style.cssText = 'padding:16px;display:flex;justify-content:center;background:rgba(0,0,0,0.7);'
      const closeBtn = document.createElement('button')
      closeBtn.textContent = 'Cancel'
      closeBtn.style.cssText = 'padding:12px 32px;font-size:16px;background:#333;color:#fff;border:none;border-radius:8px;cursor:pointer;'
      bottomBar.appendChild(closeBtn)
      overlay.appendChild(bottomBar)

      document.body.appendChild(overlay)

      await video.play()

      // Canvas for frame capture
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d', { willReadFrequently: true })

      // Use BarcodeDetector if available (Android Chrome 83+), else fall back to html5-qrcode
      let decoder
      if ('BarcodeDetector' in window) {
        decoder = new BarcodeDetector({ formats: ['qr_code'] })
      } else {
        const { Html5Qrcode } = await import('html5-qrcode')
        // We'll use canvas-based decoding below
        decoder = null
      }

      let cleaned = false
      const cleanup = () => {
        if (cleaned) return
        cleaned = true
        clearInterval(scanInterval)
        stream.getTracks().forEach(t => t.stop())
        overlay.remove()
        setScanning(false)
      }

      closeBtn.onclick = cleanup

      const processResult = async (decodedText) => {
        cleanup()

        // Parse QR data
        let qrData
        try {
          qrData = JSON.parse(decodedText)
        } catch {
          setScanError('Invalid QR code format')
          return
        }

        // Validate QR data
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
        let token = null
        if (qrData.nonce) {
          try {
            let handshakeResult
            if (useTauriIPC) {
              handshakeResult = await invoke('verify_remote_handshake', { url: workingUrl, nonce: qrData.nonce })
            } else {
              handshakeResult = await verifyHandshake(workingUrl, qrData.nonce)
            }
            if (handshakeResult.success && handshakeResult.token) {
              token = handshakeResult.token
            }
          } catch (err) {
            console.error('[QR] Handshake verification failed:', err.message)
          }
        }

        // Add the server
        const newServer = await addServer({
          name: qrData.name || 'LocalBooru Server',
          url: workingUrl,
          token,
          username: null,
          password: null,
          lastConnected: new Date().toISOString()
        })

        // Auto-connect to the new server
        await setActiveServerId(newServer.id)
        await updateServerConfig()
        onConnect?.()
      }

      // Scan frames periodically
      scanInterval = setInterval(async () => {
        if (cleaned || video.readyState < 2) return

        try {
          if (decoder && decoder.detect) {
            // BarcodeDetector API — can scan video directly
            const barcodes = await decoder.detect(video)
            if (barcodes.length > 0) {
              processResult(barcodes[0].rawValue)
            }
          } else {
            // Fallback: capture frame to canvas, then decode with html5-qrcode
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            ctx.drawImage(video, 0, 0)
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
            // Use jsQR-style decode via html5-qrcode internal
            const { Html5Qrcode } = await import('html5-qrcode')
            const blob = await new Promise(r => canvas.toBlob(r, 'image/png'))
            const file = new File([blob], 'frame.png', { type: 'image/png' })
            try {
              const result = await Html5Qrcode.scanFile(file, false)
              processResult(result)
            } catch {
              // No QR found in this frame — continue scanning
            }
          }
        } catch {
          // Scan error for this frame, ignore
        }
      }, 200) // 5 fps scanning

    } catch (err) {
      if (stream) stream.getTracks().forEach(t => t.stop())
      if (scanInterval) clearInterval(scanInterval)
      console.error('Scan error:', err)
      setScanError(err.message || 'Failed to start QR scanner. Check camera permissions.')
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
                <div className="server-select-auth-error">Authentication expired — re-scan QR to re-pair</div>
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
    </div>
  )
}

function AddServerModal({ onSave, onClose }) {
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  async function handleTest() {
    if (!url) return

    setTesting(true)
    setTestResult(null)

    // Normalize URL
    let normalizedUrl = url.trim()
    if (!normalizedUrl.startsWith('http://') && !normalizedUrl.startsWith('https://')) {
      normalizedUrl = 'http://' + normalizedUrl
    }
    // Remove trailing slash
    normalizedUrl = normalizedUrl.replace(/\/$/, '')

    setUrl(normalizedUrl)

    const result = await testServerConnection(normalizedUrl, username, password)
    setTestResult(result)
    setTesting(false)
  }

  function handleSave() {
    if (!url || !testResult?.success) return

    // Extract hostname for default name
    let defaultName = name
    if (!defaultName) {
      try {
        const urlObj = new URL(url)
        defaultName = urlObj.hostname
      } catch {
        defaultName = 'LocalBooru Server'
      }
    }

    onSave({
      name: defaultName,
      url: url.replace(/\/$/, ''),
      username: username || null,
      password: password || null,
      lastConnected: new Date().toISOString()
    })
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h3>Add Server</h3>

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
            {testResult.success ? 'Connection successful!' : `Error: ${testResult.error}`}
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
            disabled={!testResult?.success}
          >
            Save & Connect
          </button>
        </div>
      </div>
    </div>
  )
}
