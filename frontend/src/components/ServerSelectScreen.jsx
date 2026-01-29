import { useState, useEffect } from 'react'
import {
  getServers,
  addServer,
  removeServer,
  setActiveServerId,
  testServerConnection,
  pingAllServers,
  isMobileApp
} from '../serverManager'
import { updateServerConfig } from '../api'
import './ServerSelectScreen.css'

// Dynamic import for barcode scanner (only on mobile)
let BarcodeScanner = null
if (isMobileApp()) {
  import('@capacitor-mlkit/barcode-scanning').then(module => {
    BarcodeScanner = module.BarcodeScanner
  })
}

export default function ServerSelectScreen({ servers: initialServers, serverStatuses: initialStatuses, onConnect, onAddServer }) {
  const [servers, setServers] = useState(initialServers || [])
  const [statuses, setStatuses] = useState(initialStatuses || {})
  const [connecting, setConnecting] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)

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

  async function handleConnect(server) {
    setConnecting(server.id)
    // Test connection first
    const result = await testServerConnection(server.url, server.username, server.password)
    if (result.success) {
      await setActiveServerId(server.id)
      await updateServerConfig()
      onConnect?.()
    } else {
      // Update status to show it's offline
      setStatuses(prev => ({ ...prev, [server.id]: 'offline' }))
      alert(`Could not connect to ${server.name}: ${result.error}`)
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
    if (!BarcodeScanner) {
      setScanError('QR scanner not available')
      return
    }

    try {
      setScanError(null)
      setScanning(true)

      // Check camera permission
      const { camera } = await BarcodeScanner.checkPermissions()
      if (camera !== 'granted') {
        const { camera: newPerm } = await BarcodeScanner.requestPermissions()
        if (newPerm !== 'granted') {
          setScanError('Camera permission required to scan QR codes')
          setScanning(false)
          return
        }
      }

      // Start scanning
      document.body.classList.add('barcode-scanner-active')
      const result = await BarcodeScanner.scan()
      document.body.classList.remove('barcode-scanner-active')

      if (!result.barcodes.length) {
        setScanError('No QR code found')
        setScanning(false)
        return
      }

      // Parse QR data
      const rawValue = result.barcodes[0].rawValue
      let qrData
      try {
        qrData = JSON.parse(rawValue)
      } catch {
        setScanError('Invalid QR code format')
        setScanning(false)
        return
      }

      // Validate QR data
      if (qrData.type !== 'localbooru') {
        setScanError('Not a LocalBooru QR code')
        setScanning(false)
        return
      }

      // Try connecting - local first, then public
      let workingUrl = null
      let urls = []
      if (qrData.local) urls.push(qrData.local)
      if (qrData.public) urls.push(qrData.public)

      for (const url of urls) {
        const testResult = await testServerConnection(url)
        if (testResult.success) {
          workingUrl = url
          break
        }
      }

      if (!workingUrl) {
        setScanError('Could not connect to server. Make sure you are on the same network.')
        setScanning(false)
        return
      }

      // Add the server
      const newServer = await addServer({
        name: qrData.name || 'LocalBooru Server',
        url: workingUrl,
        username: null,
        password: null,
        lastConnected: new Date().toISOString()
      })

      // Auto-connect to the new server
      await setActiveServerId(newServer.id)
      await updateServerConfig()
      onConnect?.()

    } catch (err) {
      console.error('Scan error:', err)
      setScanError(err.message || 'Failed to scan QR code')
      document.body.classList.remove('barcode-scanner-active')
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

      {scanError && (
        <div className="scan-error">
          {scanError}
          <button className="dismiss-btn" onClick={() => setScanError(null)}>Dismiss</button>
        </div>
      )}

      {servers.length === 0 ? (
        <div className="no-servers">
          <p>No servers configured.</p>
          <p>Add a server or scan a QR code to get started.</p>
        </div>
      ) : (
        <div className="server-select-list">
          {servers.map(server => (
            <div
              key={server.id}
              className={`server-select-card ${connecting === server.id ? 'connecting' : ''}`}
              onClick={() => handleConnect(server)}
            >
              <div className="server-status-indicator">
                {refreshing ? (
                  <span className="status-dot testing" title="Checking..."></span>
                ) : statuses[server.id] === 'online' ? (
                  <span className="status-dot connected" title="Online"></span>
                ) : (
                  <span className="status-dot error" title="Offline"></span>
                )}
              </div>

              <div className="server-select-info">
                <div className="server-select-name">{server.name}</div>
                <div className="server-select-url">{server.url}</div>
              </div>

              <div className="server-select-actions">
                {connecting === server.id ? (
                  <span className="connecting-text">Connecting...</span>
                ) : (
                  <>
                    <button
                      className="connect-btn"
                      onClick={(e) => { e.stopPropagation(); handleConnect(server) }}
                    >
                      Connect
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
      )}

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
