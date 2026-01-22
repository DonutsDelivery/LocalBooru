import { useState, useEffect } from 'react'
import {
  getNetworkConfig,
  updateNetworkConfig,
  testPort,
  discoverUPnP,
  openUPnPPort,
  closeUPnPPort
} from '../api'
import './NetworkSettings.css'

export default function NetworkSettings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [portTestResult, setPortTestResult] = useState(null)
  const [upnpStatus, setUpnpStatus] = useState(null)
  const [upnpLoading, setUpnpLoading] = useState(false)
  const [restartRequired, setRestartRequired] = useState(false)

  // Local form state
  const [localEnabled, setLocalEnabled] = useState(false)
  const [publicEnabled, setPublicEnabled] = useState(false)
  const [localPort, setLocalPort] = useState(8790)
  const [publicPort, setPublicPort] = useState(8791)
  const [authLevel, setAuthLevel] = useState('none')

  useEffect(() => {
    loadConfig()
  }, [])

  async function loadConfig() {
    try {
      setLoading(true)
      const data = await getNetworkConfig()
      setConfig(data)
      // Initialize form state from config
      setLocalEnabled(data.settings.local_network_enabled || false)
      setPublicEnabled(data.settings.public_network_enabled || false)
      setLocalPort(data.settings.local_port || 8790)
      setPublicPort(data.settings.public_port || 8791)
      setAuthLevel(data.settings.auth_required_level || 'none')
      setUpnpStatus(data.upnp_status)
    } catch (err) {
      console.error('Failed to load network config:', err)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    try {
      setSaving(true)
      const result = await updateNetworkConfig({
        local_network_enabled: localEnabled,
        public_network_enabled: publicEnabled,
        local_port: localPort,
        public_port: publicPort,
        auth_required_level: authLevel
      })
      if (result.restart_required) {
        setRestartRequired(true)
      }
      // Reload config
      await loadConfig()
    } catch (err) {
      console.error('Failed to save network config:', err)
    } finally {
      setSaving(false)
    }
  }

  async function handleTestPort(port) {
    try {
      const result = await testPort(port)
      setPortTestResult({ port, ...result })
    } catch (err) {
      setPortTestResult({ port, available: false, error: err.message })
    }
  }

  async function handleDiscoverUPnP() {
    try {
      setUpnpLoading(true)
      const result = await discoverUPnP()
      setUpnpStatus(result)
    } catch (err) {
      setUpnpStatus({ found: false, error: err.message })
    } finally {
      setUpnpLoading(false)
    }
  }

  async function handleOpenPort() {
    try {
      setUpnpLoading(true)
      const port = publicEnabled ? publicPort : localPort
      const result = await openUPnPPort(port)
      if (result.success) {
        // Refresh UPnP status
        await handleDiscoverUPnP()
      } else {
        alert(`Failed to open port: ${result.error}`)
      }
    } catch (err) {
      alert(`Failed to open port: ${err.message}`)
    } finally {
      setUpnpLoading(false)
    }
  }

  async function handleClosePort(port) {
    try {
      setUpnpLoading(true)
      await closeUPnPPort(port)
      await handleDiscoverUPnP()
    } catch (err) {
      alert(`Failed to close port: ${err.message}`)
    } finally {
      setUpnpLoading(false)
    }
  }

  function handleRestart() {
    // Send message to Electron to restart backend
    if (window.electronAPI?.restartBackend) {
      window.electronAPI.restartBackend()
      setRestartRequired(false)
    } else {
      alert('Please restart the application manually for changes to take effect.')
    }
  }

  if (loading) {
    return (
      <div className="network-settings loading">
        <div className="spinner" />
        <span>Loading network settings...</span>
      </div>
    )
  }

  const hasChanges =
    localEnabled !== (config?.settings?.local_network_enabled || false) ||
    publicEnabled !== (config?.settings?.public_network_enabled || false) ||
    localPort !== (config?.settings?.local_port || 8790) ||
    publicPort !== (config?.settings?.public_port || 8791) ||
    authLevel !== (config?.settings?.auth_required_level || 'none')

  return (
    <div className="network-settings">
      <h2>Network Access</h2>
      <p className="settings-description">
        Configure how LocalBooru can be accessed from other devices on your network or the internet.
        Remote access is always read-only for security.
      </p>

      {restartRequired && (
        <div className="restart-notice">
          <span>Restart required for changes to take effect</span>
          <button className="restart-btn" onClick={handleRestart}>
            Restart Now
          </button>
        </div>
      )}

      {/* Local Network Section */}
      <section className="settings-section">
        <h3>Local Network Access</h3>
        <p className="section-description">
          Allow devices on your local network (LAN) to browse your library.
        </p>

        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={localEnabled}
              onChange={(e) => setLocalEnabled(e.target.checked)}
            />
            <span>Enable local network access</span>
          </label>
        </div>

        {config?.local_ip && (
          <div className="info-row">
            <span className="info-label">Your Local IP:</span>
            <code className="info-value">{config.local_ip}</code>
          </div>
        )}

        <div className="setting-row">
          <label>
            <span>Local Port:</span>
            <input
              type="number"
              value={localPort}
              onChange={(e) => setLocalPort(parseInt(e.target.value) || 8790)}
              min="1024"
              max="65535"
            />
          </label>
          <button
            className="test-btn"
            onClick={() => handleTestPort(localPort)}
          >
            Test Port
          </button>
        </div>

        {portTestResult && portTestResult.port === localPort && (
          <div className={`port-test-result ${portTestResult.available ? 'success' : 'error'}`}>
            {portTestResult.available
              ? 'Port is available'
              : `Port unavailable: ${portTestResult.error}`}
          </div>
        )}

        {config?.local_ip && (
          <div className="access-url">
            <span>Access URL:</span>
            <a href={`http://${config.local_ip}:${localPort}`} target="_blank" rel="noopener noreferrer">
              http://{config.local_ip}:{localPort}
            </a>
          </div>
        )}
      </section>

      {/* Public Network Section */}
      <section className="settings-section">
        <h3>Public Internet Access</h3>
        <p className="section-description">
          Allow access from the internet. Requires port forwarding on your router.
        </p>

        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={publicEnabled}
              onChange={(e) => setPublicEnabled(e.target.checked)}
            />
            <span>Enable public internet access</span>
          </label>
        </div>

        <div className="setting-row">
          <label>
            <span>Public Port:</span>
            <input
              type="number"
              value={publicPort}
              onChange={(e) => setPublicPort(parseInt(e.target.value) || 8791)}
              min="1024"
              max="65535"
            />
          </label>
          <button
            className="test-btn"
            onClick={() => handleTestPort(publicPort)}
          >
            Test Port
          </button>
        </div>

        {portTestResult && portTestResult.port === publicPort && (
          <div className={`port-test-result ${portTestResult.available ? 'success' : 'error'}`}>
            {portTestResult.available
              ? 'Port is available'
              : `Port unavailable: ${portTestResult.error}`}
          </div>
        )}

        {/* UPnP Section */}
        <div className="upnp-section">
          <h4>UPnP Port Forwarding</h4>
          <p className="section-description">
            Automatically configure your router to forward ports (if supported).
          </p>

          <div className="upnp-actions">
            <button
              className="upnp-btn"
              onClick={handleDiscoverUPnP}
              disabled={upnpLoading}
            >
              {upnpLoading ? 'Discovering...' : 'Discover Gateway'}
            </button>

            {upnpStatus?.found && (
              <button
                className="upnp-btn primary"
                onClick={handleOpenPort}
                disabled={upnpLoading}
              >
                Open Port {publicEnabled ? publicPort : localPort}
              </button>
            )}
          </div>

          {upnpStatus && (
            <div className={`upnp-status ${upnpStatus.found ? 'success' : 'error'}`}>
              {upnpStatus.found ? (
                <>
                  <div>Gateway found: {upnpStatus.gateway}</div>
                  {upnpStatus.external_ip && (
                    <div>External IP: <code>{upnpStatus.external_ip}</code></div>
                  )}
                </>
              ) : (
                <div>No UPnP gateway found: {upnpStatus.error}</div>
              )}
            </div>
          )}

          {publicEnabled && upnpStatus?.external_ip && (
            <div className="access-url">
              <span>Public URL:</span>
              <code>http://{upnpStatus.external_ip}:{publicPort}</code>
            </div>
          )}
        </div>
      </section>

      {/* Authentication Section */}
      <section className="settings-section">
        <h3>Authentication</h3>
        <p className="section-description">
          Require login for remote access. Manage users in the Users tab.
        </p>

        <div className="setting-row">
          <label>
            <span>Require authentication for:</span>
            <select value={authLevel} onChange={(e) => setAuthLevel(e.target.value)}>
              <option value="none">No one (open access)</option>
              <option value="public">Public internet only</option>
              <option value="local_network">Local network and public</option>
              <option value="always">Everyone (including localhost)</option>
            </select>
          </label>
        </div>
      </section>

      {/* Access Restrictions Info */}
      <section className="settings-section info-section">
        <h3>Remote Access Restrictions</h3>
        <div className="restrictions-list">
          <div className="restriction allowed">
            <span className="icon">&#10003;</span>
            <span>Browse images and tags</span>
          </div>
          <div className="restriction allowed">
            <span className="icon">&#10003;</span>
            <span>View image details and metadata</span>
          </div>
          <div className="restriction allowed">
            <span className="icon">&#10003;</span>
            <span>Download images</span>
          </div>
          <div className="restriction denied">
            <span className="icon">&#10007;</span>
            <span>Add or remove directories</span>
          </div>
          <div className="restriction denied">
            <span className="icon">&#10007;</span>
            <span>Delete images</span>
          </div>
          <div className="restriction denied">
            <span className="icon">&#10007;</span>
            <span>Modify tags or ratings</span>
          </div>
          <div className="restriction denied">
            <span className="icon">&#10007;</span>
            <span>Change settings</span>
          </div>
        </div>
      </section>

      {/* Save Button */}
      <div className="settings-actions">
        <button
          className="save-btn"
          onClick={handleSave}
          disabled={saving || !hasChanges}
        >
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  )
}
