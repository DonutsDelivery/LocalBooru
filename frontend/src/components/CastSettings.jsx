import { useState, useEffect, useRef } from 'react'
import { getCastConfig, updateCastConfig, installCastDeps } from '../api'
import './OpticalFlowSettings.css'

export default function CastSettings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [installing, setInstalling] = useState(false)
  const pollRef = useRef(null)

  useEffect(() => {
    loadConfig()
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  async function loadConfig() {
    try {
      setLoading(true)
      const data = await getCastConfig()
      setConfig(data)
      // Start polling if install is in progress
      if (data.installing) {
        setInstalling(true)
        startPolling()
      }
    } catch (e) {
      console.error('Failed to load cast config:', e)
    }
    setLoading(false)
  }

  function startPolling() {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const data = await getCastConfig()
        setConfig(data)
        if (!data.installing) {
          clearInterval(pollRef.current)
          pollRef.current = null
          setInstalling(false)
        }
      } catch (e) {
        console.error('Poll error:', e)
      }
    }, 2000)
  }

  async function handleInstall() {
    try {
      setInstalling(true)
      await installCastDeps()
      startPolling()
    } catch (e) {
      console.error('Failed to start install:', e)
      setInstalling(false)
    }
  }

  async function handleToggle(field, value) {
    setSaving(true)
    try {
      const result = await updateCastConfig({ [field]: value })
      setConfig(prev => ({ ...prev, ...result }))
    } catch (e) {
      console.error('Failed to save cast config:', e)
    }
    setSaving(false)
  }

  if (loading || !config) {
    return (
      <section className="optical-flow-settings">
        <h2>Chromecast & DLNA</h2>
        <p className="setting-description">Loading...</p>
      </section>
    )
  }

  const status = config.status || {}
  const hasDeps = status.pychromecast_installed && status.aiohttp_installed
  const missingDeps = []
  if (!status.pychromecast_installed) missingDeps.push('pychromecast')
  if (!status.upnp_installed) missingDeps.push('async-upnp-client')
  if (!status.aiohttp_installed) missingDeps.push('aiohttp')

  return (
    <section className="optical-flow-settings">
      <h2>Chromecast & DLNA</h2>
      <p className="setting-description">
        Cast videos to Chromecast, smart TVs, and DLNA/UPnP renderers on your local network.
      </p>

      {/* Dependency status */}
      <div className="deps-status" style={{ marginBottom: '12px' }}>
        <strong>Dependencies:</strong>
        <span className={`dep-badge ${status.pychromecast_installed ? 'installed' : 'missing'}`}>
          pychromecast: {status.pychromecast_installed ? '\u2713' : '\u2717'}
        </span>
        <span className={`dep-badge ${status.upnp_installed ? 'installed' : 'missing'}`}>
          async-upnp-client: {status.upnp_installed ? '\u2713' : '\u2717'}
        </span>
        <span className={`dep-badge ${status.aiohttp_installed ? 'installed' : 'missing'}`}>
          aiohttp: {status.aiohttp_installed ? '\u2713' : '\u2717'}
        </span>
      </div>

      {missingDeps.length > 0 && (
        <div style={{ marginBottom: '12px', padding: '8px 12px', background: 'rgba(255,107,107,0.1)', borderRadius: '6px', fontSize: '0.9em' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>Missing: {missingDeps.join(', ')}</span>
            <button
              onClick={handleInstall}
              disabled={installing}
              style={{
                padding: '4px 12px',
                background: installing ? 'var(--bg-tertiary)' : 'var(--accent)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: installing ? 'default' : 'pointer',
                fontSize: '0.85em',
              }}
            >
              {installing ? 'Installing...' : 'Install'}
            </button>
          </div>
          {config.install_progress && (
            <p style={{ margin: '4px 0 0', fontSize: '0.85em', opacity: 0.8 }}>
              {config.install_progress}
            </p>
          )}
        </div>
      )}

      {/* Enable toggle */}
      <div className="toggle-setting">
        <label>
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={(e) => handleToggle('enabled', e.target.checked)}
            disabled={saving || !hasDeps}
          />
          Enable casting
        </label>
      </div>

      {/* Port config (only show when enabled) */}
      {config.enabled && (
        <div className="optical-flow-field" style={{ marginTop: '12px' }}>
          <label>
            Cast media server port
            <input
              type="number"
              min="1024"
              max="65535"
              value={config.cast_media_port}
              onChange={(e) => handleToggle('cast_media_port', parseInt(e.target.value) || 8792)}
              disabled={saving}
              style={{ width: '80px', marginLeft: '8px' }}
            />
          </label>
          <p className="setting-description" style={{ marginTop: '4px', fontSize: '0.85em' }}>
            HTTP-only server for delivering media to cast devices. Only active during casting.
          </p>
        </div>
      )}
    </section>
  )
}
