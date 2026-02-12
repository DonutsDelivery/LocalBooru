import { useState, useEffect } from 'react'
import { getCastConfig, updateCastConfig } from '../api'
import './OpticalFlowSettings.css'

export default function CastSettings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  async function loadConfig() {
    try {
      setLoading(true)
      const data = await getCastConfig()
      setConfig(data)
    } catch (e) {
      console.error('Failed to load cast config:', e)
    }
    setLoading(false)
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
          Install missing dependencies:
          <code style={{ display: 'block', marginTop: '4px', padding: '4px 8px', background: 'rgba(0,0,0,0.2)', borderRadius: '4px' }}>
            pip install {missingDeps.join(' ')}
          </code>
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
