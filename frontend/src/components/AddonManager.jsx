import { useState, useEffect, useRef } from 'react'
import {
  getAddons,
  installAddon,
  probeAddon,
  startAddon,
  stopAddon,
  uninstallAddon,
  updateAddon,
} from '../api'
import { invalidateAddonCache } from '../hooks/useAddonStatus'
import { formatBackend, formatLadaReadiness, normalizeAddonStatus } from './ladaRuntimeStatus'
import './AddonManager.css'

export default function AddonManager() {
  const [addons, setAddons] = useState([])
  const [loading, setLoading] = useState(true)
  const [actionInProgress, setActionInProgress] = useState({}) // { addonId: 'installing' | 'starting' | 'stopping' | 'uninstalling' }
  const pollRef = useRef(null)

  useEffect(() => {
    let active = true
    getAddons()
      .then((data) => {
        if (active) setAddons(data.addons || [])
      })
      .catch((error) => console.error('Failed to load addons:', error))
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  // Poll while any addon is in a transitional state
  useEffect(() => {
    const isTransitional = (addon) => {
      const lifecycle = normalizeAddonStatus(addon.status).key
      return lifecycle === 'starting' || lifecycle === 'stopping' || lifecycle === 'repairing' ||
        formatLadaReadiness(addon.readiness)?.transitional === true
    }
    const hasTransitional = addons.some(isTransitional) || Object.keys(actionInProgress).length > 0
    if (hasTransitional && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        try {
          const data = await getAddons()
          setAddons(data.addons || [])
          // Stop polling if nothing is transitional anymore
          const stillTransitional = (data.addons || []).some(isTransitional)
          if (!stillTransitional && Object.keys(actionInProgress).length === 0) {
            clearInterval(pollRef.current)
            pollRef.current = null
          }
        } catch (e) {
          console.error('Addon poll error:', e)
        }
      }, 2000)
    } else if (!hasTransitional && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [addons, actionInProgress])


  async function handleAction(addonId, action) {
    setActionInProgress(prev => ({ ...prev, [addonId]: action }))
    try {
      if (action === 'install') await installAddon(addonId)
      else if (action === 'uninstall') await uninstallAddon(addonId)
      else if (action === 'start') await startAddon(addonId)
      else if (action === 'stop') await stopAddon(addonId)
      else if (action === 'repair') await updateAddon(addonId)
      else if (action === 'probe') await probeAddon(addonId)
      // Refresh after action
      const data = await getAddons()
      setAddons(data.addons || [])
      invalidateAddonCache()
      window.dispatchEvent(new CustomEvent('localbooru-addons-changed'))
    } catch (e) {
      console.error(`Addon ${action} failed:`, e)
    }
    setActionInProgress(prev => {
      const next = { ...prev }
      delete next[addonId]
      return next
    })
  }

  function getStatusBadgeClass(status) {
    switch (status) {
      case 'running': return 'addon-status running'
      case 'starting': return 'addon-status starting'
      case 'installed':
      case 'stopped': return 'addon-status installed'
      case 'error': return 'addon-status error'
      default: return 'addon-status not-installed'
    }
  }

  function getReadinessBadgeClass(readiness) {
    return readiness ? `addon-status ${readiness.badge}` : null
  }

  function getStatusLabel(status) {
    switch (status) {
      case 'not_installed': return 'Not Installed'
      case 'installed': return 'Installed'
      case 'starting': return 'Starting...'
      case 'repairing': return 'Changing...'
      case 'running': return 'Running'
      case 'stopped': return 'Stopped'
      case 'error': return 'Error'
      default: return status
    }
  }

  if (loading) {
    return (
      <section className="optical-flow-settings">
        <h2>Addons</h2>
        <p className="setting-description">Loading...</p>
      </section>
    )
  }

  return (
    <section className="optical-flow-settings addon-manager">
      <h2>Addons</h2>
      <p className="setting-description">
        Manage optional features and processing add-ons.
      </p>

      <div className="addon-grid">
        {addons.map(addon => {
          const busy = actionInProgress[addon.id]
          const lifecycle = normalizeAddonStatus(addon.status)
          const status = lifecycle.key
          const readiness = formatLadaReadiness(addon.readiness)
          const managedBundle = addon.installation === 'managed_bundle'
          const showReadiness = readiness && !['error', 'repairing', 'stopping'].includes(status)
          const statusClass = getReadinessBadgeClass(showReadiness) || getStatusBadgeClass(status)
          const statusLabel = showReadiness?.status || getStatusLabel(status)
          return (
            <div key={addon.id} className="addon-card">
              <div className="addon-card-header">
                <h3>{addon.name}</h3>
                <span className={statusClass}>
                  {statusLabel}
                </span>
              </div>
              <p className="addon-description">{addon.description}</p>
              {showReadiness && (
                <div className="addon-meta lada-readiness">
                  {showReadiness.activeBackend ? (
                    <span>Active backend: {formatBackend(showReadiness.activeBackend)}</span>
                  ) : (
                    <span>Preferred backend: {formatBackend(showReadiness.configuredBackend)}</span>
                  )}
                  {showReadiness.reason && <span>{showReadiness.reason}</span>}
                </div>
              )}
              {!showReadiness && lifecycle.reason && (
                <div className="addon-meta"><span>{lifecycle.reason}</span></div>
              )}
              {addon.port != null && <div className="addon-meta">
                <span className="addon-port">Port {addon.port}</span>
              </div>}
              <div className="addon-actions">
                {managedBundle && addon.installed && (
                  <>
                    <button
                      onClick={() => handleAction(addon.id, 'probe')}
                      disabled={!!busy || readiness?.transitional}
                      className="addon-btn start"
                    >
                      {busy === 'probe' || readiness?.transitional ? 'Verifying...' : 'Verify runtime'}
                    </button>
                    <button
                      onClick={() => {
                        if (!confirm(`Uninstall ${addon.name}?`)) return
                        handleAction(addon.id, 'uninstall')
                      }}
                      disabled={!!busy}
                      className="addon-btn uninstall"
                    >
                      {busy === 'uninstall' ? 'Removing...' : 'Uninstall'}
                    </button>
                  </>
                )}
                {managedBundle && !addon.installed && readiness?.key === 'not_installed' && (
                  <button disabled className="addon-btn install" title="Use the managed LADA installer">
                    Managed install required
                  </button>
                )}
                {managedBundle && !addon.installed && readiness?.key !== 'not_installed' && (
                  <button
                    onClick={() => {
                      if (!confirm(`Remove the incomplete ${addon.name} installation?`)) return
                      handleAction(addon.id, 'uninstall')
                    }}
                    disabled={!!busy || readiness?.transitional}
                    className="addon-btn uninstall"
                  >
                    {busy === 'uninstall' ? 'Removing...' : 'Remove incomplete install'}
                  </button>
                )}
                {!managedBundle && status === 'not_installed' && (
                  <button
                    onClick={() => handleAction(addon.id, 'install')}
                    disabled={!!busy}
                    className="addon-btn install"
                  >
                    {busy === 'install' ? 'Installing...' : 'Install'}
                  </button>
                )}
                {!managedBundle && (status === 'installed' || status === 'stopped') && addon.requires_start && (
                  <>
                    <button
                      onClick={() => handleAction(addon.id, 'start')}
                      disabled={!!busy}
                      className="addon-btn start"
                    >
                      {busy === 'start' ? 'Starting...' : 'Start'}
                    </button>
                    <button
                      onClick={() => handleAction(addon.id, 'repair')}
                      disabled={!!busy}
                      className="addon-btn"
                    >
                      {busy === 'repair' ? 'Updating...' : 'Repair'}
                    </button>
                    <button
                      onClick={() => {
                        if (!confirm(`Uninstall ${addon.name}? This will remove its virtual environment.`)) return
                        handleAction(addon.id, 'uninstall')
                      }}
                      disabled={!!busy}
                      className="addon-btn uninstall"
                    >
                      {busy === 'uninstall' ? 'Removing...' : 'Uninstall'}
                    </button>
                  </>
                )}
                {!managedBundle && (status === 'installed' || status === 'stopped') && !addon.requires_start && (
                  <button
                    onClick={() => handleAction(addon.id, 'uninstall')}
                    disabled={!!busy}
                    className="addon-btn uninstall"
                  >
                    {busy === 'uninstall' ? 'Removing...' : 'Uninstall'}
                  </button>
                )}
                {!managedBundle && status === 'running' && (
                  <button
                    onClick={() => handleAction(addon.id, 'stop')}
                    disabled={!!busy}
                    className="addon-btn stop"
                  >
                    {busy === 'stop' ? 'Stopping...' : 'Stop'}
                  </button>
                )}
                {!managedBundle && status === 'starting' && (
                  <button disabled className="addon-btn starting">
                    <span className="spinner-small"></span>
                    Starting...
                  </button>
                )}
                {!managedBundle && status === 'error' && (
                  <>
                    <button
                      onClick={() => handleAction(addon.id, 'start')}
                      disabled={!!busy}
                      className="addon-btn start"
                    >
                      Retry
                    </button>
                    <button
                      onClick={() => {
                        if (!confirm(`Uninstall ${addon.name}?`)) return
                        handleAction(addon.id, 'uninstall')
                      }}
                      disabled={!!busy}
                      className="addon-btn uninstall"
                    >
                      Uninstall
                    </button>
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
