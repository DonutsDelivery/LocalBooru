import { useState, useEffect, useRef } from 'react'
import { getAddons, installAddon, uninstallAddon, startAddon, stopAddon, updateAddon } from '../api'
import { invalidateAddonCache } from '../hooks/useAddonStatus'
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
    const hasTransitional = addons.some(a => a.status === 'starting') || Object.keys(actionInProgress).length > 0
    if (hasTransitional && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        try {
          const data = await getAddons()
          setAddons(data.addons || [])
          // Stop polling if nothing is transitional anymore
          const stillTransitional = (data.addons || []).some(a => a.status === 'starting')
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

  function getStatusLabel(status) {
    switch (status) {
      case 'not_installed': return 'Not Installed'
      case 'installed': return 'Installed'
      case 'starting': return 'Starting...'
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
          return (
            <div key={addon.id} className="addon-card">
              <div className="addon-card-header">
                <h3>{addon.name}</h3>
                <span className={getStatusBadgeClass(addon.status)}>
                  {getStatusLabel(addon.status)}
                </span>
              </div>
              <p className="addon-description">{addon.description}</p>
              {addon.port != null && <div className="addon-meta">
                <span className="addon-port">Port {addon.port}</span>
              </div>}
              <div className="addon-actions">
                {addon.status === 'not_installed' && (
                  <button
                    onClick={() => handleAction(addon.id, 'install')}
                    disabled={!!busy}
                    className="addon-btn install"
                  >
                    {busy === 'install' ? 'Installing...' : 'Install'}
                  </button>
                )}
                {(addon.status === 'installed' || addon.status === 'stopped') && addon.requires_start && (
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
                {(addon.status === 'installed' || addon.status === 'stopped') && !addon.requires_start && (
                  <button
                    onClick={() => handleAction(addon.id, 'uninstall')}
                    disabled={!!busy}
                    className="addon-btn uninstall"
                  >
                    {busy === 'uninstall' ? 'Removing...' : 'Uninstall'}
                  </button>
                )}
                {addon.status === 'running' && (
                  <button
                    onClick={() => handleAction(addon.id, 'stop')}
                    disabled={!!busy}
                    className="addon-btn stop"
                  >
                    {busy === 'stop' ? 'Stopping...' : 'Stop'}
                  </button>
                )}
                {addon.status === 'starting' && (
                  <button disabled className="addon-btn starting">
                    <span className="spinner-small"></span>
                    Starting...
                  </button>
                )}
                {addon.status === 'error' && (
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
