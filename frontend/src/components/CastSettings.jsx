import { useEffect, useState } from 'react'
import { getAddon } from '../api'
import './OpticalFlowSettings.css'

export default function CastSettings() {
  const [addon, setAddon] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getAddon('cast')
      .then((data) => setAddon(data.addon))
      .catch((error) => console.error('Failed to load cast add-on status:', error))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <section className="optical-flow-settings">
        <h2>Chromecast & DLNA</h2>
        <p className="setting-description">Loading...</p>
      </section>
    )
  }

  return (
    <section className="optical-flow-settings">
      <h2>Chromecast & DLNA</h2>
      <p className="setting-description">
        Cast devices are discovered and selected from the media viewer. Install and start this add-on from Addons before casting.
      </p>
      <div className="backend-status">
        <strong>Status:</strong>
        <span className={`backend-badge ${addon?.status === 'running' ? 'available' : 'unavailable'}`}>
          {addon?.status || 'Unavailable'}
        </span>
      </div>
    </section>
  )
}
