// SPDX-License-Identifier: GPL-3.0-only
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getConnectedHubs } from '../api'
import './SharedCollectionsPanel.css'

export default function SharedCollectionsPanel({ mediaType, galleryReturnTo }) {
  const [hubs, setHubs] = useState([])
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    getConnectedHubs().then(result => {
      if (active) setHubs(result.hubs || [])
    }).catch(() => { if (active) setError('Connected communities could not be loaded.') })
    return () => { active = false }
  }, [])
  return <section className="shared-collections-panel" aria-label="Shared collections">
    <div className="shared-collections-heading"><h2>Shared collections</h2><Link to="/settings/fediverse" state={{ returnTo: galleryReturnTo }}>Add community</Link></div>
    {error && <p role="status">{error}</p>}
    {!error && !hubs.length && <p>Add DonutBooru or another community to browse its shared collection here.</p>}
    <div className="shared-collection-links">{hubs.map(hub => hub.enabled
      ? <Link key={hub.source_id} to={`/shared/${encodeURIComponent(hub.source_id)}?media_type=${mediaType}`} state={{ galleryReturnTo }}>
          <strong>{hub.display_name}</strong><span>Community · {hub.transport_mode === 'relayed' ? 'Relayed' : 'Standard'}</span>
        </Link>
      : <Link key={hub.source_id} to={`/shared/${encodeURIComponent(hub.source_id)}?media_type=${mediaType}&creator_view=offline&offline_only=true`} state={{ galleryReturnTo }}><strong>{hub.display_name}</strong><span>Connection disabled · Browse offline copies</span></Link>
    )}</div>
  </section>
}
