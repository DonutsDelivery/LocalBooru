import { useEditionCapabilities } from '../contexts/edition'
// SPDX-License-Identifier: GPL-3.0-only
import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { getConnectedHubs } from '../api'
import './SharedCollectionsPanel.css'

export default function SharedCollectionsPanel({ mediaType, galleryReturnTo, onClose }) {
  const edition = useEditionCapabilities()
  const [hubs, setHubs] = useState([])
  const [error, setError] = useState('')
  useEffect(() => {
    if (!edition.community_hubs) return
    let active = true
    getConnectedHubs().then(result => {
      if (active) setHubs(result.hubs || [])
    }).catch(() => { if (active) setError('Connected communities could not be loaded.') })
    return () => { active = false }
  }, [edition.community_hubs])
  if (!edition.community_hubs) return null
  return <section className="sidebar-section shared-collections-panel" aria-label="Shared collections">
    <h3>Shared collections</h3>
    {error && <p role="status">{error}</p>}
    {!error && !hubs.length && <p>No connected communities.</p>}
    {hubs.map(hub => <NavLink
      key={hub.source_id}
      to={`/shared/${encodeURIComponent(hub.source_id)}?media_type=${mediaType}${hub.enabled ? '' : '&creator_view=offline&offline_only=true'}`}
      state={{ galleryReturnTo }}
      className="media-collection-link shared-collection-link"
      onClick={onClose}
    >
      <span>{hub.display_name}</span>
      {!hub.enabled && <small>Connection disabled · Browse offline copies</small>}
    </NavLink>)}
    <NavLink to="/settings/fediverse" state={{ returnTo: galleryReturnTo }} className="media-collection-link" onClick={onClose}>＋ Add community</NavLink>
  </section>
}
