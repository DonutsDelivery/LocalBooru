import { Link, useLocation } from 'react-router-dom'

const sections = [
  { path: '/', label: 'Images', key: 'image', icon: <svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" strokeWidth="2"/><circle cx="8.5" cy="8.5" r="2.5"/><path d="M21 15l-5-5L5 21h14a2 2 0 002-2v-4z"/></svg> },
  { path: '/videos', label: 'Videos', key: 'video', icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m10 8 6 4-6 4z"/></svg> },
  { path: '/music', label: 'Music', key: 'music', icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18V5l12-2v13M9 9l12-2"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg> },
]

function galleryTarget(key, path) {
  if (key === 'music') return path
  const saved = sessionStorage.getItem(`donutMediaCenter_gallery_url_${key}`)
  if (!saved) return path
  const url = new URL(saved, window.location.origin)
  return url.origin === window.location.origin && url.pathname === path ? `${url.pathname}${url.search}` : path
}

export default function MediaSectionsNav() {
  const location = useLocation()
  const collectionType = new URLSearchParams(location.search).get('media_type') === 'video' ? 'video' : 'image'
  const currentType = location.pathname.startsWith('/collections') ? collectionType
    : location.pathname === '/videos' ? 'video'
    : location.pathname === '/music' ? 'music'
    : location.pathname === '/' ? 'image' : null

  return sections.map(({ path, label, key, icon }) => (
    <Link
      key={key}
      to={galleryTarget(key, path)}
      className={`nav-btn ${currentType === key ? 'active' : ''}`}
      title={label}
      aria-label={label}
      aria-current={currentType === key ? 'page' : undefined}
    >{icon}</Link>
  ))
}
