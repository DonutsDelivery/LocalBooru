import { NavLink } from 'react-router-dom'
import './MediaSectionsNav.css'

const sections = [
  { path: '/', label: 'Images', key: 'image' },
  { path: '/videos', label: 'Videos', key: 'video' },
  { path: '/music', label: 'Music', key: 'music' },
]

export default function MediaSectionsNav() {
  return (
    <nav className="media-sections-nav" aria-label="Media libraries">
      {sections.map(({ path, label, key }) => {
        const saved = sessionStorage.getItem(`donutMediaCenter_gallery_url_${key}`)
        const target = saved?.startsWith(path + '?') ? saved : path
        return (
          <NavLink
            key={key}
            to={target}
            end={path === '/'}
            className={({ isActive }) => `media-section-link ${isActive ? 'active' : ''}`}
          >
            {label}
          </NavLink>
        )
      })}
    </nav>
  )
}
