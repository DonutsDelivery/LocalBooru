import { NavLink } from 'react-router-dom'
import MediaSectionsNav from './MediaSectionsNav'
import './Sidebar.css'

export default function SidebarNavigation() {
  return <nav className="sidebar-nav" aria-label="Navigation">
    <MediaSectionsNav />
    <NavLink to="/online" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} title="Nodes & Fediverse" aria-label="Nodes & Fediverse">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="12" cy="5" r="2" /><circle cx="5" cy="18" r="2" /><circle cx="19" cy="18" r="2" />
        <path d="M10.8 6.7 6.2 16.2M13.2 6.7l4.6 9.5M7 18h10" />
      </svg>
    </NavLink>
    <NavLink to="/directories" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} title="Directories" aria-label="Directories">
      <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z" />
      </svg>
    </NavLink>
    <NavLink to="/settings" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} title="Settings" aria-label="Settings">
      <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z" />
      </svg>
    </NavLink>
  </nav>
}
