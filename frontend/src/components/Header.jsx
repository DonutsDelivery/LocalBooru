import { useState, useRef, useEffect } from 'react'
import UserSettings from './UserSettings'
import NotificationBell from './NotificationBell'
import './Header.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://donutbooru.com'

function Header({ user, onLogin, onLogout, onUserUpdate, spoofMode, onSpoofChange, realUser }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const menuRef = useRef()

  // Use realUser for mod checks (the actual logged in user)
  const isModerator = realUser && realUser.role !== 'user'

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleDiscordLogin = () => {
    window.location.href = `${API_URL}/auth/discord`
  }

  return (
    <header className={`header ${spoofMode ? 'spoofing' : ''}`}>
      {spoofMode && (
        <div className="spoof-banner">
          👁️ Viewing as: {spoofMode === 'anonymous' ? 'Logged Out User' : 'Regular User'}
          <button onClick={() => onSpoofChange(null)}>Exit</button>
        </div>
      )}
      <div className="header-left">
        <a href="/" className="logo">
          <img src="/logo.png" alt="DonutBooru" className="logo-image" />
        </a>

        <nav className={`nav-links ${mobileMenuOpen ? 'open' : ''}`}>
          <a href="/">Browse</a>
          <a href="/?tags=hall_of_fame">Hall of Fame</a>
        </nav>
      </div>

      <div className="header-right">
        {user && <NotificationBell user={user} />}
        {user ? (
          <div className="user-menu" ref={menuRef}>
            <button
              className="user-button"
              onClick={() => setMenuOpen(!menuOpen)}
            >
              <img
                src={user.avatar_url}
                alt=""
                className="user-avatar"
              />
              <span className="user-name">{user.username}</span>
              <svg className="dropdown-arrow" viewBox="0 0 24 24" fill="currentColor">
                <path d="M7 10l5 5 5-5z"/>
              </svg>
            </button>

            <div className={`user-dropdown ${menuOpen ? 'open' : ''}`}>
                <a href="/profile">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                  </svg>
                  My Profile
                </a>
                <a href="/uploads">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
                  </svg>
                  My Uploads
                </a>
                <a href="/favorites">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                  Favorites
                </a>
                <button onClick={() => { setSettingsOpen(true); setMenuOpen(false) }}>
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
                  </svg>
                  Settings
                </button>
                {isModerator && (
                  <>
                    <div className="dropdown-divider" />
                    {!spoofMode && (
                      <a href="/moderation" className="mod-link">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
                        </svg>
                        Moderation
                      </a>
                    )}
                    <div className="spoof-selector">
                      <label>
                        <svg viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                        </svg>
                        View as:
                      </label>
                      <select
                        value={spoofMode || 'none'}
                        onChange={(e) => onSpoofChange(e.target.value === 'none' ? null : e.target.value)}
                      >
                        <option value="none">Myself (Mod)</option>
                        <option value="regular">Regular User</option>
                        <option value="anonymous">Logged Out</option>
                      </select>
                    </div>
                  </>
                )}
                <div className="dropdown-divider" />
                <button onClick={onLogout} className="logout-button">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>
                  </svg>
                  Logout
                </button>
              </div>
          </div>
        ) : (
          <button className="login-button" onClick={handleDiscordLogin}>
            <svg className="discord-icon" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
            </svg>
            Login with Discord
          </button>
        )}

        <button
          className="mobile-menu-button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
          </svg>
        </button>
      </div>

      {settingsOpen && (
        <UserSettings
          user={user}
          onClose={() => setSettingsOpen(false)}
          onUserUpdate={onUserUpdate}
        />
      )}
    </header>
  )
}

export default Header
