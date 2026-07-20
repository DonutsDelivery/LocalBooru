import { useEffect, useRef } from 'react'
import './ContextMenu.css'

function ContextMenu({ items, position, onClose }) {
  const menuRef = useRef(null)

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        onClose()
      }
    }
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose()
    }
    const handleScroll = () => onClose()

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleEscape)
    window.addEventListener('scroll', handleScroll, true)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
      window.removeEventListener('scroll', handleScroll, true)
    }
  }, [onClose])

  useEffect(() => {
    if (!menuRef.current) return

    const rect = menuRef.current.getBoundingClientRect()
    const rootStyles = getComputedStyle(document.documentElement)
    const inset = (name) => parseFloat(rootStyles.getPropertyValue(name)) || 0
    const edgeGap = 8
    const minLeft = inset('--safe-left') + edgeGap
    const minTop = inset('--safe-top') + edgeGap
    const maxLeft = Math.max(minLeft, window.innerWidth - inset('--safe-right') - rect.width - edgeGap)
    const maxTop = Math.max(minTop, window.innerHeight - inset('--safe-bottom') - rect.height - edgeGap)

    menuRef.current.style.left = `${Math.min(Math.max(position.x, minLeft), maxLeft)}px`
    menuRef.current.style.top = `${Math.min(Math.max(position.y, minTop), maxTop)}px`
  }, [position])

  return (
    <div
      ref={menuRef}
      className="context-menu"
      style={{ left: position.x, top: position.y }}
    >
      {items.map((item, i) => (
        item.separator ? (
          <div key={i} className="context-menu-separator" />
        ) : (
          <button
            key={i}
            className={`context-menu-item ${item.disabled ? 'disabled' : ''}`}
            onClick={() => {
              if (!item.disabled) {
                item.onClick()
                onClose()
              }
            }}
          >
            {item.icon && <span className="context-menu-icon">{item.icon}</span>}
            <span className="context-menu-label">{item.label}</span>
          </button>
        )
      ))}
    </div>
  )
}

export default ContextMenu
