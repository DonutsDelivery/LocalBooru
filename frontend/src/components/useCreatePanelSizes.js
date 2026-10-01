import { useEffect, useRef, useState } from 'react'

const STORAGE_KEY = 'donut-create-panel-sizes'
const clamp = (value, min, max) => Math.min(Math.max(value, min), max)

export default function useCreatePanelSizes(expanded, expand) {
  const workspace = useRef(null)
  const drag = useRef(null)
  const [bounds, setBounds] = useState({ width: 0, height: 0 })
  const [preferences, setPreferences] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}
      return Object.fromEntries(['left', 'right', 'lower'].filter(key => Number.isFinite(saved[key])).map(key => [key, saved[key]]))
    } catch { return {} }
  })
  const [resizing, setResizing] = useState(false)
  const [desktop, setDesktop] = useState(() => typeof window !== 'undefined' && (window.matchMedia ? window.matchMedia('(min-width: 941px)').matches : window.innerWidth > 940))
  useEffect(() => {
    const measure = () => {
      setDesktop(window.matchMedia ? window.matchMedia('(min-width: 941px)').matches : window.innerWidth > 940)
      const rect = workspace.current?.getBoundingClientRect()
      if (rect) setBounds({ width: rect.width, height: rect.height })
    }
    measure()
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(measure) : null
    if (workspace.current) observer?.observe(workspace.current)
    window.addEventListener('resize', measure)
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure) }
  }, [])
  const sideMax = Math.max(200, bounds.width - 520)
  const left = clamp(preferences.left ?? (bounds.width > 1280 ? 300 : 260), 200, sideMax)
  const right = clamp(preferences.right ?? (bounds.width > 1280 ? 280 : 245), 200, Math.max(200, bounds.width - left - 320))
  const sizes = { left, right, lower: clamp(preferences.lower ?? Math.min(230, Math.max(150, bounds.height * 0.2)), 120, Math.max(120, bounds.height - 240)) }
  const limits = { left: [200, Math.max(200, bounds.width - right - 320)], right: [200, Math.max(200, bounds.width - left - 320)], lower: [120, Math.max(120, bounds.height - 240)] }
  function remember(next) {
    setPreferences(next)
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)) } catch { /* Resizing still works when storage is unavailable. */ }
  }
  function separator(key, label) {
    const horizontal = key === 'lower'
    const anchored = horizontal ? preferences : { ...preferences, [key === 'left' ? 'right' : 'left']: sizes[key === 'left' ? 'right' : 'left'] }
    function finish(event, cancel = false) {
      const current = drag.current
      if (!current || current.key !== key || current.pointerId !== event.pointerId) return
      drag.current = null
      setResizing(false)
      if (cancel) setPreferences(current.before)
      else remember(current.next)
      if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    }
    return {
      role: 'separator', tabIndex: desktop ? 0 : -1, 'aria-label': label,
      'aria-orientation': horizontal ? 'horizontal' : 'vertical',
      'aria-valuemin': limits[key][0], 'aria-valuemax': Math.round(limits[key][1]), 'aria-valuenow': Math.round(sizes[key]),
      'aria-valuetext': Math.round(sizes[key]) + ' pixels',
      title: label + ' · Drag or use arrow keys · Double-click to reset',
      onPointerDown(event) {
        if (!desktop || event.button !== 0 || drag.current) return
        event.preventDefault()
        event.currentTarget.focus()
        event.currentTarget.setPointerCapture(event.pointerId)
        if (horizontal) expand(true)
        drag.current = { key, pointerId: event.pointerId, start: horizontal ? event.clientY : event.clientX, size: sizes[key], before: preferences, anchored, limits: limits[key], next: anchored }
        setResizing(true)
      },
      onPointerMove(event) {
        const current = drag.current
        if (!current || current.key !== key || current.pointerId !== event.pointerId) return
        const delta = (horizontal ? event.clientY : event.clientX) - current.start
        const value = clamp(current.size + delta * (key === 'left' ? 1 : -1), ...current.limits)
        current.next = { ...current.anchored, [key]: value }
        setPreferences(current.next)
      },
      onPointerUp: finish,
      onPointerCancel: event => finish(event, true),
      onLostPointerCapture: event => finish(event),
      onKeyDown(event) {
        if (!desktop) return
        const change = horizontal ? { ArrowUp: 1, ArrowDown: -1 } : key === 'left' ? { ArrowRight: 1, ArrowLeft: -1 } : { ArrowLeft: 1, ArrowRight: -1 }
        let next = event.key === 'Home' ? limits[key][0] : event.key === 'End' ? limits[key][1] : change[event.key] ? clamp(sizes[key] + change[event.key] * (event.shiftKey ? 50 : 10), ...limits[key]) : null
        if (next === null) return
        event.preventDefault()
        if (horizontal) expand(true)
        remember({ ...anchored, [key]: next })
      },
      onDoubleClick() {
        const next = { ...preferences }; delete next[key]; remember(next)
      },
    }
  }
  return { workspace, resizing, desktop, separator, style: desktop ? { '--create-left-size': left + 'px', '--create-right-size': right + 'px', '--create-lower-size': expanded ? sizes.lower + 'px' : 'auto', '--create-lower-handle': expanded ? sizes.lower + 'px' : '44px' } : {} }
}
