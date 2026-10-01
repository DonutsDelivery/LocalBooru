import { useEffect, useRef, useState } from 'react'

// Scale is relative to the source image, so 100% means one CSS pixel per image pixel.
function clampView(view, scale, size, viewport) {
  const limitX = Math.max(0, (size.width * scale - viewport.width) / 2)
  const limitY = Math.max(0, (size.height * scale - viewport.height) / 2)
  return { ...view, x: Math.min(limitX, Math.max(-limitX, view.x)), y: Math.min(limitY, Math.max(-limitY, view.y)) }
}

export default function CreateResultPreview({ src, imageKey, alt, onError }) {
  const stage = useRef(null)
  const pointers = useRef(new Map())
  const pinch = useRef(null)
  const viewRef = useRef({ mode: 'fit', scale: 1, x: 0, y: 0 })
  const [view, setView] = useState(viewRef.current)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [viewport, setViewport] = useState({ width: 0, height: 0 })
  const fit = size.width && size.height && viewport.width && viewport.height ? Math.min(viewport.width / size.width, viewport.height / size.height) : 1
  const scale = view.mode === 'fit' ? fit : view.scale
  const minimum = Math.min(0.1, fit)
  const maximum = Math.max(8, fit)

  function update(next) {
    const currentScale = next.mode === 'fit' ? fit : next.scale
    const bounded = clampView(next, currentScale, size, viewport)
    viewRef.current = bounded
    setView(bounded)
  }

  useEffect(() => {
    const initial = { mode: 'fit', scale: 1, x: 0, y: 0 }
    viewRef.current = initial
    setView(initial)
    pointers.current.clear()
    pinch.current = null
  }, [imageKey])

  useEffect(() => {
    const element = stage.current
    const measure = () => setViewport({ width: element.clientWidth, height: element.clientHeight })
    measure()
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(measure) : null
    observer?.observe(element)
    window.addEventListener('resize', measure)
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure) }
  }, [])

  useEffect(() => {
    const current = viewRef.current
    const bounded = clampView(current, current.mode === 'fit' ? fit : current.scale, size, viewport)
    if (bounded.x !== current.x || bounded.y !== current.y) {
      viewRef.current = bounded
      setView(bounded)
    }
  }, [fit, size, viewport])

  function zoom(nextScale, clientX, clientY, previousX = clientX, previousY = clientY) {
    const current = viewRef.current
    const oldScale = current.mode === 'fit' ? fit : current.scale
    const next = Math.min(maximum, Math.max(minimum, nextScale))
    const rect = stage.current.getBoundingClientRect()
    const x = clientX == null ? 0 : clientX - rect.left - viewport.width / 2
    const y = clientY == null ? 0 : clientY - rect.top - viewport.height / 2
    const oldX = previousX == null ? 0 : previousX - rect.left - viewport.width / 2
    const oldY = previousY == null ? 0 : previousY - rect.top - viewport.height / 2
    update({ mode: 'zoom', scale: next, x: x - (oldX - current.x) * next / oldScale, y: y - (oldY - current.y) * next / oldScale })
  }

  // Native listener is non-passive so wheel zoom does not scroll the surrounding panel.
  const wheel = useRef(null)
  wheel.current = event => {
    event.preventDefault()
    zoom((viewRef.current.mode === 'fit' ? fit : viewRef.current.scale) * Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.003), event.clientX, event.clientY)
  }
  useEffect(() => {
    const element = stage.current
    const handler = event => wheel.current(event)
    element.addEventListener('wheel', handler, { passive: false })
    return () => element.removeEventListener('wheel', handler)
  }, [])

  function gesture() {
    const [a, b] = [...pointers.current.values()]
    return b ? { distance: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } : null
  }

  function move(event) {
    const previous = pointers.current.get(event.pointerId)
    if (!previous) return
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    const next = gesture()
    if (next && pinch.current?.distance > 0) {
      const currentScale = viewRef.current.mode === 'fit' ? fit : viewRef.current.scale
      zoom(currentScale * next.distance / pinch.current.distance, next.x, next.y, pinch.current.x, pinch.current.y)
    } else if (!next) {
      update({ ...viewRef.current, x: viewRef.current.x + event.clientX - previous.x, y: viewRef.current.y + event.clientY - previous.y })
    }
    pinch.current = next
  }

  function release(event) {
    pointers.current.delete(event.pointerId)
    pinch.current = gesture()
  }

  const loaded = !!size.width
  return <div className="create-result-viewer">
    <div className="create-preview-zoom" role="group" aria-label="Preview zoom">
      <button type="button" aria-pressed={view.mode === 'fit'} onClick={() => update({ mode: 'fit', scale: 1, x: 0, y: 0 })}>Fit</button>
      <button type="button" aria-pressed={view.mode !== 'fit' && scale === 1} disabled={!loaded} onClick={() => update({ mode: 'zoom', scale: 1, x: 0, y: 0 })}>100%</button>
      <button type="button" aria-label="Zoom out preview" disabled={!loaded || scale <= minimum} onClick={() => zoom(scale / 1.25)}>−</button>
      <output aria-label="Preview scale">{Math.round(scale * 100)}%</output>
      <button type="button" aria-label="Zoom in preview" disabled={!loaded || scale >= maximum} onClick={() => zoom(scale * 1.25)}>+</button>
    </div>
    <div ref={stage} className="create-result-viewport" tabIndex={0} aria-label="Image preview. Scroll to zoom, drag to pan."
      onDoubleClick={() => viewRef.current.mode === 'fit' ? update({ mode: 'zoom', scale: 1, x: 0, y: 0 }) : update({ mode: 'fit', scale: 1, x: 0, y: 0 })}
      onKeyDown={event => {
        if (['+', '=', '-', '0', '1', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) event.preventDefault()
        if (event.key === '+' || event.key === '=') zoom(scale * 1.25)
        else if (event.key === '-') zoom(scale / 1.25)
        else if (event.key === '0') update({ mode: 'fit', scale: 1, x: 0, y: 0 })
        else if (event.key === '1') update({ mode: 'zoom', scale: 1, x: 0, y: 0 })
        else if (event.key.startsWith('Arrow')) update({ ...viewRef.current, x: view.x + (event.key === 'ArrowLeft' ? 40 : event.key === 'ArrowRight' ? -40 : 0), y: view.y + (event.key === 'ArrowUp' ? 40 : event.key === 'ArrowDown' ? -40 : 0) })
      }}
      onPointerDown={event => {
        if (event.pointerType === 'mouse' && event.button !== 0) return
        pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
        pinch.current = gesture()
        event.currentTarget.setPointerCapture?.(event.pointerId)
      }} onPointerMove={move} onPointerUp={release} onPointerCancel={release} onLostPointerCapture={release}>
      <img src={src} alt={alt} draggable={false} onError={onError}
        onLoad={event => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
        style={loaded ? { width: size.width * scale, height: size.height * scale, left: `calc(50% + ${view.x}px)`, top: `calc(50% + ${view.y}px)` } : undefined}
        className={loaded ? 'create-result-image' : 'create-preview-image'} />
    </div>
  </div>
}
