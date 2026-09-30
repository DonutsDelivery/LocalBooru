import { useEffect, useRef, useState } from 'react'
import './CreateEditCanvas.css'

const EMPTY_MASK = () => ({ version: 1, strokes: [], inverted: false })

function readMask(value, reference) {
  try {
    const mask = typeof value === 'string' ? JSON.parse(value || 'null') : value
    return mask?.version === 1 && mask.image === reference && Array.isArray(mask.strokes)
      ? mask : { ...EMPTY_MASK(), image: reference }
  } catch { return { ...EMPTY_MASK(), image: reference } }
}

// Selection coordinates match DonutEditStudio's source-image mask format.
// The displayed image is never flattened into a new upload or sampled as pixels.
export default function CreateEditCanvas({ reference, imageUrl, maskData, enabled, selectedArea, disabled, onChange, onUpload }) {
  const canvas = useRef(null)
  const upload = useRef(null)
  const stroke = useRef(null)
  const [dimensions, setDimensions] = useState(null)
  const [tool, setTool] = useState('paint')
  const [size, setSize] = useState(7)
  const [document, setDocument] = useState(() => readMask(maskData, reference))
  const [imageError, setImageError] = useState('')
  const [areaEditing, setAreaEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const pendingChange = useRef(false)
  const change = useRef(onChange)
  useEffect(() => { change.current = onChange }, [onChange])
  const incomingMask = typeof maskData === 'string' ? maskData : JSON.stringify(maskData)

  useEffect(() => {
    stroke.current = null
    // The image URL is an external workflow change, so reset its loaded geometry.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDimensions(null)
    setImageError('')
  }, [imageUrl])

  useEffect(() => {
    if (stroke.current) return
    const next = readMask(incomingMask, reference)
    // Synchronize selections changed in the always-mounted advanced editor.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDocument(next)
    setAreaEditing(selectedArea ?? (next.strokes.length > 0 || next.inverted === true || !!next.outpaint))
  }, [incomingMask, reference, selectedArea])

  useEffect(() => {
    const surface = canvas.current
    if (!surface || !dimensions) return
    const scale = Math.min(1, 1600 / Math.max(dimensions.width, dimensions.height))
    surface.width = Math.max(1, Math.round(dimensions.width * scale))
    surface.height = Math.max(1, Math.round(dimensions.height * scale))
    const ctx = surface.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, surface.width, surface.height)
    for (const entry of document.strokes) {
      if (!Array.isArray(entry.points) || !entry.points.length) continue
      const diameter = Math.max(1, Number(entry.size) * Math.min(surface.width, surface.height))
      ctx.globalCompositeOperation = entry.erase ? 'destination-out' : 'source-over'
      ctx.fillStyle = '#00d99a'
      ctx.strokeStyle = '#00d99a'
      ctx.lineWidth = diameter
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      const points = entry.points.map(([x, y]) => [x * (surface.width - 1), y * (surface.height - 1)])
      if (entry.shape === 'rectangle' && points.length === 2) {
        const [[x1, y1], [x2, y2]] = points
        ctx.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1))
      } else {
        ctx.beginPath()
        points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y))
        ctx.stroke()
        for (const [x, y] of points) {
          ctx.beginPath()
          ctx.arc(x, y, diameter / 2, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
    if (document.inverted === true) {
      ctx.globalCompositeOperation = 'xor'
      ctx.fillStyle = '#00d99a'
      ctx.fillRect(0, 0, surface.width, surface.height)
    }
    ctx.globalCompositeOperation = 'source-over'
  }, [document, dimensions])

  const placement = !!document.outpaint || document.strokes.some(entry => entry.shape === 'rectangle')
  const locked = disabled || saving
  const canPaint = enabled && !locked && !!dimensions && !imageError && areaEditing && !placement

  function commitChange(patch) {
    if (pendingChange.current) return
    pendingChange.current = true
    setSaving(true)
    Promise.resolve().then(() => change.current(patch)).finally(() => {
      pendingChange.current = false
      setSaving(false)
    }).catch(() => {})
  }

  function publish(next, active) {
    setDocument(next)
    setAreaEditing(active)
    const saved = next.strokes.length > 0 || next.inverted === true || !!next.outpaint
    commitChange({ inpaint: active, editMask: saved || active ? next : '' })
  }

  function point(event) {
    const box = canvas.current.getBoundingClientRect()
    return [Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)),
      Math.min(1, Math.max(0, (event.clientY - box.top) / box.height))]
  }

  function start(event) {
    if (pendingChange.current || !canPaint || event.button !== 0 || stroke.current) return
    const usedPoints = document.strokes.reduce((total, entry) => total + (entry.points?.length || 0), 0)
    if (document.strokes.length >= 1000 || usedPoints >= 10000) return
    canvas.current.setPointerCapture(event.pointerId)
    stroke.current = { base: document, remainingPoints: 10000 - usedPoints, pointerId: event.pointerId, size: size / 100, erase: tool === 'erase', points: [point(event)] }
    setDocument(current => ({ ...current, strokes: [...current.strokes, { size: stroke.current.size, erase: stroke.current.erase, points: [...stroke.current.points] }] }))
    event.preventDefault()
  }

  function move(event) {
    const active = stroke.current
    if (!active || active.pointerId !== event.pointerId || active.points.length >= active.remainingPoints) return
    const next = point(event), previous = active.points.at(-1)
    if (Math.hypot(next[0] - previous[0], next[1] - previous[1]) < 0.001) return
    active.points.push(next)
    setDocument(current => ({ ...current, strokes: [...current.strokes.slice(0, -1), { size: active.size, erase: active.erase, points: [...active.points] }] }))
  }

  function finish(event) {
    const active = stroke.current
    if (!active || active.pointerId !== event.pointerId) return
    stroke.current = null
    const savedStroke = { size: active.size, erase: active.erase, points: [...active.points] }
    const base = active.base
    const next = { ...base, strokes: [...base.strokes, savedStroke] }
    setDocument(next)
    commitChange({ inpaint: true, editMask: next })
    if (canvas.current.hasPointerCapture(event.pointerId)) canvas.current.releasePointerCapture(event.pointerId)
  }

  function uploadFile(file) {
    if (file && !locked) onUpload(file)
  }

  return <section className="create-edit-workspace" aria-label="Image editing workspace">
    <header className="create-edit-heading">
      <div><h2>Edit your image</h2><p>Describe a change, or paint the area you want to replace.</p></div>
      <button type="button" disabled={locked} onClick={() => upload.current.click()}>{imageUrl ? 'Replace image' : 'Choose image'}</button>
    </header>
    <input ref={upload} type="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp,image/tiff" hidden
      aria-label="Upload image to editing canvas" onChange={event => { uploadFile(event.target.files?.[0]); event.target.value = '' }} />
    {imageUrl ? <>
      <div className="create-edit-toolbar" aria-label="Editing tools">
        <button type="button" aria-pressed={!areaEditing} disabled={locked} onClick={() => { setAreaEditing(false); commitChange({ inpaint: false }) }}>Whole image</button>
        <button type="button" aria-pressed={areaEditing} disabled={locked || !enabled} onClick={() => {
          setAreaEditing(true)
          commitChange({ inpaint: true })
        }}>Selected area</button>
        {areaEditing && <>
          <button type="button" aria-pressed={tool === 'paint'} disabled={locked || placement} onClick={() => setTool('paint')}>Brush</button>
          <button type="button" aria-pressed={tool === 'erase'} disabled={locked || placement} onClick={() => setTool('erase')}>Erase</button>
          <label className="create-brush-size">Brush size <input type="range" min="1" max="40" value={size} disabled={locked || placement} onChange={event => setSize(Number(event.target.value))} /><span>{size}%</span></label>
          <button type="button" disabled={locked || !document.strokes.length || placement} onClick={() => {
            const next = { ...document, strokes: document.strokes.slice(0, -1) }
            publish(next, next.strokes.length > 0 || next.inverted === true)
          }}>Undo</button>
          <button type="button" disabled={locked} onClick={() => publish({ ...EMPTY_MASK(), image: reference }, false)}>Clear selection</button>
        </>}
      </div>
      {placement && <p className="create-edit-hint">An advanced selection is active. Clear selection to paint a new area, or adjust placement in Advanced editor.</p>}
      {imageError && <p className="create-message error" role="alert">{imageError}</p>}
      <div className="create-edit-stage-wrap">
        <div className="create-edit-stage" style={dimensions ? { aspectRatio: `${dimensions.width} / ${dimensions.height}`, maxWidth: Math.round(dimensions.width / dimensions.height * 520) } : undefined}>
          <img src={imageUrl} alt="Image to edit" onLoad={event => setDimensions({ width: event.target.naturalWidth, height: event.target.naturalHeight })}
            onError={() => setImageError('The reference image could not be loaded. Choose it again to reconnect.')} />
          {dimensions && <canvas ref={canvas} className={canPaint ? 'can-paint' : ''} style={{ opacity: areaEditing ? undefined : 0 }} aria-label="Paint the area to edit"
            onPointerDown={start} onPointerMove={move} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish} />}
        </div>
      </div>
      <p className="create-edit-hint">{areaEditing ? 'Green areas will change. Unselected areas stay from your original image.' : 'The instruction applies to the whole image.'}</p>
    </> : <button type="button" className="create-edit-empty" disabled={locked} onClick={() => upload.current.click()}
      onDragOver={event => { event.preventDefault() }} onDrop={event => { event.preventDefault(); uploadFile(event.dataTransfer.files?.[0]) }}>
      <span aria-hidden="true">＋</span><strong>Add an image to start editing</strong><small>Drop an image here or choose one from your device.</small>
    </button>}
  </section>
}
