import { useEffect, useRef, useState } from 'react'
import './CreateEditCanvas.css'

const EMPTY_MASK = reference => ({ version: 1, image: reference, strokes: [], inverted: false })
const ASPECTS = ['Free', 'Original', '1:1', '2:3', '3:2', '3:4', '4:3', '9:16', '16:9', '21:9']
const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value))

function readDocument(value, reference) {
  try {
    const mask = typeof value === 'string' ? JSON.parse(value || 'null') : value
    return mask?.version === 1 && mask.image === reference && Array.isArray(mask.strokes) ? mask : EMPTY_MASK(reference)
  } catch { return EMPTY_MASK(reference) }
}

function readCrop(value, reference) {
  try {
    const crop = typeof value === 'string' ? JSON.parse(value || 'null') : value
    return crop?.version === 1 && crop.image === reference && Array.isArray(crop.bounds) && crop.bounds.length === 4
      && crop.bounds.every(Number.isFinite) ? crop : null
  } catch { return null }
}

function placementRect(source, output, placement) {
  const scale = Math.min(output[0] / source.width, output[1] / source.height) * placement.scale
  const width = Math.max(1, Math.round(source.width * scale)), height = Math.max(1, Math.round(source.height * scale))
  return [Math.round((output[0] - width) * placement.x), Math.round((output[1] - height) * placement.y), width, height]
}

function expandedSize(source, output, direction, grid) {
  const area = output[0] * output[1], ratio = source.width / source.height * (direction === 'horizontal' ? 2 : 0.5)
  let width = clamp(Math.round(Math.sqrt(area * ratio) / grid) * grid, grid, 16384)
  let height = clamp(Math.round(Math.sqrt(area / ratio) / grid) * grid, grid, 16384)
  while (width * height > area && (width > grid || height > grid)) {
    const candidates = [[width - grid, height], [width, height - grid]].filter(size => size.every(edge => edge >= grid))
    candidates.sort((a, b) => Math.abs(Math.log((a[0] / a[1]) / ratio)) - Math.abs(Math.log((b[0] / b[1]) / ratio)))
    ;[width, height] = candidates[0]
  }
  return [width, height]
}

function canvasDimensions(settings, source, crop, sourceB, cropB, fallback) {
  if (!settings || !source) return fallback
  const grid = Number(settings.pixelGrid || settings.outputMultiple) || 64
  const independent = settings.editing === true && settings.geometryMode === 'Independent crops'
  const cropped = (size, document) => size && document && Array.isArray(document.bounds) && document.bounds.length === 4
    && document.bounds.every(Number.isFinite) && document.source_size?.[0] === size.width && document.source_size?.[1] === size.height
    ? [Math.max(1, Math.round(document.bounds[2] * size.width) - Math.round(document.bounds[0] * size.width)), Math.max(1, Math.round(document.bounds[3] * size.height) - Math.round(document.bounds[1] * size.height))]
    : size ? [size.width, size.height] : null
  const a = independent ? cropped(source, crop) : [source.width, source.height]
  const b = independent ? cropped(sourceB, cropB) : sourceB ? [sourceB.width, sourceB.height] : null
  const followA = independent && settings.outputCanvas === 'Follow A crop'
  if (settings.resolutionMode === 'Reference A · crop only' && (followA || !independent)) return a.map(edge => Math.max(grid, Math.floor(edge / grid) * grid))
  if (settings.resolutionMode === 'Custom' && !followA) return [Number(settings.width), Number(settings.height)].map(edge => Math.max(grid, Math.round(edge / grid) * grid))
  const published = ASPECTS.filter(aspect => aspect.includes(':')).map(aspect => aspect.split(':').map(Number))
  const text = String(settings.aspectRatio || '4:3 Standard')
  const match = text.match(/(\d+):(\d+)/)
  let ratio = match ? [Number(match[1]), Number(match[2])] : [4, 3]
  if (followA || !independent && String(settings.resolutionMode).startsWith('Reference A')) ratio = a
  else if (text.startsWith('Auto')) {
    const image = text.endsWith('B') ? b : a
    if (image) {
      ratio = independent ? image : published.slice().sort((left, right) => Math.abs(Math.log((left[0] / left[1]) / (image[0] / image[1]))) - Math.abs(Math.log((right[0] / right[1]) / (image[0] / image[1]))))[0]
    }
  }
  const scale = Math.sqrt(Number(settings.megapixels || 1) * 1024 * 1024 / (ratio[0] * ratio[1]))
  const result = ratio.map(edge => Math.max(grid, Math.round(edge * scale / grid) * grid))
  return result.every(Number.isFinite) ? result : fallback
}

// Source-coordinate crops and strokes use Donut's original documents. The
// image is not flattened, re-uploaded, or rewritten to apply a selection.
export default function CreateEditCanvas({
  reference, imageUrl, maskData, enabled, selectedArea, disabled, onChange, onUpload,
  cropData, cropEnabled = false, onCropChange, outputSize: fallbackSize = [1152, 896], pixelGrid = 64,
  outputSettings, referenceBUrl, cropBData, workspaceSwitch,
  referenceLabel = 'Reference A', referenceOptions, onReferenceChange, onDraftChange, dropReference,
}) {
  const canvas = useRef(null)
  const stage = useRef(null)
  const upload = useRef(null)
  const pointer = useRef(null)
  const pendingChange = useRef(false)
  const change = useRef(onChange)
  const cropChange = useRef(onCropChange)
  const normalDocument = useRef(null)
  const placementDocument = useRef(null)
  const placementDraft = useRef(null)
  const [dimensions, setDimensions] = useState(null)
  const [dimensionsB, setDimensionsB] = useState(null)
  const [tool, setTool] = useState('paint')
  const [panel, setPanel] = useState(enabled ? readDocument(maskData, reference).outpaint ? 'outpaint' : 'mask' : 'crop')
  const [cropTool, setCropTool] = useState('draw')
  const [size, setSize] = useState(7)
  const [mask, setMask] = useState(() => readDocument(maskData, reference))
  const [crop, setCrop] = useState(() => readCrop(cropData, reference))
  const [imageError, setImageError] = useState('')
  const [areaEditing, setAreaEditing] = useState(selectedArea === true)
  const [saving, setSaving] = useState(false)
  useEffect(() => { change.current = onChange; cropChange.current = onCropChange }, [onChange, onCropChange])
  useEffect(() => () => onDraftChange?.(false), [onDraftChange])
  const incomingMask = typeof maskData === 'string' ? maskData : JSON.stringify(maskData)
  const incomingCrop = typeof cropData === 'string' ? cropData : JSON.stringify(cropData)

  useEffect(() => {
    pointer.current = null
    // Loaded source geometry is tied to this opaque reference URL.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDimensions(null)
    setImageError('')
  }, [imageUrl])

  useEffect(() => {
    if (!referenceBUrl || !String(outputSettings?.aspectRatio).endsWith('B')) return
    let active = true
    const image = new Image()
    image.onload = () => { if (active) setDimensionsB({ width: image.naturalWidth, height: image.naturalHeight, url: referenceBUrl }) }
    image.src = referenceBUrl
    return () => { active = false; image.onload = null }
  }, [referenceBUrl, outputSettings?.aspectRatio])

  useEffect(() => {
    if (pointer.current || placementDraft.current) return
    const next = readDocument(incomingMask, reference)
    // Synchronize edits made in the same live graph's advanced view.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMask(next)
    setAreaEditing(selectedArea ?? (next.strokes.length > 0 || next.inverted === true || !!next.outpaint))
    if (next.outpaint) placementDocument.current = next
    else normalDocument.current = next
  }, [incomingMask, reference, selectedArea])

  useEffect(() => {
    if (pointer.current) return
    // The workflow supplies an exact source-coordinate crop document.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCrop(readCrop(incomingCrop, reference))
  }, [incomingCrop, reference])

  const locked = disabled || saving
  const outputSize = canvasDimensions(outputSettings, dimensions, readCrop(cropData, reference), dimensionsB?.url === referenceBUrl ? dimensionsB : null,
    readCrop(cropBData, outputSettings?.referenceB), fallbackSize)
  const invalidOutput = outputSize.some(edge => !Number.isFinite(edge) || edge < 16 || edge > 16384)
  const outpainting = areaEditing && !!mask.outpaint && panel !== 'crop'
  const stageSize = dimensions ? outpainting ? outputSize : [dimensions.width, dimensions.height] : [1, 1]
  const rectangle = dimensions && outpainting ? placementRect(dimensions, outputSize, mask.outpaint) : null
  const canPaint = enabled && !locked && !!dimensions && !imageError && areaEditing && panel !== 'crop'
  const canCrop = cropEnabled && !locked && !!dimensions && !imageError && panel === 'crop'
  const bounds = crop?.bounds || [0, 0, 1, 1]
  const staleCrop = !!crop && !!dimensions && (crop.source_size?.[0] !== dimensions.width || crop.source_size?.[1] !== dimensions.height)

  useEffect(() => {
    const surface = canvas.current
    if (!surface || !dimensions) return
    const output = areaEditing && mask.outpaint && panel !== 'crop' ? outputSize : [dimensions.width, dimensions.height]
    const scale = Math.min(1, 1600 / Math.max(...output))
    const width = Math.max(1, Math.round(output[0] * scale)), height = Math.max(1, Math.round(output[1] * scale))
    if (surface.width !== width) surface.width = width
    if (surface.height !== height) surface.height = height
    const ctx = surface.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, surface.width, surface.height)
    const outside = rect => {
      const [x, y, width, height] = rect
      ctx.fillRect(0, 0, surface.width, y)
      ctx.fillRect(0, y + height, surface.width, surface.height - y - height)
      ctx.fillRect(0, y, x, height)
      ctx.fillRect(x + width, y, surface.width - x - width, height)
    }
    ctx.fillStyle = ctx.strokeStyle = '#00d99a'
    let placed
    if (mask.outpaint && areaEditing && panel !== 'crop') {
      placed = placementRect(dimensions, output, mask.outpaint).map(edge => edge * scale)
      const [x, y, width, height] = placed
      const overlap = Math.min(mask.outpaint.overlap * scale, (width - 1) / 2, (height - 1) / 2)
      const left = x + (x > 0 ? overlap : 0), top = y + (y > 0 ? overlap : 0)
      const right = x + width - (x + width < surface.width ? overlap : 0), bottom = y + height - (y + height < surface.height ? overlap : 0)
      outside([left, top, right - left, bottom - top])
    }
    for (const entry of mask.strokes) {
      if (!Array.isArray(entry.points) || !entry.points.length) continue
      const diameter = Math.max(1, Number(entry.size) * Math.min(surface.width, surface.height))
      ctx.globalCompositeOperation = entry.erase ? 'destination-out' : 'source-over'
      ctx.lineWidth = diameter; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
      const points = entry.points.map(([x, y]) => [x * (surface.width - 1), y * (surface.height - 1)])
      if (entry.shape === 'rectangle' && points.length === 2) {
        const [[x1, y1], [x2, y2]] = points
        ctx.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1))
      } else {
        ctx.beginPath(); points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke()
        for (const [x, y] of points) { ctx.beginPath(); ctx.arc(x, y, diameter / 2, 0, Math.PI * 2); ctx.fill() }
      }
    }
    if (mask.inverted === true) { ctx.globalCompositeOperation = 'xor'; ctx.fillRect(0, 0, surface.width, surface.height) }
    ctx.globalCompositeOperation = 'source-over'
    if (placed) outside(placed)
  }, [mask, dimensions, areaEditing, panel, outputSize])

  function commitChange(patch, isCrop = false) {
    if (pendingChange.current) return
    pendingChange.current = true
    setSaving(true)
    onDraftChange?.(true)
    Promise.resolve().then(() => isCrop ? cropChange.current?.(patch) : change.current(patch)).finally(() => {
      pendingChange.current = false
      setSaving(false)
      onDraftChange?.(false)
    }).catch(() => {})
  }

  function publishMask(next, active = true, extra = {}) {
    setMask(next); setAreaEditing(active)
    if (next.outpaint) placementDocument.current = next
    else normalDocument.current = next
    commitChange({ inpaint: active, editMask: next, ...extra })
  }

  function publishCrop(next) {
    setCrop(next)
    commitChange(next ? JSON.stringify(next) : '', true)
  }

  function stagePlacement(key, value) {
    if (pendingChange.current) return
    const base = placementDraft.current || mask
    if (base.outpaint[key] === value) return
    const next = { ...base, outpaint: { ...base.outpaint, [key]: value } }
    placementDraft.current = next
    setMask(next)
    onDraftChange?.(true)
  }

  function commitPlacement() {
    const next = placementDraft.current
    if (!next || pendingChange.current) return
    placementDraft.current = null
    const current = readDocument(maskData, reference).outpaint
    if (current && ['scale', 'x', 'y', 'overlap'].every(key => current[key] === next.outpaint[key])) {
      onDraftChange?.(false)
      return
    }
    publishMask(next)
  }

  function point(event) {
    const box = stage.current.getBoundingClientRect()
    return [clamp((event.clientX - box.left) / box.width), clamp((event.clientY - box.top) / box.height)]
  }

  function cropBounds(start, end, aspect) {
    let width = Math.abs(end[0] - start[0]), height = Math.abs(end[1] - start[1])
    const ratio = aspect === 'Original' ? dimensions.width / dimensions.height : aspect === 'Free' ? null : aspect.split(':').map(Number).reduce((a, b) => a / b)
    if (ratio) {
      const normalized = ratio * dimensions.height / dimensions.width
      height = width / normalized
      const maxHeight = end[1] < start[1] ? start[1] : 1 - start[1]
      if (height > maxHeight) { height = maxHeight; width = height * normalized }
    }
    const x = end[0] < start[0] ? start[0] - width : start[0], y = end[1] < start[1] ? start[1] - height : start[1]
    return [clamp(x), clamp(y), clamp(x + width), clamp(y + height)]
  }

  function start(event) {
    if (pendingChange.current || locked || event.button !== 0 || pointer.current || (!canPaint && !canCrop)) return
    const position = point(event)
    stage.current.setPointerCapture(event.pointerId)
    onDraftChange?.(true)
    if (canCrop) {
      const handle = cropTool === 'move' ? event.target.dataset.cropHandle : null
      pointer.current = { kind: 'crop', pointerId: event.pointerId, start: position, bounds: [...bounds], aspect: crop?.aspect || 'Free', operation: handle || 'draw' }
      if (!handle) setCrop({ version: 1, image: reference, source_size: [dimensions.width, dimensions.height], aspect: crop?.aspect || 'Free', bounds: [position[0], position[1], position[0], position[1]] })
    } else if (tool === 'move' && mask.outpaint) {
      pointer.current = { kind: 'move', pointerId: event.pointerId, start: position, base: mask }
    } else {
      const usedPoints = mask.strokes.reduce((total, entry) => total + (entry.points?.length || 0), 0)
      if (mask.strokes.length >= 1000 || usedPoints >= 10000) { onDraftChange?.(false); return }
      const selection = { size: size / 100, erase: tool === 'erase', points: tool === 'rectangle' ? [position, position] : [position], ...(tool === 'rectangle' ? { shape: 'rectangle' } : {}) }
      pointer.current = { kind: 'stroke', pointerId: event.pointerId, base: mask, remainingPoints: 10000 - usedPoints, selection }
      setMask({ ...mask, strokes: [...mask.strokes, selection] })
    }
    event.preventDefault()
  }

  function move(event) {
    const active = pointer.current
    if (!active || active.pointerId !== event.pointerId) return
    const position = point(event)
    if (active.kind === 'crop') {
      let next
      if (active.operation === 'move') {
        const dx = clamp(position[0] - active.start[0], -active.bounds[0], 1 - active.bounds[2]), dy = clamp(position[1] - active.start[1], -active.bounds[1], 1 - active.bounds[3])
        next = active.bounds.map((value, index) => value + (index % 2 === 0 ? dx : dy))
      } else {
        const anchors = { nw: [active.bounds[2], active.bounds[3]], ne: [active.bounds[0], active.bounds[3]], sw: [active.bounds[2], active.bounds[1]], se: [active.bounds[0], active.bounds[1]] }
        next = cropBounds(anchors[active.operation] || active.start, position, active.aspect)
      }
      active.result = { version: 1, image: reference, source_size: [dimensions.width, dimensions.height], aspect: active.aspect, bounds: next }
      setCrop(active.result)
    } else if (active.kind === 'move') {
      const rect = placementRect(dimensions, outputSize, active.base.outpaint)
      const x = clamp(active.base.outpaint.x + (position[0] - active.start[0]) * outputSize[0] / Math.max(1, outputSize[0] - rect[2]))
      const y = clamp(active.base.outpaint.y + (position[1] - active.start[1]) * outputSize[1] / Math.max(1, outputSize[1] - rect[3]))
      active.result = { ...active.base, outpaint: { ...active.base.outpaint, x, y } }
      setMask(active.result)
    } else {
      if (active.selection.shape === 'rectangle') active.selection.points[1] = position
      else {
        if (active.selection.points.length >= active.remainingPoints) return
        const previous = active.selection.points.at(-1)
        if (Math.hypot(position[0] - previous[0], position[1] - previous[1]) < 0.001) return
        active.selection.points.push(position)
      }
      setMask({ ...active.base, strokes: [...active.base.strokes, { ...active.selection, points: [...active.selection.points] }] })
    }
  }

  function finish(event) {
    const active = pointer.current
    if (!active || active.pointerId !== event.pointerId) return
    pointer.current = null
    if (stage.current.hasPointerCapture(event.pointerId)) stage.current.releasePointerCapture(event.pointerId)
    if (active.kind === 'crop') {
      const next = active.result
      if (next && next.bounds[2] - next.bounds[0] >= 1 / dimensions.width && next.bounds[3] - next.bounds[1] >= 1 / dimensions.height) { setCropTool('move'); publishCrop(next) }
      else { setCrop(readCrop(cropData, reference)); onDraftChange?.(false) }
    } else if (active.kind === 'move') {
      if (active.result) publishMask(active.result)
      else onDraftChange?.(false)
    } else publishMask({ ...active.base, strokes: [...active.base.strokes, { ...active.selection, points: [...active.selection.points] }] })
  }

  function setAspect(aspect) {
    if (!dimensions) return
    if (aspect === 'Free') { publishCrop({ version: 1, image: reference, source_size: [dimensions.width, dimensions.height], bounds, aspect }); return }
    const ratio = aspect === 'Original' ? dimensions.width / dimensions.height : aspect.split(':').map(Number).reduce((a, b) => a / b)
    const normalized = ratio * dimensions.height / dimensions.width
    let width = bounds[2] - bounds[0], height = width / normalized
    if (height > 1) { height = 1; width = normalized }
    const x = clamp((bounds[0] + bounds[2] - width) / 2, 0, 1 - width), y = clamp((bounds[1] + bounds[3] - height) / 2, 0, 1 - height)
    publishCrop({ version: 1, image: reference, source_size: [dimensions.width, dimensions.height], aspect, bounds: [x, y, x + width, y + height] })
  }

  function selectPanel(next) {
    setPanel(next)
    if (next === 'crop') return
    if (next === 'outpaint') {
      if (!mask.outpaint) normalDocument.current = mask
      const nextDocument = mask.outpaint ? mask : placementDocument.current || { ...EMPTY_MASK(reference), outpaint: { scale: 0.7, x: 0.5, y: 0.5, overlap: 16 } }
      setTool('move'); publishMask(nextDocument)
    } else if (mask.outpaint) {
      placementDocument.current = mask
      setTool('paint'); publishMask(normalDocument.current || EMPTY_MASK(reference))
    }
  }

  function expand(direction, x, y) {
    const grid = [16, 32, 64].includes(Number(pixelGrid)) ? Number(pixelGrid) : 64
    const size = expandedSize(dimensions, outputSize, direction, grid)
    const next = { ...mask, outpaint: { ...(mask.outpaint || {}), scale: 1, x, y, overlap: mask.outpaint?.overlap ?? 16 } }
    publishMask(next, true, { resolutionMode: 'Custom', width: size[0], height: size[1], outputCanvas: 'Independent output' })
  }

  function uploadFile(file) { if (file && !locked) onUpload(file) }

  return <section className="create-edit-workspace" aria-label="Image editing workspace" data-create-drop-reference={dropReference}>
    <header className="create-edit-heading">{workspaceSwitch}<div><h2>{referenceLabel}</h2><p>{enabled ? 'Crop, select, or extend your image.' : 'Choose the part of this reference to use.'}</p></div>
      <div className="create-edit-reference-switch">{referenceOptions?.map(option => <button type="button" key={option.id} aria-pressed={option.active} disabled={locked} onClick={() => onReferenceChange?.(option.id)}>{option.label}</button>)}
        <button type="button" disabled={locked} onClick={() => upload.current.click()}>{imageUrl ? 'Replace' : 'Add image'}</button></div>
    </header>
    <input ref={upload} type="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp,image/tiff" hidden aria-label="Upload image to editing canvas"
      onChange={event => { uploadFile(event.target.files?.[0]); event.target.value = '' }} />
    {imageUrl ? <>
      <div className="create-edit-mode-tabs" role="group" aria-label="Image tools">
        {cropEnabled && <button type="button" aria-pressed={panel === 'crop'} disabled={locked} onClick={() => selectPanel('crop')}>Crop</button>}
        {enabled && <><button type="button" aria-pressed={panel === 'mask'} disabled={locked} onClick={() => selectPanel('mask')}>Select area</button>
          <button type="button" aria-pressed={panel === 'outpaint'} disabled={locked || !dimensions || invalidOutput} onClick={() => selectPanel('outpaint')}>Outpaint</button></>}
      </div>
      {panel === 'crop' && cropEnabled && <div className="create-edit-toolbar"><label className="create-crop-aspect">Crop shape<select value={crop?.aspect || 'Free'} disabled={locked || !dimensions} onChange={event => setAspect(event.target.value)}>{ASPECTS.map(aspect => <option key={aspect}>{aspect}</option>)}</select></label>
        <button type="button" aria-pressed={cropTool === 'draw'} disabled={locked} onClick={() => setCropTool('draw')}>Draw crop</button>
        <button type="button" aria-pressed={cropTool === 'move'} disabled={locked || !crop} onClick={() => setCropTool('move')}>Move / resize</button>
        <button type="button" disabled={locked || !crop} onClick={() => publishCrop(null)}>Reset crop</button><span className="create-edit-hint">Drag to crop. Move or resize the selection.</span>
      </div>}
      {enabled && panel !== 'crop' && <>
        <div className="create-edit-toolbar"><button type="button" aria-pressed={!areaEditing} disabled={locked} onClick={() => { setAreaEditing(false); commitChange({ inpaint: false }) }}>Whole image</button>
          <button type="button" aria-pressed={areaEditing} disabled={locked} onClick={() => { setAreaEditing(true); commitChange({ inpaint: true }) }}>Selected area</button>
          {areaEditing && <>
            {outpainting && <button type="button" aria-pressed={tool === 'move'} disabled={locked} onClick={() => setTool('move')}>Move image</button>}
            {[['paint', 'Brush'], ['erase', 'Erase'], ['rectangle', 'Rectangle']].map(([key, label]) => <button type="button" aria-pressed={tool === key} key={key} disabled={locked} onClick={() => setTool(key)}>{label}</button>)}
            {['paint', 'erase'].includes(tool) && <label className="create-brush-size">Size<input type="range" min="1" max="40" value={size} disabled={locked} onChange={event => setSize(Number(event.target.value))} /><span>{size}%</span></label>}
            <button type="button" disabled={locked || !mask.strokes.length} onClick={() => publishMask({ ...mask, strokes: mask.strokes.slice(0, -1) })}>Undo</button>
            <button type="button" aria-pressed={mask.inverted === true} disabled={locked} onClick={() => publishMask({ ...mask, inverted: !mask.inverted })}>Invert</button>
            <button type="button" disabled={locked} onClick={() => { normalDocument.current = null; placementDocument.current = null; publishMask(EMPTY_MASK(reference), false) }}>Clear selection</button>
          </>}
        </div>
        {outpainting && <div className="create-outpaint-controls">
          <div className="create-outpaint-presets"><span>Extend canvas</span>{[['Left', 'horizontal', 1, 0.5], ['Right', 'horizontal', 0, 0.5], ['Up', 'vertical', 0.5, 1], ['Down', 'vertical', 0.5, 0]].map(([label, direction, x, y]) => <button type="button" key={label} disabled={locked} onClick={() => expand(direction, x, y)}>{label}</button>)}</div>
          <div className="create-placement-sliders">{[['scale', 'Image size', 10, 100], ['x', 'Horizontal', 0, 100], ['y', 'Vertical', 0, 100], ['overlap', 'Edge overlap', 0, 128]].map(([key, label, min, max]) => <label key={key}>{label}<input type="range" min={min} max={max} step="1" value={Math.round(mask.outpaint[key] * (key === 'overlap' ? 1 : 100))} disabled={locked}
            onChange={event => stagePlacement(key, Number(event.target.value) / (key === 'overlap' ? 1 : 100))}
            onPointerUp={commitPlacement} onKeyUp={commitPlacement} onBlur={commitPlacement} /><span>{Math.round(mask.outpaint[key] * (key === 'overlap' ? 1 : 100))}{key === 'overlap' ? ' px' : '%'}</span></label>)}</div>
        </div>}
      </>}
      {staleCrop && <p className="create-message error" role="alert">The saved crop belongs to a different image size. Reset or draw the crop again.</p>}
      {invalidOutput && <p className="create-message error" role="alert">This crop and resolution exceed the supported canvas size. Use a wider crop or lower resolution.</p>}
      {imageError && <p className="create-message error" role="alert">{imageError}</p>}
      <div className="create-edit-stage-wrap"><div ref={stage} className={`create-edit-stage${canCrop ? ' can-crop' : ''}`} style={dimensions ? { aspectRatio: `${stageSize[0]} / ${stageSize[1]}`, maxWidth: Math.round(stageSize[0] / stageSize[1] * 570) } : undefined}
        onPointerDown={start} onPointerMove={move} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish}>
        <img src={imageUrl} alt={referenceLabel} draggable="false" style={rectangle ? { position: 'absolute', left: `${rectangle[0] / outputSize[0] * 100}%`, top: `${rectangle[1] / outputSize[1] * 100}%`, width: `${rectangle[2] / outputSize[0] * 100}%`, height: `${rectangle[3] / outputSize[1] * 100}%` } : undefined}
          onLoad={event => setDimensions({ width: event.target.naturalWidth, height: event.target.naturalHeight })} onError={() => setImageError('This reference image could not be loaded. Upload it again to reconnect.')} />
        {dimensions && <canvas ref={canvas} className={canPaint ? 'can-paint' : ''} style={{ opacity: areaEditing && panel !== 'crop' ? undefined : 0 }} aria-label="Paint the area to edit" />}
        {dimensions && panel === 'crop' && cropEnabled && <div className="create-crop-selection" data-crop-handle="move" style={{ pointerEvents: cropTool === 'draw' ? 'none' : undefined, left: `${bounds[0] * 100}%`, top: `${bounds[1] * 100}%`, width: `${(bounds[2] - bounds[0]) * 100}%`, height: `${(bounds[3] - bounds[1]) * 100}%` }}>
          {['nw', 'ne', 'sw', 'se'].map(handle => <span className={`create-crop-handle ${handle}`} data-crop-handle={handle} key={handle} />)}<span className="create-crop-size" data-crop-handle="move">{Math.round((bounds[2] - bounds[0]) * dimensions.width)} × {Math.round((bounds[3] - bounds[1]) * dimensions.height)}</span>
        </div>}
      </div></div>
      <p className="create-edit-hint">{panel === 'crop' ? 'Crops stay in the original image coordinates. Your source image stays intact.' : outpainting ? `${outputSize[0]} × ${outputSize[1]} canvas · Green space will be generated. Move or shrink the image to set its placement.` : areaEditing ? 'Green areas will change. Unselected areas stay from your original image.' : 'The instruction applies to the whole image.'}</p>
    </> : <button type="button" className="create-edit-empty" disabled={locked} onClick={() => upload.current.click()}
      onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); uploadFile(event.dataTransfer.files?.[0]) }}>
      <span aria-hidden="true">＋</span><strong>Add {referenceLabel} to begin</strong><small>Drop an image here or choose one from your device.</small>
    </button>}
  </section>
}
