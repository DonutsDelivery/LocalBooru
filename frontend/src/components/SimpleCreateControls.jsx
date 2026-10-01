import { useCallback, useEffect, useId, useRef, useState } from 'react'

const LABELS = {
  prompt: 'Describe your image', stylePrompt: 'Style prompt', negativePrompt: 'Negative prompt',
  model: 'Primary model', modelMode: 'Model recipe', secondaryModel: 'Second model', modelBlend: 'Uniform model blend',
  width: 'Width', height: 'Height', aspectRatio: 'Image shape', resolutionMode: 'Sizing mode',
  megapixels: 'Resolution (MP)', batchSize: 'Images per run', steps: 'Steps', guidance: 'Guidance',
  seed: 'Seed', seedMode: 'Seed behavior', editPrompt: 'Describe the edited image', maskFeather: 'Selection feather (px)',
  sampler: 'Sampler', scheduler: 'Scheduler', compatibilityPreset: 'Compatibility preset', tapStrength: 'TAB strength',
  decensorWeight: 'Decensor weight', nagStrength: 'NAG scale', sdaStrength: 'SDA strength', toneStrength: 'ToneLab strength', toneModel: 'Tone model',
  upscale1Scale: 'Scale', upscale2Scale: 'Scale', postUpscaleScale: 'Scale',
  upscale1Denoise: 'Denoise', upscale2Denoise: 'Denoise', postUpscaleDenoise: 'Denoise',
  upscale1Engine: 'Engine', upscale2Engine: 'Engine', upscale1Model: 'Upscale model', upscale2Model: 'Upscale model', postUpscaleModel: 'Upscale model',
  upscale1Vae: 'Upscale VAE', upscale2Vae: 'Upscale VAE', postUpscaleVae: 'Upscale VAE', upscaleModel: 'Donut upscale model',
  upscale1SeedVrSteps: 'Upscale steps', upscale2SeedVrSteps: 'Upscale steps', postUpscaleSteps: 'Upscale steps',
  upscale1SeedVrDenoise: 'Upscale denoise', upscale2SeedVrDenoise: 'Upscale denoise', postUpscaleColorCorrection: 'Color correction',
  faceDenoise: 'Face denoise', maxFaces: 'Maximum faces', facePrompt: 'Describe only the face',
  geometryMode: 'Reference layout', outputCanvas: 'Output canvas',
  groundingPx: 'Reference guidance (px)', groundingSchedule: 'Guidance schedule', groundingStartPx: 'Start guidance (px)', groundingEndPx: 'End guidance (px)',
  editLora: 'Editing LoRA', editLoraStrength: 'Editing strength', maskBMode: 'Reference B subject mask', maskBModel: 'Subject mask model',
  maskBGrow: 'Grow / shrink (px)', maskBFeather: 'Subject feather (px)', maskBBackground: 'Reference background', maskBPrompt: 'Select this subject', maskBThreshold: 'Selection threshold',
  guidanceGeometryMode: 'Reference layout', cropAX: 'Horizontal placement', cropAY: 'Vertical placement', cropBX: 'Horizontal placement', cropBY: 'Vertical placement',
}
const EMPTY_FIELDS = {}
const SLIDER_FIELDS = new Set([
  'modelBlend', 'guidance', 'megapixels', 'tapStrength', 'decensorWeight', 'nagStrength', 'sdaStrength', 'toneStrength',
  'editLoraStrength', 'maskBThreshold', 'cropAX', 'cropAY', 'cropBX', 'cropBY',
  'upscale1Scale', 'upscale2Scale', 'postUpscaleScale', 'upscale1Denoise', 'upscale2Denoise', 'postUpscaleDenoise',
  'upscale1SeedVrDenoise', 'upscale2SeedVrDenoise', 'faceDenoise',
])
const SLIDER_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'])

function modelLabel(value) {
  return String(value).split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|gguf|pth)$/i, '').replace(/_/g, ' ')
}

function displayNumber(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || Number.isInteger(value)) return value
  const rounded = Number(value.toPrecision(12))
  // Hide arithmetic noise without rounding typed drafts or changing live widget values.
  return Math.abs(rounded - value) <= Number.EPSILON * Math.abs(value) * 8 ? rounded : value
}

function SliderNumberField({ label, value, bounds, sliderBounds, disabled, onChange, onCommit, onInteraction, className = '', context = '' }) {
  const id = useId()
  const [pending, setPending] = useState(false)
  const dirty = useRef(false)
  const interacting = useRef(false)
  const committing = useRef(false)
  const finishRequested = useRef(false)
  const mounted = useRef(true)
  const min = sliderBounds ? Math.max(bounds.min ?? sliderBounds[0], sliderBounds[0]) : bounds.min
  const max = sliderBounds ? Math.min(bounds.max ?? sliderBounds[1], sliderBounds[1]) : bounds.max
  const hasSlider = Number.isFinite(min) && Number.isFinite(max) && max > min
  const numeric = value === '' ? NaN : Number(value)
  const outside = hasSlider && Number.isFinite(numeric) && (numeric < min || numeric > max)
  const sliderValue = Number.isFinite(numeric) ? Math.min(max, Math.max(min, numeric)) : min

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      onInteraction(id, false)
    }
  }, [id, onInteraction])

  function begin() {
    if (interacting.current) return
    interacting.current = true
    onInteraction(id, true)
  }

  function end() {
    interacting.current = false
    onInteraction(id, false)
  }

  async function finish() {
    if (committing.current) {
      finishRequested.current = true
      return
    }
    if (!dirty.current) { end(); return }
    // Consume before awaiting so pointerup, lost capture and blur cannot submit twice.
    dirty.current = false
    committing.current = true
    setPending(true)
    try { await onCommit() }
    finally {
      committing.current = false
      if (mounted.current) {
        setPending(false)
        if (finishRequested.current) {
          finishRequested.current = false
          finish()
        } else if (!dirty.current) end()
      }
    }
  }

  function change(next, slider = false) {
    if (slider) begin()
    dirty.current = true
    onChange(next)
  }

  return <div className={`create-field create-slider-field ${className}`}>
    <label htmlFor={`${id}-number`}>{label}</label>
    <div className="create-slider-inputs">
      {hasSlider && <input type="range" aria-label={`${label} slider${context ? ` for ${context}` : ''}`} aria-describedby={outside ? `${id}-range-note` : undefined}
        value={displayNumber(sliderValue)} min={min} max={max} step={bounds.step ?? 'any'} disabled={disabled || pending}
        onPointerDown={begin}
        onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish} onBlur={finish}
        onKeyDown={event => { if (SLIDER_KEYS.has(event.key)) begin() }}
        onKeyUp={event => { if (SLIDER_KEYS.has(event.key)) finish() }}
        onInput={event => change(event.currentTarget.value, true)} />}
      <input id={`${id}-number`} type="number" value={displayNumber(value)} min={bounds.min ?? undefined} max={bounds.max ?? undefined}
        step={bounds.step ?? 'any'} disabled={disabled || pending} onChange={event => change(event.target.value)} onBlur={finish} />
    </div>
    {outside && <small id={`${id}-range-note`} className="create-slider-note">Slider range {min} to {max}. Current value retained.</small>}
  </div>
}

function SparkIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
    <path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z" />
    <path d="m20 2 .7 1.8L22.5 4l-1.8.7L20 6.5l-.7-1.8L17.5 4l1.8-.2L20 2Z" />
  </svg>
}

export default function SimpleCreateControls({
  snapshot, activeTab, onTabChange, connected, connecting, busy, error, mobilePane,
  onPatch, onGenerate, onUpload, onUploadMask, onRetry, onAdvanced,
  onReferenceTools, referenceUrls = {}, runInstant, onRunInstantChange, onDraftChange,
  children, historyContent,
}) {
  const [drafts, setDrafts] = useState({})
  const [inputError, setInputError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [lowerTab, setLowerTab] = useState('models')
  const [sliding, setSliding] = useState(false)
  const draftRef = useRef({})
  const sliderInteractions = useRef(new Set())
  const controlsAlive = useRef(true)
  const draftReporter = useRef(onDraftChange)
  const submission = useRef(false)
  const rememberedStrength = useRef({})
  const uploadId = useId()
  const fields = snapshot?.fields || EMPTY_FIELDS
  const available = key => !!fields[key] && fields[key].available !== false
  const value = key => {
    if (Object.hasOwn(drafts, key)) return drafts[key]
    const alias = key === 'prompt' ? 'editPrompt' : key === 'editPrompt' ? 'prompt' : null
    if (alias && Object.hasOwn(drafts, alias)) return drafts[alias]
    return fields[key]?.value ?? ''
  }
  const options = key => Array.isArray(fields[key]?.options) ? fields[key].options : []
  const locked = !connected || submitting || (!!busy && busy !== 'patch')
  const editing = value('editing') === true
  const merging = value('modelMode') === 'Merge two models'
  const independentEditing = editing && value('geometryMode') === 'Independent crops'
  const followsCrop = independentEditing && value('outputCanvas') === 'Follow A crop'
  const customSize = value('resolutionMode') === 'Custom' && !followsCrop
  const presetSize = value('resolutionMode') === 'Preset'
  const usesMegapixels = independentEditing
    ? followsCrop ? value('resolutionMode') !== 'Reference A · crop only' : value('resolutionMode') !== 'Custom'
    : presetSize || value('resolutionMode') === 'Reference A · megapixels'
  const usesShape = !followsCrop && (independentEditing ? !customSize : presetSize)
  const fixedSeed = /fixed|custom|manual|keep/i.test(String(value('seedMode')))
  const hasReference = key => /^donutref:[a-f0-9]{64}$/.test(String(value(key)))
  const notifyDrafts = useCallback(next => {
    if (controlsAlive.current) onDraftChange?.(submission.current || sliderInteractions.current.size > 0 || Object.keys(next).length > 0)
  }, [onDraftChange])
  const sliderInteraction = useCallback((id, active) => {
    if (!controlsAlive.current) return
    if (active) sliderInteractions.current.add(id)
    else sliderInteractions.current.delete(id)
    setSliding(sliderInteractions.current.size > 0)
    notifyDrafts(draftRef.current)
  }, [notifyDrafts])
  useEffect(() => { draftReporter.current = onDraftChange }, [onDraftChange])
  useEffect(() => {
    controlsAlive.current = true
    const interactions = sliderInteractions.current
    return () => {
      controlsAlive.current = false
      interactions.clear()
      draftRef.current = {}
      draftReporter.current?.(false)
    }
  }, [])

  function stage(key, nextValue) {
    const next = { ...draftRef.current, [key]: nextValue }
    // The genuine shared-prompt graph executes one scene field in both tabs.
    if (key === 'prompt') delete next.editPrompt
    if (key === 'editPrompt') delete next.prompt
    draftRef.current = next
    setDrafts(next)
    notifyDrafts(next)
    setInputError('')
  }

  const commit = useCallback(async patch => {
    const normalized = {}
    try {
      for (const [key, raw] of Object.entries(patch)) {
        const field = fields[key]
        if (!field || field.available === false) throw new Error(`${LABELS[key] || 'This control'} is unavailable in the current workflow.`)
        let next = field.kind === 'number' ? Number(raw) : raw
        if (field.kind === 'loras') {
          if (!Array.isArray(raw)) throw new Error('The current LoRA rows are invalid. Reload the workflow controls.')
          next = raw.map(row => {
            if (row.model_weight === '' || row.clip_weight === '') throw new Error('Enter a number for each LoRA strength.')
            const model = Number(row.model_weight ?? 1), text = Number(row.clip_weight ?? 1)
            if (![model, text].every(strength => Number.isFinite(strength) && strength >= (field.min ?? -1000) && strength <= (field.max ?? 1000))) throw new Error('Enter a LoRA strength within the supported range.')
            return { ...row, model_weight: model, clip_weight: text }
          })
        }
        if (field.kind === 'number' && (raw === '' || !Number.isFinite(next))) throw new Error(`Enter a number for ${LABELS[key] || key}.`)
        if (field.kind === 'number' && field.min != null && next < field.min) throw new Error(`${LABELS[key] || key} must be at least ${field.min}.`)
        if (field.kind === 'number' && field.max != null && next > field.max) throw new Error(`${LABELS[key] || key} must be at most ${field.max}.`)
        normalized[key] = next
      }
      if ('width' in normalized || 'height' in normalized) {
        if (fields.resolutionMode?.available !== false && fields.resolutionMode?.options?.includes('Custom')) normalized.resolutionMode = 'Custom'
      }
      if ((normalized.resolutionMode === 'Custom' || 'aspectRatio' in normalized)
        && (normalized.editing ?? fields.editing?.value) === true
        && (normalized.geometryMode ?? fields.geometryMode?.value) === 'Independent crops'
        && fields.outputCanvas?.available !== false && fields.outputCanvas?.options?.includes('Independent output')) normalized.outputCanvas = 'Independent output'
    } catch (validationError) {
      setInputError(validationError.message)
      onRunInstantChange?.(false)
      throw validationError
    }
    if (!Object.keys(normalized).length) return
    await onPatch(normalized)
    const remaining = { ...draftRef.current }
    for (const [key, raw] of Object.entries(patch)) if (Object.is(remaining[key], raw)) delete remaining[key]
    draftRef.current = remaining
    setDrafts(remaining)
    notifyDrafts(remaining)
  }, [fields, onPatch, onRunInstantChange, notifyDrafts])

  useEffect(() => {
    if (!runInstant || !connected || busy || submitting || sliding || !Object.keys(drafts).length) return
    const timer = setTimeout(() => {
      if (sliderInteractions.current.size || submission.current) return
      commit({ ...draftRef.current }).catch(() => {})
    }, 450)
    return () => clearTimeout(timer)
  }, [runInstant, connected, busy, submitting, sliding, drafts, commit])

  function commitField(key) {
    if (Object.hasOwn(draftRef.current, key)) return commit({ [key]: draftRef.current[key] }).catch(() => {})
  }

  function choose(key, nextValue) {
    stage(key, nextValue)
    commit({ [key]: nextValue }).catch(() => {})
  }

  async function chooseTab(tab) {
    if (tab === activeTab) return
    if (tab === 'tuning' || !connected || !available('editing')) {
      onTabChange(tab)
      return
    }
    try {
      await commit({ ...draftRef.current, editing: tab === 'edit' })
      onTabChange(tab)
    } catch { /* The host keeps the actionable bridge error visible. */ }
  }

  async function generate(event) {
    event.preventDefault()
    if (submission.current || locked || sliding || (!!busy && busy !== 'patch')) return
    submission.current = true
    setSubmitting(true)
    notifyDrafts(draftRef.current)
    setInputError('')
    try {
      await commit({ ...draftRef.current })
      await onGenerate()
    } catch { /* Keep drafts and the host's error for a deliberate retry. */ }
    finally {
      submission.current = false
      setSubmitting(false)
      notifyDrafts(draftRef.current)
    }
  }

  async function upload(event, target, mask = false) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (file.type && !file.type.startsWith('image/')) {
      setInputError('Choose an image file.')
      return
    }
    try {
      if (mask) await onUploadMask(file)
      else await onUpload(file, target)
    } catch { /* The host reports the scoped upload failure. */ }
  }

  function selectField(key, label = LABELS[key]) {
    if (!available(key)) return null
    const choices = options(key), current = value(key)
    const published = choices.some(option => Object.is(option, current))
    const isModel = /model|lora|vae/i.test(key) && key !== 'modelMode'
    return <label className="create-field" key={key}>{label}
      <select value={String(current)} disabled={locked || choices.length === 0} onChange={event => {
        const selected = choices.find(option => String(option) === event.target.value)
        if (selected != null) choose(key, selected)
      }}>
        {!published && <option value={String(current)}>{current ? (isModel ? modelLabel(current) : String(current)) : 'Current workflow setting'}</option>}
        {choices.map(option => <option key={String(option)} value={String(option)}>{isModel ? modelLabel(option) : String(option)}</option>)}
      </select>
    </label>
  }

  function textField(key, label = LABELS[key], placeholder, rows = 3) {
    if (!available(key)) return null
    return <label className="create-field" key={key}>{label}
      <textarea rows={rows} value={value(key)} placeholder={placeholder} disabled={locked}
        onChange={event => stage(key, event.target.value)} onBlur={() => commitField(key)} />
    </label>
  }

  function numberField(key, label = LABELS[key]) {
    if (!available(key)) return null
    const field = fields[key]
    if (SLIDER_FIELDS.has(key)) return <SliderNumberField key={key} label={label} value={value(key)} bounds={field} disabled={locked}
      onChange={next => stage(key, next)} onCommit={() => commitField(key)} onInteraction={sliderInteraction} />
    return <label className="create-field" key={key}>{label}
      <input type="number" value={displayNumber(value(key))} min={field.min ?? undefined} max={field.max ?? undefined} step={field.step ?? 'any'} disabled={locked}
        onChange={event => stage(key, event.target.value)} onBlur={() => commitField(key)} />
    </label>
  }

  function switchField(key, label) {
    if (!available(key)) return null
    return <label className="create-toggle" key={key}><span>{label}</span>
      <input type="checkbox" role="switch" checked={value(key) === true} disabled={locked} onChange={event => choose(key, event.target.checked)} />
    </label>
  }

  function feature(key, title, contents, note) {
    if (!available(key)) return null
    return <section className="create-effect-group" key={key}>{switchField(key, title)}
      {value(key) === true && <div className="create-effect-settings">{contents}{note && <p className="create-control-note">{note}</p>}</div>}
    </section>
  }

  function strengthFeature(key, title, contents, note) {
    if (!available(key)) return null
    const active = Number(value(key)) !== 0
    return <section className="create-effect-group" key={key}>
      <label className="create-toggle"><span>{title}<small>{active ? displayNumber(value(key)) : 'Off'}</small></span>
        <input type="checkbox" role="switch" checked={active} disabled={locked} onChange={event => {
          if (!event.target.checked) rememberedStrength.current[key] = value(key)
          const initial = rememberedStrength.current[key] || fields[key].default || Math.min(fields[key].max ?? 1, Math.max(fields[key].min ?? 0, 1))
          choose(key, event.target.checked ? Number(initial) : 0)
        }} />
      </label>
      {(active || Object.hasOwn(drafts, key)) && <div className="create-effect-settings">{numberField(key)}{contents}{note && <p className="create-control-note">{note}</p>}</div>}
    </section>
  }

  function referenceCard(key, label, scope) {
    if (!available(key)) return null
    const inputId = `${uploadId}-${key}`
    return <div className="create-reference-card" key={key}>
      <div className="create-reference-card-heading"><strong>{label}</strong>{hasReference(key) && <button type="button" className="create-text-button" disabled={locked} onClick={() => choose(key, '')}>Remove</button>}</div>
      <input id={inputId} className="create-sr-input" type="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp,image/tiff" disabled={locked} onChange={event => upload(event, key)} />
      <label className={`create-reference-upload${locked ? ' disabled' : ''}`} htmlFor={inputId}>
        {referenceUrls[key] ? <img src={referenceUrls[key]} alt={`${label} attached to this workflow`} /> : <span className="create-reference-plus" aria-hidden="true">＋</span>}
        <strong>{busy === 'upload-reference' ? 'Uploading…' : hasReference(key) ? 'Replace image' : 'Add image'}</strong>
      </label>
      {hasReference(key) && <button type="button" className="create-reference-tools" disabled={locked} onClick={() => onReferenceTools?.(scope, key.endsWith('B') ? 'B' : 'A')}>Crop & image tools</button>}
    </div>
  }

  const ratios = options('aspectRatio').map(option => {
    const match = String(option).match(/(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/)
    if (!match) return null
    const ratio = Number(match[1]) / Number(match[2])
    return { value: option, ratio, ratioLabel: match[0].replace(/\s/g, '') }
  }).filter(Boolean)

  function sizing() {
    return <>
      {ratios.length > 0 && <fieldset className="create-shape-field"><legend>Image shape</legend>
        <div className="create-shape-options">{ratios.map(shape => <button type="button" key={String(shape.value)} disabled={locked}
          aria-pressed={value('aspectRatio') === shape.value && usesShape} onClick={() => {
            const patch = { aspectRatio: shape.value }
            stage('aspectRatio', shape.value)
            if (options('resolutionMode').includes('Preset') && available('resolutionMode')) { patch.resolutionMode = 'Preset'; stage('resolutionMode', 'Preset') }
            commit(patch).catch(() => {})
          }}><span className="create-shape-glyph" aria-hidden="true"><span style={{ width: `${Math.min(26, 18 * Math.sqrt(shape.ratio))}px`, height: `${Math.min(26, 18 / Math.sqrt(shape.ratio))}px` }} /></span><strong>{shape.ratioLabel}</strong></button>)}</div>
      </fieldset>}
      {followsCrop && <p className="create-control-note">Following Reference A’s crop. Choosing a shape sets an independent output canvas.</p>}
      <details className="create-prompt-options"><summary>Resolution <span>{customSize ? `${value('width')} × ${value('height')}` : usesMegapixels ? `${displayNumber(value('megapixels')) || '1'} MP` : 'Reference crop'}</span></summary>
        {selectField('resolutionMode')}{usesMegapixels && numberField('megapixels')}
        {independentEditing && selectField('outputCanvas')}
        {customSize && <div className="create-control-pair">{numberField('width')}{numberField('height')}</div>}
      </details>
    </>
  }

  function loraRows() {
    if (!available('loras')) return null
    let rows = value('loras')
    if (typeof rows === 'string') { try { rows = JSON.parse(rows) } catch { rows = [] } }
    if (!Array.isArray(rows)) rows = []
    const choices = options('loras'), bounds = fields.loras
    const update = (index, key, next) => choose('loras', rows.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: next } : row))
    return <section className="create-lora-section">
      <header><div><h3>LoRAs</h3><p>Mix a style or subject into the current model recipe.</p></div><button type="button" disabled={locked || !choices.length} onClick={() => choose('loras', [...rows, {
        id: globalThis.crypto?.randomUUID?.() || `lora-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        enabled: true, lora_name: choices[0], model_weight: 1, clip_weight: 1,
      }])}>＋ Add LoRA</button></header>
      {rows.length === 0 && <p className="create-control-note">No LoRAs selected. Add one from the backend’s installed models.</p>}
      <div className="create-lora-rows">{rows.map((row, index) => <div className={`create-lora-row${row.enabled === false ? ' is-disabled' : ''}`} key={row.id || index}>
        <label className="create-lora-enabled"><input type="checkbox" checked={row.enabled !== false} disabled={locked} aria-label={`Enable LoRA ${index + 1}`} onChange={event => update(index, 'enabled', event.target.checked)} /></label>
        <label className="create-field create-lora-name">LoRA<select value={row.lora_name || ''} disabled={locked || row.enabled === false || !choices.length} onChange={event => update(index, 'lora_name', event.target.value)}>
          {!choices.includes(row.lora_name) && <option value={row.lora_name || ''}>{row.lora_name ? modelLabel(row.lora_name) : 'Choose an installed LoRA'}</option>}
          {choices.map(choice => <option value={choice} key={choice}>{modelLabel(choice)}</option>)}
        </select></label>
        {row.enabled !== false && ['model_weight', 'clip_weight'].map((key, strengthIndex) => <SliderNumberField key={key}
          className={`create-lora-strength create-lora-${strengthIndex === 0 ? 'model' : 'text'}-strength`} label={strengthIndex === 0 ? 'Model strength' : 'Text strength'} context={`LoRA ${index + 1}`}
          value={row[key] ?? 1} bounds={{ min: bounds.min ?? -1000, max: bounds.max ?? 1000, step: bounds.step ?? 0.01 }} sliderBounds={[-2, 2]} disabled={locked}
          onChange={next => stage('loras', rows.map((item, itemIndex) => itemIndex === index ? { ...item, [key]: next } : item))}
          onCommit={() => commitField('loras')} onInteraction={sliderInteraction} />)}
        <button type="button" className="create-lora-remove" disabled={locked} aria-label={`Remove LoRA ${index + 1}`} onClick={() => choose('loras', rows.filter((_, rowIndex) => rowIndex !== index))}>×</button>
      </div>)}</div>
    </section>
  }

  function upscaleSettings(prefix) {
    const seedVr = value(`${prefix}Engine`) === 'SeedVR2'
    return <>{selectField(`${prefix}Engine`)}
      {seedVr ? <>{selectField(`${prefix}Model`)}{selectField(`${prefix}Vae`)}<div className="create-control-pair">{numberField(`${prefix}SeedVrSteps`)}{numberField(`${prefix}SeedVrDenoise`)}</div>{numberField(`${prefix}Scale`)}</>
        : <>{selectField('upscaleModel')}<div className="create-control-pair">{numberField(`${prefix}Scale`)}{numberField(`${prefix}Denoise`)}</div></>}
    </>
  }

  const lowerMode = mobilePane === 'history' ? 'history' : mobilePane === 'models' ? 'models' : lowerTab

  // Comfy widget steps are not based on HTML's min offset; commit validates authored drafts.
  return <form className="create-simple-workspace" onSubmit={generate} noValidate>
    <aside className="create-controls" aria-label="Image controls">
      <div className="create-control-tabs" role="tablist" aria-label="Studio controls">
        {['create', 'edit', 'tuning'].map(tab => <button type="button" role="tab" id={`create-tab-${tab}`} key={tab}
          aria-selected={activeTab === tab} aria-controls={`create-panel-${tab}`} tabIndex={activeTab === tab ? 0 : -1} disabled={submitting || sliding || !!busy}
          onKeyDown={event => {
            const tabs = ['create', 'edit', 'tuning'], index = tabs.indexOf(tab)
            const next = event.key === 'ArrowRight' ? tabs[(index + 1) % tabs.length] : event.key === 'ArrowLeft' ? tabs[(index + tabs.length - 1) % tabs.length] : event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : null
            if (next) { event.preventDefault(); event.currentTarget.parentElement.querySelector(`#create-tab-${next}`)?.focus(); chooseTab(next) }
          }} onClick={() => chooseTab(tab)}>{tab === 'create' ? 'Create' : tab === 'edit' ? 'Edit' : 'Tuning'}</button>)}
      </div>
      <div className="create-controls-scroll" role="tabpanel" id={`create-panel-${activeTab}`} aria-labelledby={`create-tab-${activeTab}`}>
        <div className="create-controls-intro"><h2>{activeTab === 'edit' ? 'Shape your image' : activeTab === 'tuning' ? 'Generation settings' : 'What do you imagine?'}</h2>
          <p>{activeTab === 'edit' ? 'Describe the whole finished image, then guide the changes.' : activeTab === 'tuning' ? 'Tune the current workflow without losing its recipe.' : 'A subject, a setting, and a little atmosphere.'}</p></div>
        {!connected && <div className="create-connection-note" role="status"><span className={connecting ? 'create-spinner' : 'create-connection-dot'} aria-hidden="true" />
          <div><strong>{connecting ? 'Preparing your studio…' : 'Studio not connected'}</strong><p>{error || snapshot?.error || 'Waiting for ComfyUI and the workflow to finish loading. This can take up to five minutes. You can close the studio while it starts.'}</p>
            {!connecting && <button type="button" onClick={onRetry}>Retry connection</button>}</div>
        </div>}
        {activeTab !== 'tuning' && <>
          {textField(activeTab === 'edit' ? 'editPrompt' : 'prompt', 'Describe your image', activeTab === 'edit' ? 'The finished image: keep the subject, change the background to a sunny beach…' : 'A cozy cabin beside a lake, morning mist, warm sunlight…', 6)}
          {value('faceDetail') === true && textField('facePrompt', 'Describe only the face', 'Expression, facial features, makeup…', 3)}
          {(available('stylePrompt') || available('negativePrompt')) && <details className="create-prompt-options"><summary>Style & exclusions <span>Optional</span></summary>
            {textField('stylePrompt', 'Style prompt', 'Lighting, colors, artistic style…')}{textField('negativePrompt', 'Avoid in the image', 'Things you want to leave out…')}
          </details>}
          {activeTab === 'create' && <>
            {available('referenceGuidance') && <section className="create-reference-section">{switchField('referenceGuidance', 'Reference guidance')}
              {value('referenceGuidance') === true && <><p className="create-control-note">Guide the look or subject with images. This stays separate from editing.</p>
                <div className="create-reference-pair">{referenceCard('guidanceReferenceA', 'Reference A', 'guidance')}{value('guidanceUseReferenceB') === true && referenceCard('guidanceReferenceB', 'Reference B', 'guidance')}</div>
                {switchField('guidanceUseReferenceB', 'Use a second reference')}{selectField('guidanceGeometryMode')}
              </>}
            </section>}
            {sizing()}
          </>}
          {activeTab === 'edit' && <>
            <div className="create-reference-pair">{referenceCard('referenceA', 'Reference A', 'edit')}{value('useReferenceB') === true && referenceCard('referenceB', 'Reference B', 'edit')}</div>
            {switchField('useReferenceB', 'Use a second reference')}
            {value('useReferenceB') === true && available('maskBMode') && <details className="create-prompt-options"><summary>Reference B subject <span>{String(value('maskBMode'))}</span></summary>
              {selectField('maskBMode')}{value('maskBMode') === 'Auto subject' && selectField('maskBModel')}
              {value('maskBMode') === 'Prompt selection' && <>{textField('maskBPrompt', 'Select this subject', 'Hat, jacket, person…', 2)}{numberField('maskBThreshold')}</>}
              {value('maskBMode') !== 'Off' && <><div className="create-control-pair">{numberField('maskBGrow')}{numberField('maskBFeather')}</div>{selectField('maskBBackground')}</>}
              {onUploadMask && hasReference('referenceB') && <label className="create-mask-upload">Upload subject mask<input type="file" accept="image/png,image/jpeg,image/webp" disabled={locked} onChange={event => upload(event, null, true)} /></label>}
              <p className="create-control-note">White in a subject mask keeps the subject. Black removes the background.</p>
            </details>}
            <details className="create-prompt-options"><summary>Layout & guidance <span>Image tools</span></summary>
              {selectField('geometryMode')}{selectField('outputCanvas')}
              {value('geometryMode') === 'Legacy output-linked' && <><div className="create-control-pair">{numberField('cropAX')}{numberField('cropAY')}</div>{value('useReferenceB') === true && <div className="create-control-pair">{numberField('cropBX')}{numberField('cropBY')}</div>}</>}
              {numberField('groundingPx')}{selectField('groundingSchedule')}
              {available('groundingSchedule') && value('groundingSchedule') !== 'constant' && <div className="create-control-pair">{numberField('groundingStartPx')}{numberField('groundingEndPx')}</div>}
              {selectField('editLora')}{value('editLora') && value('editLora') !== 'None' && numberField('editLoraStrength')}
            </details>
            {value('inpaint') === true && numberField('maskFeather')}{sizing()}
          </>}
        </>}
        {activeTab === 'tuning' && <>
          <div className="create-control-pair">{numberField('steps')}{numberField('guidance')}</div>
          {selectField('sampler')}{selectField('scheduler')}{numberField('batchSize')}{selectField('seedMode')}{fixedSeed && numberField('seed')}
          {sizing()}
          <button type="button" className="create-advanced-link" disabled={locked} onClick={() => commit({ ...draftRef.current }).then(onAdvanced).catch(() => {})}>Open full workflow editor <span aria-hidden="true">↗</span></button>
        </>}
        {inputError && <p className="create-message error" role="alert">{inputError}</p>}
      </div>
      <footer className="create-generate-footer">
        <label className="create-instant-toggle"><input type="checkbox" checked={runInstant === true} disabled={locked || sliding || !!busy || (editing && !hasReference('referenceA'))} onChange={event => onRunInstantChange?.(event.target.checked)} />Run Instant</label>
        <button type="submit" className="create-primary create-generate" disabled={locked || sliding || (!!busy && busy !== 'patch') || (editing && !hasReference('referenceA'))}>
          <SparkIcon /><span>{busy === 'generate' ? 'Adding to queue…' : editing ? 'Generate edit' : 'Generate image'}</span>{Number(value('batchSize')) > 1 && <small>×{value('batchSize')}</small>}
        </button>
        <p>{busy === 'patch' ? 'Updating workflow…' : editing && !hasReference('referenceA') ? 'Add Reference A to begin.' : runInstant ? 'Runs the latest acknowledged draft when the queue is empty.' : connected ? 'Generate uses the current workflow.' : 'Connect the studio to generate.'}</p>
      </footer>
    </aside>
    <main className="create-center-workspace">{children}</main>
    <aside className="create-effects" aria-label="Image effects">
      <header className="create-pane-heading"><h2>Effects</h2><span>Current recipe</span></header>
      <div className="create-effects-scroll">
        {(available('compatibilityPreset') || available('tapStrength')) && <section className="create-effect-group">{selectField('compatibilityPreset')}{value('compatibilityPreset') !== 'Off' && numberField('tapStrength')}</section>}
        {feature('decensor', 'Decensor', numberField('decensorWeight'))}
        {feature('upscale1', 'First upscale', upscaleSettings('upscale1'))}{feature('upscale2', 'Second upscale', upscaleSettings('upscale2'))}
        {feature('faceDetail', 'Face detail', <>{numberField('faceDenoise')}{numberField('maxFaces')}</>)}
        {feature('postUpscale', 'Final upscale', <>{selectField('postUpscaleModel')}{selectField('postUpscaleVae')}{numberField('postUpscaleScale')}
          <div className="create-control-pair">{numberField('postUpscaleSteps')}{numberField('postUpscaleDenoise')}</div>{selectField('postUpscaleColorCorrection')}</>)}
        {strengthFeature('nagStrength', 'NAG', null, 'Scale 0 turns NAG off.')}{strengthFeature('sdaStrength', 'SDA')}
        {strengthFeature('toneStrength', 'ToneLab', <>{selectField('toneModel')}{switchField('toneApplyToEdits', 'Apply to edits')}</>)}
        {!['compatibilityPreset', 'tapStrength', 'decensor', 'upscale1', 'upscale2', 'faceDetail', 'postUpscale', 'nagStrength', 'sdaStrength', 'toneStrength'].some(available)
          && <p className="create-control-note">Effects appear when the loaded workflow exposes them.</p>}
      </div>
    </aside>
    <section className={`create-lower-pane create-lower-${lowerMode}`} aria-label="Models and image history">
      <div className="create-lower-tabs" role="tablist" aria-label="Models and history">{['models', 'history'].map(tab => <button type="button" role="tab" aria-selected={lowerMode === tab} key={tab} onClick={() => setLowerTab(tab)}>{tab === 'models' ? 'Models & LoRAs' : 'History'}</button>)}</div>
      <div className="create-lower-scroll">{lowerMode === 'models' ? <div className="create-model-workspace">
        {['modelMode', 'model'].some(available) && <section className="create-model-recipe"><h3>Model recipe</h3>{selectField('modelMode')}{selectField('model')}
          {merging && <>{selectField('secondaryModel')}{numberField('modelBlend')}<p className="create-control-note">Changing this blend applies one balance across the model. Advanced block weights stay intact until you change it.</p></>}
        </section>}{loraRows()}
      </div> : historyContent}</div>
    </section>
  </form>
}
