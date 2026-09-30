import { useId, useRef, useState } from 'react'

const LABELS = {
  prompt: 'Prompt', stylePrompt: 'Style prompt', negativePrompt: 'Negative prompt',
  model: 'Model', modelMode: 'Model recipe', secondaryModel: 'Second model',
  width: 'Width', height: 'Height', aspectRatio: 'Image shape', resolutionMode: 'Sizing mode',
  megapixels: 'Resolution (MP)', batchSize: 'Images per run', steps: 'Steps', guidance: 'Guidance',
  seed: 'Seed', seedMode: 'Seed behavior', editPrompt: 'Edit instructions', maskFeather: 'Selection feather',
}

function modelLabel(value) {
  return String(value).split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|gguf)$/i, '').replace(/_/g, ' ')
}

function SparkIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
    <path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z" />
    <path d="m20 2 .7 1.8L22.5 4l-1.8.7L20 6.5l-.7-1.8L17.5 4l1.8-.2L20 2Z" />
  </svg>
}

export default function SimpleCreateControls({
  snapshot, activeTab, onTabChange, connected, connecting, busy, error,
  onPatch, onGenerate, onUpload, onRetry, onAdvanced,
}) {
  const [drafts, setDrafts] = useState({})
  const [inputError, setInputError] = useState('')
  const [uploadedName, setUploadedName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const draftRef = useRef({})
  const submission = useRef(false)
  const uploadId = useId()
  const fields = snapshot?.fields || {}
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
  const customSize = value('resolutionMode') === 'Custom'
  const presetSize = value('resolutionMode') === 'Preset'
  const usesMegapixels = presetSize || value('resolutionMode') === 'Reference A · megapixels'
  const fixedSeed = /fixed|custom|manual|keep/i.test(String(value('seedMode')))
  const reference = value('referenceA')
  const hasReference = typeof reference === 'string' && reference.length > 0 && !/^(none|null)$/i.test(reference)

  function stage(key, nextValue) {
    const next = { ...draftRef.current, [key]: nextValue }
    // Genuine v5 uses one executed scene prompt for both views. Keep one
    // local override, so a later edit can never submit two conflicting values.
    if (key === 'prompt') delete next.editPrompt
    if (key === 'editPrompt') delete next.prompt
    draftRef.current = next
    setDrafts(next)
    setInputError('')
  }

  async function commit(patch) {
    const normalized = {}
    try {
      for (const [key, raw] of Object.entries(patch)) {
        if (!available(key)) throw new Error(`${LABELS[key] || 'This control'} is unavailable in the current workflow.`)
        const field = fields[key]
        const next = field.kind === 'number' ? Number(raw) : raw
        if (field.kind === 'number' && (raw === '' || !Number.isFinite(next))) throw new Error(`Enter a number for ${LABELS[key] || key}.`)
        if (field.min != null && next < field.min) throw new Error(`${LABELS[key] || key} must be at least ${field.min}.`)
        if (field.max != null && next > field.max) throw new Error(`${LABELS[key] || key} must be at most ${field.max}.`)
        normalized[key] = next
      }
      if ('width' in normalized || 'height' in normalized) {
        const customMode = options('resolutionMode').find(option => option === 'Custom')
        if (customMode != null && available('resolutionMode')) normalized.resolutionMode = customMode
      }
    } catch (validationError) {
      setInputError(validationError.message)
      throw validationError
    }
    if (!Object.keys(normalized).length) return
    await onPatch(normalized)
    const remaining = { ...draftRef.current }
    for (const [key, raw] of Object.entries(patch)) {
      if (Object.is(remaining[key], raw)) delete remaining[key]
    }
    draftRef.current = remaining
    setDrafts(remaining)
  }

  function commitField(key) {
    if (Object.hasOwn(draftRef.current, key)) commit({ [key]: draftRef.current[key] }).catch(() => {})
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
    if (submission.current || locked || busy) return
    submission.current = true
    setSubmitting(true)
    setInputError('')
    try {
      await commit({ ...draftRef.current })
      await onGenerate()
    } catch { /* Keep the draft for retry after validation or backend failure. */ }
    finally {
      submission.current = false
      setSubmitting(false)
    }
  }

  async function upload(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (file.type && !file.type.startsWith('image/')) {
      setInputError('Choose an image to edit.')
      return
    }
    try {
      await onUpload(file)
      setUploadedName(file.name)
      onTabChange('edit')
    } catch { /* The host reports upload errors without losing this session. */ }
  }

  function selectField(key, label = LABELS[key]) {
    if (!available(key)) return null
    const choices = options(key)
    const current = value(key)
    const published = choices.some(option => Object.is(option, current))
    return <label className="create-field" key={key}>{label}
      <select value={String(current)} disabled={locked || choices.length === 0} onChange={event => {
        const selected = choices.find(option => String(option) === event.target.value)
        if (selected != null) choose(key, selected)
      }}>
        {!published && <option value={String(current)}>{current ? (key === 'model' || key === 'secondaryModel' ? modelLabel(current) : String(current)) : 'Current workflow setting'}</option>}
        {choices.map(option => <option key={String(option)} value={String(option)}>{key === 'model' || key === 'secondaryModel' ? modelLabel(option) : String(option)}</option>)}
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
    return <label className="create-field" key={key}>{label}
      <input type="number" value={value(key)} min={field.min ?? undefined} max={field.max ?? undefined} step={field.step ?? 'any'} disabled={locked}
        onChange={event => stage(key, event.target.value)} onBlur={() => commitField(key)} />
    </label>
  }

  const ratios = options('aspectRatio').map(option => {
    const match = String(option).match(/(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/)
    if (!match) return null
    const ratio = Number(match[1]) / Number(match[2])
    const label = String(option).replace(match[0], '').trim() || (ratio === 1 ? 'Square' : ratio > 1 ? 'Landscape' : 'Portrait')
    return { value: option, ratio, label, ratioLabel: match[0].replace(/\s/g, '') }
  }).filter(Boolean)

  return <aside className="create-controls" aria-label="Image controls">
    <div className="create-control-tabs" role="tablist" aria-label="Studio controls">
      {['create', 'edit', 'tuning'].map(tab => <button type="button" role="tab" id={`create-tab-${tab}`} key={tab}
        aria-selected={activeTab === tab} aria-controls={`create-panel-${tab}`} tabIndex={activeTab === tab ? 0 : -1} disabled={submitting || !!busy}
        onKeyDown={event => {
          const tabs = ['create', 'edit', 'tuning']
          const index = tabs.indexOf(tab)
          const next = event.key === 'ArrowRight' ? tabs[(index + 1) % tabs.length] : event.key === 'ArrowLeft' ? tabs[(index + tabs.length - 1) % tabs.length] : event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : null
          if (next) {
            event.preventDefault()
            event.currentTarget.parentElement.querySelector(`#create-tab-${next}`)?.focus()
            chooseTab(next)
          }
        }} onClick={() => chooseTab(tab)}>{tab === 'create' ? 'Create' : tab === 'edit' ? 'Edit' : 'Tuning'}</button>)}
    </div>
    <form className="create-controls-form" onSubmit={generate}>
      <div className="create-controls-scroll" role="tabpanel" id={`create-panel-${activeTab}`} aria-labelledby={`create-tab-${activeTab}`}>
        <div className="create-controls-intro">
          <h2>{activeTab === 'edit' ? 'Make it your own' : activeTab === 'tuning' ? 'Fine tune your image' : 'Start with an idea'}</h2>
          <p>{activeTab === 'edit' ? 'Choose a reference and describe what to change.' : activeTab === 'tuning' ? 'Adjust the settings of your current workflow.' : 'Describe what you want to see. The workflow handles the rest.'}</p>
        </div>
        {!connected && <div className="create-connection-note" role="status">
          <span className={connecting ? 'create-spinner' : 'create-connection-dot'} aria-hidden="true" />
          <div><strong>{connecting ? 'Preparing your studio…' : 'Studio not connected'}</strong>
            <p>{error || snapshot?.error || 'Controls will be ready when the workflow finishes loading.'}</p>
            {!connecting && <button type="button" onClick={onRetry}>Retry connection</button>}
          </div>
        </div>}
        {activeTab === 'create' && <>
          {selectField('model') || <p className="create-control-note">The current workflow model is managed in the Advanced editor.</p>}
          {available('modelMode') && <p className="create-recipe-note"><span>Current recipe</span><strong>{String(value('modelMode'))}</strong>
            {merging && available('secondaryModel') && value('secondaryModel') && <span>with {modelLabel(value('secondaryModel'))}</span>}</p>}
          {textField('prompt', 'Describe your image', 'A cozy cabin beside a lake, morning mist, warm sunlight…', 6)}
          {ratios.length > 0 && <fieldset className="create-shape-field">
            <legend>Image shape</legend>
            <div className="create-shape-options">{ratios.map(shape => <button type="button" key={String(shape.value)} disabled={locked}
              aria-pressed={value('aspectRatio') === shape.value && presetSize}
              onClick={() => {
                const patch = { aspectRatio: shape.value }
                stage('aspectRatio', shape.value)
                const preset = options('resolutionMode').find(option => option === 'Preset')
                if (preset != null && available('resolutionMode')) {
                  patch.resolutionMode = preset
                  stage('resolutionMode', preset)
                }
                commit(patch).catch(() => {})
              }}>
              <span className="create-shape-glyph" aria-hidden="true"><span style={{ width: `${Math.min(32, 23 * Math.sqrt(shape.ratio))}px`, height: `${Math.min(32, 23 / Math.sqrt(shape.ratio))}px` }} /></span>
              <strong>{shape.ratioLabel}</strong><span>{shape.label}</span>
            </button>)}</div>
            <p className="create-control-note">{customSize && available('width') && available('height') ? `${value('width')} × ${value('height')} · Custom size` : `${value('resolutionMode') || 'Current workflow size'}${usesMegapixels && available('megapixels') ? ` · ${value('megapixels')} MP` : ''}`}</p>
          </fieldset>}
          {usesMegapixels && numberField('megapixels')}
          {(available('stylePrompt') || available('negativePrompt')) && <details className="create-prompt-options">
            <summary>Style & exclusions <span>Optional</span></summary>
            {textField('stylePrompt', 'Style prompt', 'Lighting, colors, artistic style…')}
            {textField('negativePrompt', 'Avoid in the image', 'Things you want to leave out…')}
          </details>}
        </>}
        {activeTab === 'edit' && <>
          {available('referenceA') && <div className="create-reference-control">
            <span className="create-field-heading">Reference image</span>
            <input id={uploadId} className="create-sr-input" type="file" accept="image/*" disabled={locked || !available('referenceA')} onChange={upload} />
            <label className={`create-reference-upload${locked || !available('referenceA') ? ' disabled' : ''}`} htmlFor={uploadId}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" /></svg>
              <strong>{busy === 'upload-reference' ? 'Uploading image…' : hasReference ? 'Replace reference image' : 'Choose an image'}</strong>
              <span>{uploadedName || (hasReference ? 'Reference attached to this workflow' : 'Upload a photo, drawing, or generated image')}</span>
            </label>
          </div>}
          {textField('editPrompt', 'What would you like to change?', 'Change the background to a sunny beach, keep the subject…', 6)}
          {available('inpaint') && <p className="create-control-note">Edit the whole image, or paint an area in the canvas.</p>}
          {!available('editPrompt') && <p className="create-control-note">This workflow has no editable instruction field. Open Advanced to choose an editing preset.</p>}
          {selectField('model')}
          {value('inpaint') === true && numberField('maskFeather', 'Selection feather (px)')}
          <p className="create-control-note">Use Tuning for seed, steps, and model recipe settings.</p>
        </>}
        {activeTab === 'tuning' && <>
          {['modelMode', 'model'].some(available) && <section className="create-tuning-section"><h3>Model recipe</h3>
            {selectField('modelMode')}{selectField('model')}{merging && selectField('secondaryModel')}
          </section>}
          {['steps', 'guidance', 'batchSize', 'seedMode'].some(available) && <section className="create-tuning-section"><h3>Generation</h3>
            <div className="create-control-pair">{numberField('steps')}{numberField('guidance')}</div>
            {numberField('batchSize')}{selectField('seedMode')}{fixedSeed && numberField('seed')}
          </section>}
          {available('resolutionMode') && <details className="create-prompt-options"><summary>Image size <span>{String(value('resolutionMode')).split(' · ')[0]}</span></summary>
            {selectField('resolutionMode')}{presetSize && selectField('aspectRatio')}{usesMegapixels && numberField('megapixels')}
            {customSize && <div className="create-control-pair">{numberField('width')}{numberField('height')}</div>}
          </details>}
          {(available('stylePrompt') || available('negativePrompt')) && <details className="create-prompt-options"><summary>Style & exclusions <span>Optional</span></summary>
            {textField('stylePrompt')}{textField('negativePrompt', 'Avoid in the image')}
          </details>}
          <button type="button" className="create-advanced-link" onClick={onAdvanced}>Open full workflow editor <span aria-hidden="true">↗</span></button>
        </>}
        {inputError && <p className="create-message error" role="alert">{inputError}</p>}
      </div>
      <footer className="create-generate-footer">
        <button type="submit" className="create-primary create-generate" disabled={locked || !!busy || (editing && !hasReference)}>
          <SparkIcon /><span>{busy === 'generate' ? 'Adding to queue…' : editing ? 'Generate edit' : 'Generate image'}</span>
          {Number(value('batchSize')) > 1 && <small>×{value('batchSize')}</small>}
        </button>
        <p>{busy === 'patch' ? 'Updating workflow…' : editing && !hasReference ? 'Upload a reference image to begin.' : connected ? 'Your image will appear in Results.' : 'Connect the studio to generate.'}</p>
      </footer>
    </form>
  </aside>
}
