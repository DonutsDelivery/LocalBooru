import { useEffect, useRef, useState } from 'react'
import { getEnhancerStatus, startEnhancer, cancelEnhancer, createErrorMessage, formatCreateBytes } from '../services/donutCreate'

function facePositionText(position) {
  if (typeof position === 'string') return position
  const descriptions = []
  if (position?.left_of?.length) descriptions.push(`left of ${position.left_of.join(', ')}`)
  if (position?.above?.length) descriptions.push(`above ${position.above.join(', ')}`)
  return descriptions.join('; ') || 'no additional position details'
}

export default function CreatePromptEnhancer({ sessionId, config, disabled, unavailableReason, onConfigure, onPauseInstant }) {
  const modelRoles = { qwen: 'Rewriter', minicpm: 'Preserver' }
  const [open, setOpen] = useState(false)
  const [status, setStatus] = useState(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const alive = useRef(false)
  const operation = useRef(null)
  const requestRevision = useRef(0)
  const model = config?.model || 'minicpm'
  const enabled = config?.enabled === true
  const selected = status?.models?.find(item => item.id === model)
  const job = status?.job
  const working = pending || status?.busy
  const result = config?.lastRun
  const faces = Array.isArray(result?.face_prompts) && result.face_prompts.length > 1 ? result.face_prompts : []
  const relativeFaces = faces.length > 0 && typeof faces[0].position === 'object'

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      // Per-run work is owned by the queue adapter, not this settings panel.
      if (operation.current) cancelEnhancer(sessionId, operation.current).catch(() => {})
    }
  }, [sessionId])

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    let timer
    async function refresh() {
      const revision = requestRevision.current
      try {
        const next = await getEnhancerStatus(sessionId, controller.signal)
        if (!controller.signal.aborted && revision === requestRevision.current) setStatus(next)
        if (!controller.signal.aborted) timer = setTimeout(refresh, next.busy ? 1000 : 5000)
      } catch (failure) {
        if (!controller.signal.aborted) {
          setError(createErrorMessage(failure))
          timer = setTimeout(refresh, 5000)
        }
      }
    }
    refresh()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [open, sessionId])

  async function configure(next) {
    setPending(true)
    setError('')
    try {
      onPauseInstant?.()
      await onConfigure(next)
    } catch (failure) {
      if (alive.current) setError(createErrorMessage(failure))
    } finally { if (alive.current) setPending(false) }
  }

  async function install() {
    if (pending) return
    requestRevision.current += 1
    onPauseInstant?.()
    setPending(true)
    setError('')
    try {
      const next = await startEnhancer(sessionId, { kind: 'install', model })
      if (!alive.current) {
        if (next.job?.id) await cancelEnhancer(sessionId, next.job.id)
        return
      }
      operation.current = next.job?.id
      requestRevision.current += 1
      setStatus(next)
    } catch (failure) {
      if (alive.current) setError(createErrorMessage(failure))
    } finally { if (alive.current) setPending(false) }
  }

  async function cancel() {
    requestRevision.current += 1
    setPending(true)
    try {
      const next = await cancelEnhancer(sessionId, job.id)
      if (alive.current) { requestRevision.current += 1; setStatus(next); operation.current = null }
    } catch (failure) {
      if (alive.current) setError(createErrorMessage(failure))
    } finally { if (alive.current) setPending(false) }
  }

  return <details className="create-enhancer" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Prompt enhancer</summary>
    {open && <div>
      <p className="create-control-note">Runs locally before every image run when enabled. Uses your image and face briefs with that run’s seed; your original inputs stay unchanged. Ask it to invent or vary details within your brief. Downloads once, then works offline. Keeps the selected model in system RAM between runs for faster starts, leaving GPU memory for image generation.</p>
      <p className="create-control-note">Leave the face brief blank to derive one identifiable visible face from Describe your image. A supplied face brief takes priority; scenes with hidden or ambiguous faces keep the face prompt empty.</p>
      <p className="create-control-note">For multiple faces, describe which characters are left of or above others, or give facial descriptions explicitly from left to right. Each visible character keeps a separate face prompt.</p>
      <label className="create-field">Enhancer model<select value={model} disabled={working || !onConfigure} onChange={event => configure({ enabled, model: event.target.value })}>
        {(status?.models || [{ id: 'minicpm', name: 'MiniCPM5-2B · loading…' }]).map(item => <option key={item.id} value={item.id}>{modelRoles[item.id] ? `${modelRoles[item.id]} · ` : ''}{item.name}{item.size ? ` · ${item.quantization} · ${formatCreateBytes(item.size)}` : ''}</option>)}
      </select></label>
      <label className="create-instant-toggle"><input type="checkbox" checked={enabled} disabled={working || (!enabled && (disabled || !selected?.ready || status?.supported === false)) || !onConfigure} onChange={event => configure({ enabled: event.target.checked, model })} />Enhance every run</label>
      {unavailableReason && <p role="status" className="create-control-note">{unavailableReason}</p>}
      {status?.loaded_model === model && !working && <p role="status" className="create-control-note">Model loaded in RAM · Ready for the next run.</p>}
      {status?.supported === false && <p role="alert">The managed enhancer is unavailable on this platform.</p>}
      <div className="create-enhancer-actions">
        <button type="button" disabled={!selected || working || !status.supported} onClick={install}>{selected?.ready ? 'Repair / verify model' : 'Download model'}</button>
        {job?.state === 'running' && <button type="button" disabled={pending} onClick={cancel}>Cancel enhancement / setup</button>}
      </div>
      {working && <p role="status">{job?.state !== 'running' ? 'Enhancer is in use by another studio.' : job.kind === 'install' ? `Setting up model… ${Math.round((status.setup?.progress || 0) * 100)}%` : 'Enhancing this run’s prompts…'}</p>}
      {job?.state === 'cancelled' && <p role="status">Cancelled. Your original prompts are unchanged.</p>}
      {(error || job?.error) && <p className="create-error" role="alert">{error || job.error}</p>}
      {result && <section className="create-enhancer-preview" aria-label="Enhanced prompt preview">
        <h3>Last run’s enhanced prompts</h3>
        <p className="create-control-note">{modelRoles[result.model] || result.model} · Seed {result.seed}</p>
        <label className="create-field">Enhanced image prompt<textarea readOnly rows={6} value={result.main_prompt} /></label>
        {faces.length ? <section aria-label="Ordered character face prompts">
          <p className="create-control-note">{relativeFaces ? 'Character face prompts with relative positions.' : 'Character face prompts, ordered from left to right.'}</p>
          {faces.map((face, index) => <div key={`${index}-${face.character}`}>
            <h4>{face.character} · {facePositionText(face.position)}</h4>
            <label className="create-field">Enhanced face prompt for {face.character}<textarea readOnly rows={3} value={face.face_prompt} /></label>
          </div>)}
        </section> : <>
          <label className="create-field">Enhanced face prompt<textarea readOnly rows={3} value={result.face_prompt} /></label>
          {!result.face_prompt && <p className="create-control-note">{Array.isArray(result.face_prompts) ? 'No face plan was resolved for this run.' : 'This run used an empty face prompt.'}</p>}
        </>}
      </section>}
    </div>}
  </details>
}
