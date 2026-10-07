import { useEffect, useRef, useState } from 'react'
import { getEnhancerStatus, startEnhancer, cancelEnhancer, createErrorMessage, formatCreateBytes } from '../services/donutCreate'

export default function CreatePromptEnhancer({ sessionId, imageBrief, faceBrief, disabled, unavailableReason, onApply, onPauseInstant }) {
  const [open, setOpen] = useState(false)
  const [status, setStatus] = useState(null)
  const [model, setModel] = useState('minicpm')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [originals, setOriginals] = useState(null)
  const [applied, setApplied] = useState(false)
  const alive = useRef(false)
  const operation = useRef(null)
  const requestRevision = useRef(0)
  const current = useRef({ imageBrief, faceBrief })
  current.current = { imageBrief, faceBrief }
  const selected = status?.models?.find(item => item.id === model)
  const job = status?.job
  const result = job?.id === operation.current && job?.kind === 'enhance' && job.state === 'complete' && originals ? job.result : null
  const stale = originals && (originals.imageBrief !== imageBrief || originals.faceBrief !== faceBrief)
  const working = pending || status?.busy

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
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
        if (!controller.signal.aborted && revision === requestRevision.current) {
          setStatus(next)
        }
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

  async function start(kind) {
    if (pending) return
    const source = { ...current.current }
    requestRevision.current += 1
    operation.current = null
    onPauseInstant?.()
    setPending(true)
    setError('')
    setApplied(false)
    setOriginals(kind === 'enhance' ? source : null)
    try {
      const next = await startEnhancer(sessionId, { kind, model,
        ...(kind === 'enhance' ? { image_brief: source.imageBrief, face_brief: source.faceBrief } : {}) })
      if (!alive.current) {
        if (next.job?.id) await cancelEnhancer(sessionId, next.job.id)
        return
      }
      operation.current = next.job?.id
      requestRevision.current += 1
      setStatus(next)
    } catch (failure) {
      if (alive.current) setError(createErrorMessage(failure))
    } finally {
      if (alive.current) setPending(false)
    }
  }

  async function cancel() {
    requestRevision.current += 1
    setPending(true)
    try {
      const next = await cancelEnhancer(sessionId, job.id)
      if (alive.current) { requestRevision.current += 1; setStatus(next); setOriginals(null); operation.current = null }
    } catch (failure) {
      if (alive.current) setError(createErrorMessage(failure))
    } finally { if (alive.current) setPending(false) }
  }

  async function apply() {
    if (!result || stale) return
    setPending(true)
    setError('')
    try {
      // The parent checks the latest drafts again immediately before its atomic patch.
      onPauseInstant?.()
      await onApply(result, originals)
      if (alive.current) setApplied(true)
    } catch (failure) {
      if (alive.current) setError(createErrorMessage(failure))
    } finally { if (alive.current) setPending(false) }
  }

  return <details className="create-enhancer" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Prompt enhancer</summary>
    {open && <div>
      <p className="create-control-note">Runs locally on the computer hosting Create. Downloads once, then works offline. Uses both your image and face descriptions. Review before applying.</p>
      <label className="create-field">Enhancer model<select value={model} disabled={working} onChange={event => { setModel(event.target.value); setOriginals(null) }}>
        {(status?.models || [{ id: 'minicpm', name: 'MiniCPM5-2B · loading…' }]).map(item => <option key={item.id} value={item.id}>{item.name}{item.size ? ` · ${item.quantization} · ${formatCreateBytes(item.size)}` : ''}</option>)}
      </select></label>
      {unavailableReason && <p role="status" className="create-control-note">{unavailableReason}</p>}
      {status?.supported === false && <p role="alert">The managed enhancer is unavailable on this platform.</p>}
      <div className="create-enhancer-actions">
        <button type="button" disabled={!selected || working || !status.supported} onClick={() => start('install')}>{selected?.ready ? 'Repair / verify model' : 'Download model'}</button>
        <button type="button" disabled={!selected?.ready || working || disabled || !imageBrief.trim()} onClick={() => start('enhance')}>Enhance prompts</button>
        {job?.state === 'running' && <button type="button" disabled={pending} onClick={cancel}>Cancel enhancement / setup</button>}
      </div>
      {working && <p role="status">{job?.state !== 'running' ? 'Enhancer is in use by another studio.' : job.kind === 'install' ? `Setting up model… ${Math.round((status.setup?.progress || 0) * 100)}%` : 'Enhancing both prompts…'}</p>}
      {job?.state === 'cancelled' && <p role="status">Cancelled. Downloads can resume; your original prompts are unchanged.</p>}
      {(error || job?.error) && <p className="create-error" role="alert">{error || job.error}</p>}
      {result && <section className="create-enhancer-preview" aria-label="Enhanced prompt preview">
        <h3>Review enhanced prompts</h3>
        <label className="create-field">Enhanced image prompt<textarea readOnly rows={6} value={result.main_prompt} /></label>
        <label className="create-field">Enhanced face prompt<textarea readOnly rows={3} value={result.face_prompt} /></label>
        {!result.face_prompt && <p className="create-control-note">No single stated face was found. The face prompt will be empty.</p>}
        <details><summary>Original prompts</summary><p className="create-enhancer-original">{originals.imageBrief}</p><p className="create-enhancer-original">{originals.faceBrief || 'Face prompt was empty.'}</p></details>
        {stale && !applied && <p role="status">Your prompts changed. Enhance again to use the latest edits.</p>}
        <div className="create-enhancer-actions"><button type="button" disabled={disabled || working || stale || applied} onClick={apply}>Apply both prompts</button>
          <button type="button" disabled={pending} onClick={cancel}>Discard preview</button></div>
        {applied && <p role="status">Both prompts applied. Generate when you are ready.</p>}
      </section>}
    </div>}
  </details>
}
