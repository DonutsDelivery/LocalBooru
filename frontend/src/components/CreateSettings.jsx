import { useCallback, useEffect, useRef, useState } from 'react'
import { getAddon, installAddon, startAddon } from '../api'
import {
  cancelCreateSetup,
  createErrorMessage,
  formatCreateBytes,
  getCreateStatus,
  saveCreateConfig,
  setupCreateBackend,
  startCreateBackend,
  stopCreateBackend,
  restartCreateBackend,
} from '../services/donutCreate'
import './CreateStudio.css'

export default function CreateSettings({ onStatusChange }) {
  const [addon, setAddon] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [action, setAction] = useState(null)
  const [error, setError] = useState('')
  const [mode, setMode] = useState('managed')
  const [backendUrl, setBackendUrl] = useState('http://127.0.0.1:8188')
  const [comfyDirectory, setComfyDirectory] = useState('')
  const [pythonExecutable, setPythonExecutable] = useState('')
  const [runtime, setRuntime] = useState('cuda')
  const [profile, setProfile] = useState('base')
  const [hfToken, setHfToken] = useState('')
  const [civitaiKey, setCivitaiKey] = useState('')
  const [switchingServer, setSwitchingServer] = useState(false)
  const initialized = useRef(false)
  const generation = useRef(0)
  const requests = useRef(new AbortController())
  const serverChanging = useRef(false)
  const mounted = useRef(false)
  const actionLock = useRef(false)
  const actionRevision = useRef(0)
  const refreshGeneration = useRef(null)
  const statusCallback = useRef(onStatusChange)

  useEffect(() => { statusCallback.current = onStatusChange }, [onStatusChange])

  const refresh = useCallback(async () => {
    if (serverChanging.current || refreshGeneration.current === generation.current) return
    const requestGeneration = generation.current
    const requestRevision = actionRevision.current
    const signal = requests.current.signal
    refreshGeneration.current = requestGeneration
    try {
      const response = await getAddon('donut-create', { signal })
      const nextAddon = response.addon
      if (!mounted.current || requestGeneration !== generation.current || requestRevision !== actionRevision.current) return
      setAddon(nextAddon)
      if (nextAddon.installed && nextAddon.status === 'running') {
        const nextStatus = await getCreateStatus(signal)
        if (!mounted.current || requestGeneration !== generation.current || requestRevision !== actionRevision.current) return
        setStatus(nextStatus)
        if (!initialized.current) {
          initialized.current = true
          setMode(nextStatus.mode || 'managed')
          setBackendUrl(nextStatus.mode === 'existing' ? nextStatus.backend_url || 'http://127.0.0.1:8188' : 'http://127.0.0.1:8188')
          setComfyDirectory(nextStatus.comfy_directory || '')
          setPythonExecutable(nextStatus.python_executable || '')
          setRuntime(nextStatus.setup?.default_runtime || 'cuda')
          setProfile(nextStatus.setup?.default_profile || 'base')
        }
        statusCallback.current?.(nextStatus)
      } else {
        setStatus(null)
        statusCallback.current?.(null)
      }
    } finally {
      if (refreshGeneration.current === requestGeneration) refreshGeneration.current = null
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    requests.current = new AbortController()
    const poll = () => {
      if (serverChanging.current || refreshGeneration.current === generation.current) return
      const requestGeneration = generation.current
      return refresh()
        .catch(loadError => { if (mounted.current && requestGeneration === generation.current) setError(createErrorMessage(loadError)) })
        .finally(() => { if (mounted.current && requestGeneration === generation.current) setLoading(false) })
    }
    const changingServer = () => {
      generation.current += 1
      requests.current.abort()
      requests.current = new AbortController()
      serverChanging.current = true
      initialized.current = false
      actionLock.current = false
      actionRevision.current += 1
      setSwitchingServer(true)
      setLoading(true)
      setAddon(null)
      setStatus(null)
      setAction(null)
      setError('')
      setHfToken('')
      setCivitaiKey('')
      statusCallback.current?.(null)
    }
    const changedServer = () => {
      serverChanging.current = false
      setSwitchingServer(false)
      poll()
    }
    poll()
    const timer = setInterval(poll, 2500)
    window.addEventListener('localbooru-addons-changed', poll)
    window.addEventListener('donut-create-server-changing', changingServer)
    window.addEventListener('donut-create-server-changed', changedServer)
    return () => {
      mounted.current = false
      generation.current += 1
      requests.current.abort()
      clearInterval(timer)
      window.removeEventListener('localbooru-addons-changed', poll)
      window.removeEventListener('donut-create-server-changing', changingServer)
      window.removeEventListener('donut-create-server-changed', changedServer)
    }
  }, [refresh])

  async function runAction(label, operation, interrupt = false) {
    if ((actionLock.current && !interrupt) || serverChanging.current) return
    if (interrupt) {
      generation.current += 1
      requests.current.abort()
      requests.current = new AbortController()
    }
    actionLock.current = true
    const operationRevision = ++actionRevision.current
    const requestGeneration = generation.current
    const signal = requests.current.signal
    const ensureCurrent = () => {
      if (!mounted.current || requestGeneration !== generation.current || operationRevision !== actionRevision.current || signal.aborted) {
        throw new DOMException('The server connection changed.', 'AbortError')
      }
    }
    setAction(label)
    setError('')
    try {
      await operation(signal, ensureCurrent)
      ensureCurrent()
      await refresh()
    } catch (actionError) {
      if (mounted.current && requestGeneration === generation.current && operationRevision === actionRevision.current) setError(createErrorMessage(actionError))
    } finally {
      if (requestGeneration === generation.current && operationRevision === actionRevision.current) {
        actionLock.current = false
        if (mounted.current) setAction(null)
      }
    }
  }

  async function activateAddon(signal, ensureCurrent) {
    if (!addon?.installed) await installAddon('donut-create', undefined, { signal })
    ensureCurrent()
    await startAddon('donut-create', { signal })
    ensureCurrent()
    window.dispatchEvent(new CustomEvent('localbooru-addons-changed'))
  }

  function installRuntime() {
    const credentials = {
      ...(hfToken.trim() ? { hf_token: hfToken.trim() } : {}),
      ...(civitaiKey.trim() ? { civitai_api_key: civitaiKey.trim() } : {}),
    }
    // Tokens are supplied only to this setup request and never saved as config.
    setHfToken('')
    setCivitaiKey('')
    runAction('Installing', async (signal, ensureCurrent) => {
      if (status?.mode !== 'managed') await saveCreateConfig({ mode: 'managed' }, signal)
      ensureCurrent()
      await setupCreateBackend({ runtime, profile, ...credentials }, signal)
    })
  }

  function configureBackend() {
    runAction('Connecting', async (signal, ensureCurrent) => {
      const nextStatus = await saveCreateConfig({ mode,
        ...(mode === 'existing' ? { backend_url: backendUrl.trim() }
          : mode === 'local' ? { comfy_directory: comfyDirectory.trim(),
            ...(pythonExecutable.trim() ? { python_executable: pythonExecutable.trim() } : {}) } : {}),
      }, signal)
      ensureCurrent()
      setComfyDirectory(nextStatus.comfy_directory || '')
      setPythonExecutable(nextStatus.python_executable || '')
      setStatus(nextStatus)
      statusCallback.current?.(nextStatus)
    })
  }

  const setup = status?.setup || {}
  const installing = setup.running === true
  const busy = loading || action !== null || switchingServer
  const active = addon?.installed && addon.status === 'running'
  const configured = status?.mode === mode && (mode !== 'existing' || status.backend_url === backendUrl.trim())
    && (mode !== 'local' || (status.comfy_directory === comfyDirectory.trim() && (status.python_executable || '') === pythonExecutable.trim()))
  const controllable = ['managed', 'local'].includes(status?.mode)
  const catalog = status?.catalog || setup.catalog || {}
  const profiles = catalog.profiles || [
    { id: 'base', label: 'Neutral starter', description: 'Workflow v5 with Krea2 and no aesthetic LoRA.' },
    { id: 'workflow', label: 'Original workflow v5', description: 'The original v5 model and LoRA selections.' },
    { id: 'all', label: 'Complete catalog', description: 'All bundled workflow model choices.' },
  ]
  const selection = profiles.find(option => option.id === profile) || {}
  const downloadBytes = selection.download_bytes ?? selection.total_bytes
  const diskBytes = selection.required_bytes ?? (setup.profile === profile ? setup.disk?.required_bytes : null)
  const supportedRuntimes = setup.supported_runtimes || ['cuda', 'cpu', 'mps']
  const downloadedBytes = setup.downloaded_bytes ?? setup.completed_bytes ?? 0
  const totalBytes = setup.total_bytes || 0
  const progress = Number.isFinite(setup.progress) ? Math.min(100, Math.round(setup.progress * 100))
    : totalBytes > 0 ? Math.min(100, Math.round(downloadedBytes / totalBytes * 100)) : null
  const setupError = setup.error

  return (
    <section className="create-settings">
      <h2>Donut Create</h2>
      <p className="settings-description">Create images with the DonutUI studio and DonutNodes workflow v5.</p>
      {error && <p className="create-message error" role="alert">{error}</p>}
      {switchingServer && <p className="create-message" role="status">Connecting to the selected server…</p>}

      <div className="create-settings-card">
        <h3>Creator add-on</h3>
        <p>The optional add-on controls the studio and an isolated ComfyUI installation.</p>
        <p className="create-download-note">The neutral starter requires about 32 GB of models, the original workflow v5 selections about 46 GB, and the complete catalog about 63 GB, plus ComfyUI and its dependencies.</p>
        {!active && (
          <button type="button" className="create-primary" disabled={busy || addon?.status === 'starting'} onClick={() => runAction('Activating', activateAddon)}>
            {action === 'Activating' || addon?.status === 'starting' ? 'Starting creator…' : addon?.installed ? 'Start creator add-on' : 'Install creator add-on'}
          </button>
        )}
        {active && <span className="create-ready">Creator add-on is running</span>}
      </div>

      <fieldset className="create-settings-card" disabled={!active || busy || installing}>
        <legend>ComfyUI backend</legend>
        <label className="create-radio"><input type="radio" name="create-backend" value="managed" checked={mode === 'managed'} onChange={() => setMode('managed')} />Managed ComfyUI</label>
        <label className="create-radio"><input type="radio" name="create-backend" value="local" checked={mode === 'local'} onChange={() => setMode('local')} />Manage existing local installation</label>
        <label className="create-radio"><input type="radio" name="create-backend" value="existing" checked={mode === 'existing'} onChange={() => setMode('existing')} />Connect existing ComfyUI</label>
        {mode === 'existing' && (
          <label className="create-field">Backend URL
            <input type="url" value={backendUrl} placeholder="http://127.0.0.1:8188" onChange={event => setBackendUrl(event.target.value)} />
          </label>
        )}
        {mode === 'local' && <>
          <label className="create-field">ComfyUI folder on this server<input value={comfyDirectory} placeholder="Folder containing main.py" onChange={event => setComfyDirectory(event.target.value)} /></label>
          <label className="create-field">Python executable<input value={pythonExecutable} placeholder="Auto-detect the installation’s venv" onChange={event => setPythonExecutable(event.target.value)} /></label>
          <p className="create-download-note">Uses your installed ComfyUI, models and node packs. DMC starts it on port 18010 and manages the process it launches.</p>
        </>}
        <button type="button" disabled={configured || status?.backend?.owned || (mode === 'local' && !comfyDirectory.trim()) || (mode === 'existing' && !/^https?:\/\//i.test(backendUrl.trim()))} onClick={configureBackend}>
          {action === 'Connecting' ? 'Connecting…' : mode === 'existing' ? 'Connect backend' : mode === 'local' ? 'Use this installation' : 'Use managed backend'}
        </button>
        {status?.backend?.owned && !configured && <p className="create-download-note">Stop ComfyUI before changing its backend configuration.</p>}
      </fieldset>

      {mode === 'managed' && (
        <div className="create-settings-card">
          <h3>Install ComfyUI, node packs and models</h3>
          <div className="create-setting-fields">
            <label className="create-field">Runtime
              <select value={runtime} disabled={!active || busy || installing} onChange={event => setRuntime(event.target.value)}>
                {supportedRuntimes.map(option => <option key={option} value={option}>{({ cuda: 'NVIDIA CUDA', cpu: 'CPU', mps: 'Apple Silicon' })[option] || option}</option>)}
              </select>
            </label>
            <label className="create-field">Models
              <select value={profile} disabled={!active || busy || installing} onChange={event => setProfile(event.target.value)}>
                {profiles.map(option => <option key={option.id} value={option.id}>{option.label}</option>)}
              </select>
            </label>
          </div>
          <p className="create-download-note">
            {downloadBytes != null ? `${formatCreateBytes(downloadBytes)} download` : `About ${({ base: '32', workflow: '46', all: '63' })[profile]} GB of models`}
            {selection.model_count != null && ` · ${selection.model_count} models`}
            {diskBytes != null && ` · ${formatCreateBytes(diskBytes)} disk required`}
            {setup.disk?.available_bytes != null && ` · ${formatCreateBytes(setup.disk.available_bytes)} available`}
          </p>
          {selection.description && <p>{selection.description}</p>}
          <details className="create-token-fields">
            <summary>Model access tokens (optional)</summary>
            <label className="create-field">Hugging Face token<input type="password" autoComplete="off" value={hfToken} disabled={busy || installing} onChange={event => setHfToken(event.target.value)} /></label>
            <label className="create-field">Civitai API key<input type="password" autoComplete="off" value={civitaiKey} disabled={busy || installing} onChange={event => setCivitaiKey(event.target.value)} /></label>
            <p>Tokens are used for this setup request and are not saved.</p>
          </details>
          <div className="create-actions">
            <button type="button" className="create-primary" disabled={!active || busy || installing} onClick={installRuntime}>
              {action === 'Installing' ? 'Starting setup…' : ['failed', 'error', 'cancelled'].includes(setup.state) ? 'Retry setup' : 'Install selected runtime and models'}
            </button>
            {installing && <button type="button" disabled={action === 'Cancelling'} onClick={() => runAction('Cancelling', cancelCreateSetup, true)}>{action === 'Cancelling' ? 'Cancelling…' : 'Cancel setup'}</button>}
          </div>
          {(installing || setup.state) && (
            <div className="create-setup-progress" role="status" aria-live="polite">
              <strong>{setup.message || setup.phase || setup.state || 'Preparing setup'}</strong>
              {progress !== null && <progress value={progress} max="100" aria-label="Setup download progress" />}
              {totalBytes > 0 && <span>{formatCreateBytes(downloadedBytes)} of {formatCreateBytes(totalBytes)}{progress !== null && ` (${progress}%)`}</span>}
              {setup.current_file && <span>{setup.current_file}</span>}
              {setup.state === 'cancelled' && <span>Setup cancelled. Verified downloads will be reused when you retry.</span>}
              {setupError && <p className="create-message error" role="alert">{setupError}</p>}
            </div>
          )}
        </div>
      )}

      {active && (
        <div className="create-settings-card">
          <h3>Backend status</h3>
          <p role="status">{action || (status?.backend?.ready ? 'Ready for image creation' : status?.backend?.owned && !status?.backend?.running ? 'ComfyUI is starting… This can take several minutes.' : status?.backend?.running ? 'Checking workflow nodes and models…' : 'Backend is stopped or unavailable')}</p>
          {status?.backend?.error && <p className="create-message error" role="alert">{status.backend.error}</p>}
          {status?.backend?.missing_nodes?.length > 0 && <p>Missing node packs or nodes: {status.backend.missing_nodes.join(', ')}</p>}
          {status?.backend?.missing_models?.length > 0 && <p>Missing models: {status.backend.missing_models.join(', ')}</p>}
          {!controllable && <p>To start, stop and restart from DMC, select Manage existing local installation on the computer hosting ComfyUI.</p>}
          {controllable && (
            <div className="create-actions">
              <button type="button" disabled={busy || installing || status?.backend?.running || status?.backend?.owned} onClick={() => runAction('Starting backend', startCreateBackend)}>{action === 'Starting backend' ? 'Starting backend…' : 'Start ComfyUI'}</button>
              <button type="button" disabled={loading || switchingServer || installing || !status?.backend?.owned || (action !== null && action !== 'Starting backend')} onClick={() => runAction('Stopping backend', stopCreateBackend, true)}>{action === 'Stopping backend' ? 'Stopping backend…' : 'Stop ComfyUI'}</button>
              <button type="button" disabled={busy || installing || !status?.backend?.owned} onClick={() => runAction('Restarting backend', restartCreateBackend)}>{action === 'Restarting backend' ? 'Restarting backend…' : 'Restart ComfyUI'}</button>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
