import { useEffect, useState } from 'react'
import {
  downloadModel,
  getAddonHealth,
  getAutoTaggerConfig,
  getModels,
  updateAutoTaggerConfig,
} from '../api'
import {
  formatExecutionState,
  formatPackageVersions,
  formatPreload,
  formatProviderList,
  formatProviderMetric,
  formatTimings,
} from './autoTaggerRuntime'
import './OpticalFlowSettings.css'

const TAGGER_MODEL_IDS = new Set(['vit-v3', 'eva02-large-v3', 'swinv2-v3'])

export default function AutoTaggerSettings() {
  const [config, setConfig] = useState(null)
  const [models, setModels] = useState([])
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [refreshingHealth, setRefreshingHealth] = useState(false)
  const [downloading, setDownloading] = useState(null)

  const load = async () => {
    try {
      const [nextConfig, modelResponse] = await Promise.all([
        getAutoTaggerConfig(),
        getModels(),
      ])
      setConfig(nextConfig)
      setModels((modelResponse.models || []).filter((model) => TAGGER_MODEL_IDS.has(model.name)))
      try {
        setHealth(await getAddonHealth('auto-tagger'))
        setHealthError(null)
      } catch (error) {
        setHealth(null)
        setHealthError(error.message || 'Runtime status is unavailable')
      }
    } catch (error) {
      console.error('Failed to load Auto Tagger settings:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const save = async () => {
    setSaving(true)
    try {
      const next = await updateAutoTaggerConfig(config)
      setConfig(next)
      await load()
    } catch (error) {
      console.error('Failed to save Auto Tagger settings:', error)
    } finally {
      setSaving(false)
    }
  }

  const refreshHealth = async () => {
    setRefreshingHealth(true)
    try {
      setHealth(await getAddonHealth('auto-tagger'))
      setHealthError(null)
    } catch (error) {
      console.error('Failed to refresh Auto Tagger runtime status:', error)
      setHealth(null)
      setHealthError(error.message || 'Runtime status is unavailable')
    } finally {
      setRefreshingHealth(false)
    }
  }

  const download = async (modelName) => {
    setDownloading(modelName)
    try {
      await downloadModel(modelName)
      const interval = window.setInterval(async () => {
        const response = await getModels()
        const nextModels = (response.models || []).filter((model) => TAGGER_MODEL_IDS.has(model.name))
        setModels(nextModels)
        const model = nextModels.find((entry) => entry.name === modelName)
        if (model?.status !== 'downloading') {
          window.clearInterval(interval)
          setDownloading(null)
        }
      }, 1000)
    } catch (error) {
      console.error('Failed to download Auto Tagger model:', error)
      setDownloading(null)
    }
  }

  if (loading || !config) {
    return <section className="optical-flow-settings"><h2>Auto Tagger</h2><p className="settings-description">Loading...</p></section>
  }

  const requestedDevice = health?.requested_device || config.device
  const availableProviders = formatProviderList(health?.available_providers)
  const registeredProviders = formatProviderList(health?.registered_providers)
  const observedExecution = formatExecutionState(health?.execution_state)
  const providerNodeCounts = formatProviderMetric(health?.provider_node_counts)
  const providerDurations = formatProviderMetric(health?.provider_duration_ms, 'ms')
  const lastTimings = formatTimings(health?.last_timings_ms)
  const diagnostics = health?.runtime_diagnostics
  const deployment = diagnostics?.deployment
  const runtimePackages = formatPackageVersions(diagnostics?.packages)
  const preload = formatPreload(diagnostics?.preload)
  const modelIdentity = health?.model_identity

  return (
    <section className="optical-flow-settings">
      <h2>Auto Tagger</h2>
      <p className="settings-description">Configure the model, confidence thresholds, and inference device used for automatic tagging.</p>

      <section className="settings-section">
        <h3>Model</h3>
        {models.map((model) => (
          <div key={model.name} className="setting-row">
            <label>
              <input
                type="radio"
                name="tagger-model"
                value={model.name}
                checked={config.model === model.name}
                disabled={model.status !== 'downloaded'}
                onChange={() => setConfig((current) => ({ ...current, model: model.name }))}
              />
              <span>{model.display_name}</span>
            </label>
            <span className={`backend-badge ${model.status === 'downloaded' ? 'available' : 'unavailable'}`}>
              {model.status === 'downloaded' ? 'Downloaded' : model.status}
            </span>
            {model.status !== 'downloaded' && (
              <button onClick={() => download(model.name)} disabled={downloading === model.name}>
                {downloading === model.name ? 'Downloading...' : 'Download'}
              </button>
            )}
          </div>
        ))}
      </section>

      <section className="settings-section">
        <h3>Inference</h3>
        <div className="setting-row">
          <label>Device
            <select value={config.device} onChange={(event) => setConfig((current) => ({ ...current, device: event.target.value }))}>
              <option value="auto">Auto</option>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
        </div>
        <div className="setting-row">
          <label>General confidence
            <input type="number" min="0" max="1" step="0.01" value={config.general_threshold}
              onChange={(event) => setConfig((current) => ({ ...current, general_threshold: Number(event.target.value) }))} />
          </label>
          <label>Character confidence
            <input type="number" min="0" max="1" step="0.01" value={config.character_threshold}
              onChange={(event) => setConfig((current) => ({ ...current, character_threshold: Number(event.target.value) }))} />
          </label>
        </div>
      </section>

      <section className="settings-section">
        <h3>Runtime</h3>
        <p className="setting-note">Requested device: {requestedDevice}</p>
        <p className="setting-note">Runtime available: {availableProviders}</p>
        <p className="setting-note">Session registered: {registeredProviders}</p>
        <p className="setting-note">Observed execution: {observedExecution}</p>
        <p className="setting-note">Provider node counts: {providerNodeCounts}</p>
        <p className="setting-note">Provider durations: {providerDurations}</p>
        <p className="setting-note">Last prediction timings: {lastTimings}</p>
        <p className="setting-note">Deployment revision: {deployment?.installed_revision || 'Not recorded'} / desired {deployment?.desired_revision || 'Not available'}</p>
        <p className="setting-note">Deployment runtime: {deployment?.runtime || 'Not available'}</p>
        <p className="setting-note">Python / ONNX Runtime: {diagnostics?.python_version || 'Not available'} / {diagnostics?.onnxruntime_version || 'Not available'}</p>
        <p className="setting-note">Platform: {diagnostics?.platform || 'Not available'} ({diagnostics?.architecture || 'unknown architecture'})</p>
        <p className="setting-note">Runtime packages: {runtimePackages}</p>
        <p className="setting-note">Native library preload: {preload}</p>
        <p className="setting-note">Model identity: {modelIdentity ? `${modelIdentity.name} — ${modelIdentity.sha256 || 'hash unavailable'} (${modelIdentity.bytes ?? 'size unavailable'} bytes)` : 'Not loaded'}</p>
        <p className="setting-note">Provider options: {diagnostics?.provider_options ? JSON.stringify(diagnostics.provider_options) : 'Not available'}</p>
        {healthError && <p className="optical-flow-status warning">Runtime status unavailable: {healthError}</p>}
        {deployment?.warning && <p className="optical-flow-status warning">{deployment.warning}</p>}
        {diagnostics?.ort_debug_error && <p className="optical-flow-status warning">ORT diagnostics: {diagnostics.ort_debug_error}</p>}
        {health?.provider_warning && <p className="optical-flow-status warning">{health.provider_warning}</p>}
        {health?.profile_warning && <p className="optical-flow-status warning">{health.profile_warning}</p>}
        <button onClick={refreshHealth} disabled={refreshingHealth}>
          {refreshingHealth ? 'Refreshing...' : 'Refresh Runtime Status'}
        </button>
      </section>

      <button onClick={save} disabled={saving}>{saving ? 'Saving...' : 'Save Auto Tagger settings'}</button>
    </section>
  )
}
