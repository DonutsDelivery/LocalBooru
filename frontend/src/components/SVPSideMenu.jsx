import { useState, useEffect } from 'react'
import {
  getSVPConfig,
  updateSVPConfig
} from '../api'
import './SVPSideMenu.css'

export default function SVPSideMenu({ isOpen, onClose }) {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Local form state
  const [enabled, setEnabled] = useState(false)
  const [targetFps, setTargetFps] = useState(60)
  const [preset, setPreset] = useState('balanced')
  // Key settings
  const [useNvof, setUseNvof] = useState(true)
  const [shader, setShader] = useState(23)
  const [artifactMasking, setArtifactMasking] = useState(100)
  const [frameInterpolation, setFrameInterpolation] = useState(2)
  // Advanced custom params
  const [customSuper, setCustomSuper] = useState('')
  const [customAnalyse, setCustomAnalyse] = useState('')
  const [customSmooth, setCustomSmooth] = useState('')

  useEffect(() => {
    if (isOpen) {
      loadConfig()
    }
  }, [isOpen])

  async function loadConfig() {
    try {
      setLoading(true)
      const data = await getSVPConfig()
      setConfig(data)
      setEnabled(data.enabled || false)
      setTargetFps(data.target_fps || 60)
      setPreset(data.preset || 'balanced')
      setUseNvof(data.use_nvof !== false)
      setShader(data.shader || 23)
      setArtifactMasking(data.artifact_masking ?? 100)
      setFrameInterpolation(data.frame_interpolation ?? 2)
      setCustomSuper(data.custom_super || '')
      setCustomAnalyse(data.custom_analyse || '')
      setCustomSmooth(data.custom_smooth || '')
      // Show advanced if custom params are set
      if (data.custom_super || data.custom_analyse || data.custom_smooth) {
        setShowAdvanced(true)
      }
    } catch (err) {
      console.error('Failed to load SVP config:', err)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    try {
      setSaving(true)
      await updateSVPConfig({
        enabled,
        target_fps: targetFps,
        preset,
        use_nvof: useNvof,
        shader,
        artifact_masking: artifactMasking,
        frame_interpolation: frameInterpolation,
        custom_super: customSuper || null,
        custom_analyse: customAnalyse || null,
        custom_smooth: customSmooth || null,
      })
      await loadConfig()
    } catch (err) {
      console.error('Failed to save SVP config:', err)
    } finally {
      setSaving(false)
    }
  }

  // Handle Escape key to close menu
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isOpen, onClose])

  if (!isOpen) return null

  if (loading) {
    return (
      <>
        <div className="svp-menu-backdrop" onClick={onClose} />
        <div className="svp-side-menu">
          <div className="svp-menu-header">
            <h2>SVP Settings</h2>
            <button className="svp-menu-close" onClick={onClose}>
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
              </svg>
            </button>
          </div>
          <div className="svp-menu-content" onClick={(e) => e.stopPropagation()}>
            <div className="loading">
              <div className="spinner" />
              <span>Loading SVP settings...</span>
            </div>
          </div>
        </div>
      </>
    )
  }

  const hasChanges =
    enabled !== (config?.enabled || false) ||
    targetFps !== (config?.target_fps || 60) ||
    preset !== (config?.preset || 'balanced') ||
    useNvof !== (config?.use_nvof !== false) ||
    shader !== (config?.shader || 23) ||
    artifactMasking !== (config?.artifact_masking ?? 100) ||
    frameInterpolation !== (config?.frame_interpolation ?? 2) ||
    (customSuper || '') !== (config?.custom_super || '') ||
    (customAnalyse || '') !== (config?.custom_analyse || '') ||
    (customSmooth || '') !== (config?.custom_smooth || '')

  const status = config?.status || {}
  const presets = config?.presets || {}
  const isReady = status.ready
  const hasVapourSynth = status.vapoursynth_available
  const hasSVPPlugins = status.svp_plugins_available
  const hasVspipe = status.vspipe_available
  const hasSourceFilter = status.source_filter_available
  const hasNvenc = status.nvenc_available
  const hasNvof = status.nvof_ready  // NVIDIA Optical Flow (hardware)
  const missing = status.missing || []

  // Source filter details
  const hasBestsource = status.bestsource_available
  const hasFfms2 = status.ffms2_available
  const hasLsmas = status.lsmas_available
  const sourceFilterName = hasBestsource ? 'bestsource' : hasFfms2 ? 'ffms2' : hasLsmas ? 'lsmas' : 'none'

  return (
    <>
      <div className="svp-menu-backdrop" onClick={onClose} />
      <div className="svp-side-menu">
        <div className="svp-menu-header">
          <h2>SVP Settings</h2>
          <button className="svp-menu-close" onClick={onClose}>
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
            </svg>
          </button>
        </div>

        <div className="svp-menu-content" onClick={(e) => e.stopPropagation()}>
          {/* Status badges */}
          <div className="backend-status">
            <strong>Components:</strong>
            <span className={`backend-badge ${hasVapourSynth ? 'available' : 'unavailable'}`}>
              VS: {hasVapourSynth ? '✓' : '✗'}
            </span>
            <span className={`backend-badge ${hasSVPPlugins ? 'available' : 'unavailable'}`}>
              SVP: {hasSVPPlugins ? '✓' : '✗'}
            </span>
            <span className={`backend-badge ${hasVspipe ? 'available' : 'unavailable'}`}>
              VP: {hasVspipe ? '✓' : '✗'}
            </span>
            <span className={`backend-badge ${hasSourceFilter ? 'available' : 'unavailable'}`}>
              SRC: {hasSourceFilter ? sourceFilterName : '✗'}
            </span>
            <span className={`backend-badge ${hasNvof ? 'available' : 'unavailable'}`} title="NVIDIA Optical Flow">
              OF: {hasNvof ? '✓' : '✗'}
            </span>
            <span className={`backend-badge ${hasNvenc ? 'available' : 'unavailable'}`}>
              ENC: {hasNvenc ? '✓' : '✗'}
            </span>
          </div>

          {isReady && (
            <div className="optical-flow-status info">
              <span className="status-icon">✓</span>
              <span>
                SVP ready
                {hasNvof && useNvof ? ' (NVOF)' : ''}
                {hasNvenc ? ' (NVENC)' : ''}
              </span>
            </div>
          )}

          {!isReady && (
            <div className="optical-flow-status warning">
              <span className="status-icon">!</span>
              <div>
                <span>Missing: {missing.join(', ')}</span>
              </div>
            </div>
          )}

          {/* Enable/Disable */}
          <section className="settings-section">
            <h3>Enable</h3>
            <div className="setting-row">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => setEnabled(e.target.checked)}
                  disabled={!isReady}
                />
                <span>Use SVP for videos</span>
              </label>
            </div>
          </section>

          {/* Target FPS */}
          <section className="settings-section">
            <h3>Target FPS</h3>
            <div className="setting-row fps-row">
              <input
                type="range"
                min="24"
                max="144"
                step="6"
                value={targetFps}
                onChange={(e) => setTargetFps(parseInt(e.target.value))}
                className="fps-slider"
              />
              <span className="fps-value">{targetFps}</span>
            </div>
          </section>

          {/* Quality Preset */}
          <section className="settings-section">
            <h3>Preset</h3>
            <div className="setting-row">
              <select
                value={preset}
                onChange={(e) => setPreset(e.target.value)}
                className="quality-select"
              >
                {Object.entries(presets).map(([key, info]) => (
                  <option key={key} value={key}>
                    {info.name}
                  </option>
                ))}
              </select>
            </div>
          </section>

          {/* Key SVP Settings */}
          <section className="settings-section">
            <h3>Settings</h3>

            {/* Frames interpolation mode */}
            <div className="setting-row">
              <label>Interpolation:</label>
              <select
                value={frameInterpolation}
                onChange={(e) => setFrameInterpolation(parseInt(e.target.value))}
                className="quality-select"
              >
                <option value={1}>Uniform</option>
                <option value={2}>Adaptive</option>
              </select>
            </div>

            {/* SVP Shader */}
            <div className="setting-row" style={{marginTop: '8px'}}>
              <label>Shader:</label>
              <select
                value={shader}
                onChange={(e) => setShader(parseInt(e.target.value))}
                className="quality-select"
              >
                <option value={1}>1. Fastest</option>
                <option value={2}>2. Sharp</option>
                <option value={11}>11. Simple Lite</option>
                <option value={13}>13. Standard</option>
                <option value={21}>21. Simple</option>
                <option value={23}>23. Complicated</option>
              </select>
            </div>

            {/* Artifact Masking */}
            <div className="setting-row" style={{marginTop: '8px'}}>
              <label>Artifacts:</label>
              <select
                value={artifactMasking}
                onChange={(e) => setArtifactMasking(parseInt(e.target.value))}
                className="quality-select"
              >
                <option value={0}>Disabled</option>
                <option value={50}>Weak</option>
                <option value={100}>Average</option>
                <option value={150}>Strong</option>
                <option value={200}>Strongest</option>
              </select>
            </div>

            {/* NVIDIA Optical Flow */}
            <div className="setting-row" style={{marginTop: '8px'}}>
              <label>NVIDIA OF:</label>
              <select
                value={useNvof ? 'use' : 'dont'}
                onChange={(e) => setUseNvof(e.target.value === 'use')}
                className="quality-select"
                disabled={!hasNvof}
              >
                <option value="use">Use</option>
                <option value="dont">Don't use</option>
              </select>
            </div>
          </section>

          {/* Advanced Settings Toggle */}
          <section className="settings-section">
            <h3>
              <button
                className="toggle-advanced"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                {showAdvanced ? '▼' : '▶'} Advanced
              </button>
            </h3>

            {showAdvanced && (
              <div className="advanced-settings">
                <p className="setting-note" style={{fontSize: '0.8em'}}>
                  Custom SVPflow parameters (JSON)
                </p>

                <div className="setting-row">
                  <label style={{fontSize: '0.85em'}}>Super</label>
                  <textarea
                    value={customSuper}
                    onChange={(e) => setCustomSuper(e.target.value)}
                    placeholder='{gpu:1,pel:2}'
                    rows={2}
                  />
                </div>

                <div className="setting-row">
                  <label style={{fontSize: '0.85em'}}>Analyse</label>
                  <textarea
                    value={customAnalyse}
                    onChange={(e) => setCustomAnalyse(e.target.value)}
                    placeholder='{gpu:1,block:{w:16}}'
                    rows={2}
                  />
                </div>

                <div className="setting-row">
                  <label style={{fontSize: '0.85em'}}>Smooth</label>
                  <textarea
                    value={customSmooth}
                    onChange={(e) => setCustomSmooth(e.target.value)}
                    placeholder='{gpuid:0,algo:23}'
                    rows={2}
                  />
                </div>
              </div>
            )}
          </section>
        </div>

        {/* Save Button */}
        <div className="svp-menu-footer">
          <button
            className="save-btn"
            onClick={handleSave}
            disabled={saving || !hasChanges}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </>
  )
}
