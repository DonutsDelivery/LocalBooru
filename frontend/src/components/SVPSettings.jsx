import { useState, useEffect } from 'react'
import {
  getSVPConfig,
  updateSVPConfig
} from '../api'
import './OpticalFlowSettings.css'  // Reuse the same styles

export default function SVPSettings() {
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
    loadConfig()
  }, [])

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

  if (loading) {
    return (
      <div className="optical-flow-settings loading">
        <div className="spinner" />
        <span>Loading SVP settings...</span>
      </div>
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
  const missing = status.missing || []

  // Source filter details
  const hasBestsource = status.bestsource_available
  const hasFfms2 = status.ffms2_available
  const hasLsmas = status.lsmas_available
  const sourceFilterName = hasBestsource ? 'bestsource' : hasFfms2 ? 'ffms2' : hasLsmas ? 'lsmas' : 'none'

  return (
    <div className="optical-flow-settings svp-settings">
      <h2>SVP Interpolation (Recommended)</h2>
      <p className="settings-description">
        High-quality frame interpolation using SVP (SmoothVideo Project).
        Best results with motion-compensated interpolation. Requires more processing power.
      </p>

      {/* Status badges */}
      <div className="backend-status">
        <strong>Components:</strong>
        <span className={`backend-badge ${hasVapourSynth ? 'available' : 'unavailable'}`}>
          VapourSynth: {hasVapourSynth ? '✓' : '✗'}
        </span>
        <span className={`backend-badge ${hasSVPPlugins ? 'available' : 'unavailable'}`}>
          SVPflow: {hasSVPPlugins ? '✓' : '✗'}
        </span>
        <span className={`backend-badge ${hasVspipe ? 'available' : 'unavailable'}`}>
          vspipe: {hasVspipe ? '✓' : '✗'}
        </span>
        <span className={`backend-badge ${hasSourceFilter ? 'available' : 'unavailable'}`}>
          Source: {hasSourceFilter ? sourceFilterName : '✗'}
        </span>
        <span className={`backend-badge ${hasNvenc ? 'available' : 'unavailable'}`}>
          NVENC: {hasNvenc ? '✓' : '✗'}
        </span>
      </div>

      {isReady && (
        <div className="optical-flow-status info">
          <span className="status-icon">✓</span>
          <span>SVP is ready! {hasNvenc ? 'Using NVENC for hardware encoding.' : 'Using software encoding.'}</span>
        </div>
      )}

      {!isReady && (
        <div className="optical-flow-status warning">
          <span className="status-icon">!</span>
          <div>
            <span>SVP is not ready. Missing: {missing.join(', ')}</span>
            {!hasSourceFilter && (
              <div style={{marginTop: '8px', fontSize: '0.85em'}}>
                <strong>Install source filter:</strong> <code>pacman -S vapoursynth-plugin-bestsource</code>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Enable/Disable */}
      <section className="settings-section">
        <h3>Enable SVP Interpolation</h3>
        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              disabled={!isReady}
            />
            <span>Automatically use SVP for video playback</span>
          </label>
        </div>
        <p className="setting-note">
          When enabled, videos will automatically play with SVP frame interpolation.
          SVP provides higher quality results using motion-compensated interpolation.
        </p>
      </section>

      {/* Target FPS */}
      <section className="settings-section">
        <h3>Target Frame Rate</h3>
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
          <span className="fps-value">{targetFps} fps</span>
        </div>
        <p className="setting-note">
          Common targets: 60 fps (smooth), 120 fps (very smooth), 144 fps (gaming monitors).
          Higher values require more processing power.
        </p>
      </section>

      {/* Quality Preset */}
      <section className="settings-section">
        <h3>Quality Preset</h3>
        <div className="setting-row">
          <select
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
            className="quality-select"
          >
            {Object.entries(presets).map(([key, info]) => (
              <option key={key} value={key}>
                {info.name} - {info.description}
              </option>
            ))}
          </select>
        </div>
        <p className="setting-note">
          <strong>Fast:</strong> Real-time capable, lower quality<br />
          <strong>Balanced:</strong> Good tradeoff (recommended)<br />
          <strong>Quality/Max:</strong> Best results, may buffer<br />
          <strong>Animation:</strong> Optimized for anime<br />
          <strong>Film:</strong> Natural motion for movies
        </p>
      </section>

      {/* Key SVP Settings */}
      <section className="settings-section">
        <h3>SVP Settings</h3>

        {/* Frames interpolation mode */}
        <div className="setting-row">
          <label style={{minWidth: '180px'}}>Frames interpolation mode:</label>
          <select
            value={frameInterpolation}
            onChange={(e) => setFrameInterpolation(parseInt(e.target.value))}
            className="quality-select"
          >
            <option value={1}>Uniform (max fluidity)</option>
            <option value={2}>Adaptive</option>
          </select>
        </div>

        {/* SVP Shader */}
        <div className="setting-row" style={{marginTop: '12px'}}>
          <label style={{minWidth: '180px'}}>SVP Shader:</label>
          <select
            value={shader}
            onChange={(e) => setShader(parseInt(e.target.value))}
            className="quality-select"
          >
            <option value={1}>1. Fastest (slow PCs)</option>
            <option value={2}>2. Sharp (anime)</option>
            <option value={11}>11. Simple Lite</option>
            <option value={13}>13. Standard</option>
            <option value={21}>21. Simple</option>
            <option value={23}>23. Complicated</option>
          </select>
        </div>

        {/* Artifact Masking */}
        <div className="setting-row" style={{marginTop: '12px'}}>
          <label style={{minWidth: '180px'}}>Artifacts Masking:</label>
          <select
            value={artifactMasking}
            onChange={(e) => setArtifactMasking(parseInt(e.target.value))}
            className="quality-select"
          >
            <option value={0}>Disabled</option>
            <option value={50}>Weakest</option>
            <option value={75}>Weak</option>
            <option value={100}>Average</option>
            <option value={150}>Strong</option>
            <option value={200}>Strongest</option>
          </select>
        </div>

        {/* NVIDIA Optical Flow */}
        <div className="setting-row" style={{marginTop: '12px'}}>
          <label style={{minWidth: '180px'}}>Use NVIDIA Optical Flow:</label>
          <select
            value={useNvof ? 'use' : 'dont'}
            onChange={(e) => setUseNvof(e.target.value === 'use')}
            className="quality-select"
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
            {showAdvanced ? '▼' : '▶'} Advanced Settings
          </button>
        </h3>

        {showAdvanced && (
          <div className="advanced-settings">
            <p className="setting-note">
              Override preset with custom SVPflow parameters (JSON format).
              Leave empty to use preset defaults.
              <a
                href="https://www.svp-team.com/wiki/Manual:SVPflow"
                target="_blank"
                rel="noopener noreferrer"
              >
                {' '}SVPflow documentation
              </a>
            </p>

            <div className="setting-row">
              <label>Super Parameters (svp1.Super)</label>
              <textarea
                value={customSuper}
                onChange={(e) => setCustomSuper(e.target.value)}
                placeholder='{gpu:1,pel:2,scale:{up:0,down:2}}'
                rows={2}
              />
              <small>Motion estimation resolution. pel=1,2,4 (accuracy), scale (hierarchical)</small>
            </div>

            <div className="setting-row">
              <label>Analyse Parameters (svp1.Analyse)</label>
              <textarea
                value={customAnalyse}
                onChange={(e) => setCustomAnalyse(e.target.value)}
                placeholder='{gpu:1,block:{w:16,h:16,overlap:2},...}'
                rows={3}
              />
              <small>Motion vector analysis. block (size), search (algorithm), refine (quality)</small>
            </div>

            <div className="setting-row">
              <label>Smooth Parameters (svp2.SmoothFps)</label>
              <textarea
                value={customSmooth}
                onChange={(e) => setCustomSmooth(e.target.value)}
                placeholder='{gpuid:0,algo:23,mask:{area:100},scene:{}}'
                rows={2}
              />
              <small>Frame rendering. algo=13,23 (method), mask (artifact reduction), scene (detection)</small>
            </div>
          </div>
        )}
      </section>

      {/* How SVP Works */}
      <section className="settings-section info-section">
        <h3>About SVP</h3>
        <p className="setup-note">
          <strong>SVP (SmoothVideo Project)</strong> uses advanced motion estimation
          to create smooth intermediate frames. Unlike simple blending, SVP tracks
          object motion and warps frames along motion vectors.
        </p>
        <ul className="feature-list">
          <li><strong>Motion Estimation:</strong> SVPflow analyzes motion between frames using hierarchical block matching</li>
          <li><strong>Motion Compensation:</strong> Frames are warped along motion vectors for natural movement</li>
          <li><strong>Artifact Masking:</strong> Problem areas (occlusions, fast motion) are detected and handled specially</li>
          <li><strong>GPU Acceleration:</strong> Both motion estimation and rendering use GPU compute</li>
        </ul>
      </section>

      {/* Save Button */}
      <div className="settings-actions">
        <button
          className="save-btn"
          onClick={handleSave}
          disabled={saving || !hasChanges}
        >
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  )
}
