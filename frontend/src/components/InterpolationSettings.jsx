import { useState, useEffect } from 'react'
import './OpticalFlowSettings.css'  // Reuse existing styles

// Check if running in Tauri environment
const IS_TAURI = typeof window !== 'undefined' && window.__TAURI__ !== undefined

// Tauri invoke wrapper
async function tauriInvoke(cmd, args = {}) {
  if (!IS_TAURI) {
    throw new Error('Not running in Tauri environment')
  }
  const { invoke } = await import('@tauri-apps/api/core')
  return invoke(cmd, args)
}

export default function InterpolationSettings() {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  // Backend detection
  const [backends, setBackends] = useState([])
  const [presets, setPresets] = useState([])

  // Configuration state
  const [enabled, setEnabled] = useState(false)
  const [backend, setBackend] = useState('none')
  const [preset, setPreset] = useState('balanced')
  const [targetFps, setTargetFps] = useState(60)
  const [useNvof, setUseNvof] = useState(true)
  const [svpAlgorithm, setSvpAlgorithm] = useState(23)
  const [artifactMasking, setArtifactMasking] = useState(100)

  // Original config for change detection
  const [originalConfig, setOriginalConfig] = useState(null)

  useEffect(() => {
    loadData()
  }, [])

  async function loadData() {
    if (!IS_TAURI) {
      setError('Interpolation settings are only available in the Tauri desktop app')
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      setError(null)

      // Load backends and presets in parallel
      const [backendsData, presetsData, configData] = await Promise.all([
        tauriInvoke('interpolation_detect_backends'),
        tauriInvoke('interpolation_get_presets'),
        tauriInvoke('interpolation_get_config'),
      ])

      setBackends(backendsData)
      setPresets(presetsData)

      // Set form state from config
      setEnabled(configData.enabled)
      setBackend(configData.backend)
      setPreset(configData.preset)
      setTargetFps(configData.target_fps)
      setUseNvof(configData.use_nvof)
      setSvpAlgorithm(configData.svp_algorithm)
      setArtifactMasking(configData.artifact_masking)

      setOriginalConfig(configData)
    } catch (err) {
      console.error('Failed to load interpolation settings:', err)
      setError(err.toString())
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    if (!IS_TAURI) return

    try {
      setSaving(true)
      setError(null)

      await tauriInvoke('interpolation_set_config', {
        config: {
          enabled,
          backend,
          preset,
          target_fps: targetFps,
          use_nvof: useNvof,
          gpu_id: 0,
          svp_algorithm: svpAlgorithm,
          artifact_masking: artifactMasking,
          scene_sensitivity: 50,
        }
      })

      // Reload to confirm
      await loadData()
    } catch (err) {
      console.error('Failed to save interpolation settings:', err)
      setError(err.toString())
    } finally {
      setSaving(false)
    }
  }

  function handleBackendChange(newBackend) {
    setBackend(newBackend)
    setEnabled(newBackend !== 'none')
  }

  if (!IS_TAURI) {
    return (
      <div className="optical-flow-settings">
        <h2>Frame Interpolation (Tauri Only)</h2>
        <div className="optical-flow-status warning">
          <span className="status-icon">!</span>
          <span>
            Frame interpolation settings are only available in the Tauri desktop application.
            This feature uses native video processing pipelines that require desktop integration.
          </span>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="optical-flow-settings loading">
        <div className="spinner" />
        <span>Loading interpolation settings...</span>
      </div>
    )
  }

  const hasChanges = originalConfig && (
    enabled !== originalConfig.enabled ||
    backend !== originalConfig.backend ||
    preset !== originalConfig.preset ||
    targetFps !== originalConfig.target_fps ||
    useNvof !== originalConfig.use_nvof ||
    svpAlgorithm !== originalConfig.svp_algorithm ||
    artifactMasking !== originalConfig.artifact_masking
  )

  // Find available backends
  const svpBackend = backends.find(b => b.backend === 'svp')
  const rifeBackend = backends.find(b => b.backend === 'rife_ncnn')
  const minterpolateBackend = backends.find(b => b.backend === 'minterpolate')

  const availableBackends = backends.filter(b => b.available)
  const hasAnyBackend = availableBackends.length > 0

  // Get current backend info
  const currentBackend = backends.find(b => b.backend === backend)
  const nvofAvailable = svpBackend?.nvof_available || false

  return (
    <div className="optical-flow-settings interpolation-settings">
      <h2>Frame Interpolation</h2>
      <p className="settings-description">
        Convert low frame rate videos (24/30fps) to smooth high frame rate (60/120fps) playback
        using motion-compensated interpolation.
      </p>

      {error && (
        <div className="optical-flow-status error">
          <span className="status-icon">!</span>
          <span>{error}</span>
        </div>
      )}

      {/* Backend Status */}
      <section className="settings-section">
        <h3>Available Backends</h3>
        <div className="backend-status">
          <span
            className={`backend-badge ${svpBackend?.available ? 'available' : 'unavailable'}`}
            title={svpBackend?.status}
          >
            SVP: {svpBackend?.available ? (svpBackend.nvof_available ? 'NVOF' : 'Ready') : 'Not Found'}
          </span>
          <span
            className={`backend-badge ${rifeBackend?.available ? 'available' : 'unavailable'}`}
            title={rifeBackend?.status}
          >
            RIFE: {rifeBackend?.available ? 'Ready' : 'Not Found'}
          </span>
          <span
            className={`backend-badge ${minterpolateBackend?.available ? 'available' : 'unavailable'}`}
            title={minterpolateBackend?.status}
          >
            FFmpeg: {minterpolateBackend?.available ? 'Ready' : 'Not Found'}
          </span>
        </div>

        {!hasAnyBackend && (
          <div className="optical-flow-status warning">
            <span className="status-icon">!</span>
            <div>
              <span>No interpolation backends available.</span>
              <div style={{marginTop: '8px', fontSize: '0.85em'}}>
                <strong>Install options:</strong>
                <ul style={{margin: '4px 0', paddingLeft: '20px'}}>
                  <li>SVP (recommended): <code>yay -S svp</code></li>
                  <li>FFmpeg minterpolate is usually available by default</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {hasAnyBackend && (
          <div className="optical-flow-status info">
            <span className="status-icon">i</span>
            <span>
              {availableBackends.length} backend{availableBackends.length > 1 ? 's' : ''} available.
              {svpBackend?.available && nvofAvailable && ' SVP with NVIDIA Optical Flow recommended for best quality.'}
              {svpBackend?.available && !nvofAvailable && ' SVP available for high-quality interpolation.'}
              {!svpBackend?.available && minterpolateBackend?.available && ' FFmpeg minterpolate available as fallback.'}
            </span>
          </div>
        )}
      </section>

      {/* Backend Selection */}
      <section className="settings-section">
        <h3>Interpolation Backend</h3>
        <div className="setting-row">
          <select
            value={backend}
            onChange={(e) => handleBackendChange(e.target.value)}
            className="quality-select"
          >
            <option value="none">Disabled</option>
            {svpBackend?.available && (
              <option value="svp">SVP (Best Quality) {nvofAvailable && '+ NVOF'}</option>
            )}
            {rifeBackend?.available && (
              <option value="rife_ncnn">RIFE Neural Network (Vulkan)</option>
            )}
            {minterpolateBackend?.available && (
              <option value="minterpolate">FFmpeg minterpolate (CPU)</option>
            )}
          </select>
        </div>
        <p className="setting-note">
          <strong>SVP:</strong> Professional motion interpolation with optical flow. Best quality.<br />
          <strong>RIFE:</strong> Neural network based. Good quality, GPU accelerated via Vulkan.<br />
          <strong>FFmpeg:</strong> Built-in minterpolate filter. Always available but CPU intensive.
        </p>
      </section>

      {/* Target FPS */}
      <section className="settings-section">
        <h3>Target Frame Rate</h3>
        <div className="setting-row fps-row">
          <input
            type="range"
            min="30"
            max="144"
            step="6"
            value={targetFps}
            onChange={(e) => setTargetFps(parseInt(e.target.value))}
            className="fps-slider"
            disabled={backend === 'none'}
          />
          <span className="fps-value">{targetFps} fps</span>
        </div>
        <div className="fps-presets">
          <button
            className={targetFps === 60 ? 'active' : ''}
            onClick={() => setTargetFps(60)}
            disabled={backend === 'none'}
          >
            60fps
          </button>
          <button
            className={targetFps === 120 ? 'active' : ''}
            onClick={() => setTargetFps(120)}
            disabled={backend === 'none'}
          >
            120fps
          </button>
          <button
            className={targetFps === 144 ? 'active' : ''}
            onClick={() => setTargetFps(144)}
            disabled={backend === 'none'}
          >
            144fps
          </button>
        </div>
        <p className="setting-note">
          Match your monitor's refresh rate for best results. Higher values require more processing power.
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
            disabled={backend === 'none'}
          >
            {presets.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
        {presets.find(p => p.id === preset) && (
          <p className="setting-note">
            {presets.find(p => p.id === preset).description}
          </p>
        )}
      </section>

      {/* SVP-specific settings */}
      {backend === 'svp' && (
        <section className="settings-section">
          <h3>SVP Settings</h3>

          {/* NVIDIA Optical Flow */}
          {nvofAvailable && (
            <div className="setting-row">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={useNvof}
                  onChange={(e) => setUseNvof(e.target.checked)}
                />
                <span>Use NVIDIA Optical Flow (NVOF)</span>
              </label>
            </div>
          )}
          {nvofAvailable && (
            <p className="setting-note" style={{marginTop: '4px'}}>
              NVOF uses dedicated hardware on RTX 20xx+ GPUs for faster processing and lower CPU usage.
            </p>
          )}

          {/* Algorithm */}
          <div className="setting-row" style={{marginTop: '12px'}}>
            <label style={{minWidth: '150px'}}>SVP Algorithm:</label>
            <select
              value={svpAlgorithm}
              onChange={(e) => setSvpAlgorithm(parseInt(e.target.value))}
              className="quality-select"
            >
              <option value={1}>1. Fastest (weak hardware)</option>
              <option value={2}>2. Sharp (anime)</option>
              <option value={11}>11. Simple Lite</option>
              <option value={13}>13. Standard</option>
              <option value={21}>21. Simple</option>
              <option value={23}>23. Complicated (best quality)</option>
            </select>
          </div>

          {/* Artifact Masking */}
          <div className="setting-row" style={{marginTop: '12px'}}>
            <label style={{minWidth: '150px'}}>Artifact Masking:</label>
            <select
              value={artifactMasking}
              onChange={(e) => setArtifactMasking(parseInt(e.target.value))}
              className="quality-select"
            >
              <option value={0}>Disabled (maximum smoothness)</option>
              <option value={50}>Weak</option>
              <option value={100}>Balanced</option>
              <option value={150}>Strong</option>
              <option value={200}>Maximum (fewest artifacts)</option>
            </select>
          </div>
          <p className="setting-note" style={{marginTop: '4px'}}>
            Higher masking reduces artifacts in problematic areas but may reduce smoothness.
          </p>
        </section>
      )}

      {/* Resolution Limits */}
      {currentBackend && currentBackend.max_resolution && (
        <section className="settings-section">
          <h3>Performance Notes</h3>
          <div className="optical-flow-status info">
            <span className="status-icon">i</span>
            <span>
              {currentBackend.backend === 'svp' && nvofAvailable && (
                <>Maximum recommended resolution for real-time: <strong>4K (3840x2160)</strong> with NVOF</>
              )}
              {currentBackend.backend === 'svp' && !nvofAvailable && (
                <>Maximum recommended resolution for real-time: <strong>1080p (1920x1080)</strong></>
              )}
              {currentBackend.backend === 'rife_ncnn' && (
                <>Maximum recommended resolution for real-time: <strong>1440p (2560x1440)</strong></>
              )}
              {currentBackend.backend === 'minterpolate' && (
                <>Maximum recommended resolution for real-time: <strong>720p (1280x720)</strong>. CPU intensive.</>
              )}
            </span>
          </div>
        </section>
      )}

      {/* How It Works */}
      <section className="settings-section info-section">
        <h3>How Frame Interpolation Works</h3>
        <p className="setup-note">
          Frame interpolation analyzes motion between video frames and generates new intermediate
          frames to create smoother playback. This is especially useful for movies (24fps) and
          console games (30fps) on high refresh rate monitors.
        </p>
        <ul className="feature-list">
          <li><strong>Motion Estimation:</strong> Analyzes how objects move between frames</li>
          <li><strong>Optical Flow:</strong> Creates motion vectors for each pixel region</li>
          <li><strong>Frame Synthesis:</strong> Generates new frames by warping along motion vectors</li>
          <li><strong>Artifact Handling:</strong> Detects and masks problem areas like occlusions</li>
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
        {hasChanges && (
          <button
            className="reset-btn"
            onClick={loadData}
            disabled={saving}
          >
            Reset
          </button>
        )}
      </div>
    </div>
  )
}
