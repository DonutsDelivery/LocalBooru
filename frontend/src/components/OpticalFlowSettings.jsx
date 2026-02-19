import { useState, useEffect } from 'react'
import {
  getOpticalFlowConfig,
  updateOpticalFlowConfig
} from '../api'
import './OpticalFlowSettings.css'

export default function OpticalFlowSettings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  // Local form state
  const [enabled, setEnabled] = useState(false)
  const [targetFps, setTargetFps] = useState(60)
  const [quality, setQuality] = useState('fast')

  useEffect(() => {
    loadConfig()
  }, [])

  async function loadConfig() {
    try {
      setLoading(true)
      const data = await getOpticalFlowConfig()
      setConfig(data)
      setEnabled(data.enabled || false)
      setTargetFps(data.target_fps || 60)
      setQuality(data.quality || 'fast')
    } catch (err) {
      console.error('Failed to load optical flow config:', err)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    try {
      setSaving(true)
      await updateOpticalFlowConfig({
        enabled,
        target_fps: targetFps,
        quality
      })
      await loadConfig()
    } catch (err) {
      console.error('Failed to save optical flow config:', err)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="optical-flow-settings loading">
        <div className="spinner" />
        <span>Loading interpolation settings...</span>
      </div>
    )
  }

  const hasChanges =
    enabled !== (config?.enabled || false) ||
    targetFps !== (config?.target_fps || 60) ||
    quality !== (config?.quality || 'fast')

  return (
    <div className="optical-flow-settings">
      <h2>Frame Interpolation</h2>
      <p className="settings-description">
        Increase video frame rate using FFmpeg's motion-compensated interpolation.
        Converts 24/30fps videos to smooth 60fps+ playback.
      </p>

      {/* Backend status */}
      <div className="backend-status">
        <span className="backend-badge available">
          FFmpeg minterpolate: Built-in
        </span>
      </div>

      <div className="optical-flow-status info">
        <span className="status-icon">i</span>
        <span>
          Uses FFmpeg's minterpolate filter with motion-compensated frame blending.
          No additional software required.
        </span>
      </div>

      {/* Enable/Disable */}
      <section className="settings-section">
        <h3>Enable Interpolation</h3>
        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            <span>Auto-start interpolation when playing videos</span>
          </label>
        </div>
        <p className="setting-note">
          When enabled, videos will automatically play with frame interpolation applied.
          The video is re-encoded in real-time and streamed via HLS.
        </p>
      </section>

      {/* Target FPS */}
      <section className="settings-section">
        <h3>Target Frame Rate</h3>
        <div className="setting-row fps-row">
          <input
            type="range"
            min="30"
            max="120"
            step="5"
            value={targetFps}
            onChange={(e) => setTargetFps(parseInt(e.target.value))}
            className="fps-slider"
          />
          <span className="fps-value">{targetFps} fps</span>
        </div>
        <div className="fps-presets">
          <button
            className={targetFps === 48 ? 'active' : ''}
            onClick={() => setTargetFps(48)}
          >
            48fps
          </button>
          <button
            className={targetFps === 60 ? 'active' : ''}
            onClick={() => setTargetFps(60)}
          >
            60fps
          </button>
          <button
            className={targetFps === 120 ? 'active' : ''}
            onClick={() => setTargetFps(120)}
          >
            120fps
          </button>
        </div>
        <p className="setting-note">
          Match your monitor's refresh rate for best results. Higher values need more CPU/GPU.
          60fps is recommended for most systems.
        </p>
      </section>

      {/* Quality preset */}
      <section className="settings-section">
        <h3>Quality Preset</h3>
        <div className="setting-row">
          <select
            value={quality}
            onChange={(e) => setQuality(e.target.value)}
            className="quality-select"
          >
            <option value="fast">Fast (real-time, lower quality)</option>
            <option value="balanced">Balanced (good quality, may buffer)</option>
            <option value="quality">Quality (best quality, slowest)</option>
          </select>
        </div>
        <p className="setting-note">
          Fast is recommended for real-time playback. Higher quality settings
          may cause buffering on slower hardware or high-resolution videos.
        </p>
      </section>

      {/* How it works */}
      <section className="settings-section info-section">
        <h3>How It Works</h3>
        <p className="setup-note">
          Frame interpolation analyzes motion between video frames and generates
          new intermediate frames for smoother playback. This is especially useful
          for movies (24fps) and console recordings (30fps) on high refresh rate monitors.
        </p>
        <ul className="feature-list">
          <li><strong>Motion estimation:</strong> Analyzes how objects move between frames</li>
          <li><strong>Frame synthesis:</strong> Generates new frames along motion vectors</li>
          <li><strong>HLS streaming:</strong> Processed video streams in real-time via HLS</li>
          <li><strong>Hardware encoding:</strong> Uses NVENC when available for fast encoding</li>
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
