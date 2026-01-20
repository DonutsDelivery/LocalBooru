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
  const [useGpu, setUseGpu] = useState(true)
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
      setUseGpu(data.use_gpu !== false)
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
        use_gpu: useGpu,
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
    useGpu !== (config?.use_gpu !== false) ||
    quality !== (config?.quality || 'fast')

  const backend = config?.backend || {}
  const hasOpenCVCuda = backend.cv2_cuda_available
  const hasTorchCuda = backend.cuda_available
  const hasGpuBackend = backend.gpu_backend !== null
  const hasCpuBackend = backend.cv2_available
  const hasAnyBackend = backend.any_backend_available
  const gpuBackendName = backend.gpu_backend === 'opencv_cuda' ? 'OpenCV CUDA' :
                         backend.gpu_backend === 'torch_cuda' ? 'PyTorch CUDA' : null

  return (
    <div className="optical-flow-settings">
      <h2>Frame Interpolation</h2>
      <p className="settings-description">
        Smooth video playback with AI-generated intermediate frames.
        Uses optical flow to create fluid motion at higher frame rates.
      </p>

      {/* Backend status */}
      <div className="backend-status">
        <strong>Backends:</strong>
        <span className={`backend-badge ${hasOpenCVCuda ? 'available' : 'unavailable'}`}>
          OpenCV CUDA: {hasOpenCVCuda ? '✓' : '✗'}
        </span>
        <span className={`backend-badge ${hasTorchCuda ? 'available' : 'unavailable'}`}>
          PyTorch CUDA: {hasTorchCuda ? '✓' : '✗'}
        </span>
        <span className={`backend-badge ${hasCpuBackend ? 'available' : 'unavailable'}`}>
          CPU Fallback: {hasCpuBackend ? '✓' : '✗'}
        </span>
      </div>

      {gpuBackendName && (
        <div className="optical-flow-status info">
          <span className="status-icon">⚡</span>
          <span>Using <strong>{gpuBackendName}</strong> for GPU-accelerated interpolation</span>
        </div>
      )}

      {!hasGpuBackend && hasCpuBackend && (
        <div className="optical-flow-status warning">
          <span className="status-icon">!</span>
          <span>No GPU backend available. CPU interpolation will be slow. Install OpenCV with CUDA support for GPU acceleration.</span>
        </div>
      )}

      {!hasAnyBackend && (
        <div className="optical-flow-status warning">
          <span className="status-icon">!</span>
          <span>No interpolation backend available. Install OpenCV (<code>pip install opencv-python</code>).</span>
        </div>
      )}

      {/* Enable/Disable */}
      <section className="settings-section">
        <h3>Enable Interpolation Button</h3>
        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              disabled={!hasAnyBackend}
            />
            <span>Show interpolation button in video player</span>
          </label>
        </div>
        <p className="setting-note">
          When viewing videos, a wave button will appear to play with frame interpolation.
          The video streams in-app with smooth motion applied.
        </p>
      </section>

      {/* Target FPS */}
      <section className="settings-section">
        <h3>Target Frame Rate</h3>
        <div className="setting-row fps-row">
          <input
            type="range"
            min="15"
            max="120"
            step="5"
            value={targetFps}
            onChange={(e) => setTargetFps(parseInt(e.target.value))}
            className="fps-slider"
          />
          <span className="fps-value">{targetFps} fps</span>
        </div>
        <p className="setting-note">
          Higher frame rates are smoother but require more processing power.
          60 fps is recommended for most videos.
        </p>
      </section>

      {/* GPU toggle */}
      {hasGpuBackend && (
        <section className="settings-section">
          <h3>GPU Acceleration</h3>
          <div className="setting-row">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={useGpu}
                onChange={(e) => setUseGpu(e.target.checked)}
              />
              <span>Use GPU for interpolation (faster, better quality)</span>
            </label>
          </div>
          <p className="setting-note">
            GPU interpolation uses a neural network for higher quality results.
            Disable to use CPU-based optical flow (slower but works without CUDA).
          </p>
        </section>
      )}

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
            <option value="balanced">Balanced (good quality, slower)</option>
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
          <strong>Native interpolation:</strong> Unlike SVP which requires external software,
          this feature uses built-in optical flow to generate intermediate frames.
          When you click the interpolation button on a video, LocalBooru analyzes motion
          between frames and creates smooth transitions at your target frame rate.
        </p>
        <ul className="feature-list">
          <li><strong>GPU mode:</strong> Uses a lightweight neural network for high-quality interpolation</li>
          <li><strong>CPU mode:</strong> Uses OpenCV Farneback optical flow algorithm</li>
          <li><strong>Streaming:</strong> Video is processed in real-time and streamed via HLS</li>
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
