import { useState, useEffect } from 'react'
import {
  getWhisperConfig,
  updateWhisperConfig
} from '../api'
import './OpticalFlowSettings.css'

export default function WhisperSubtitleSettings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Local form state
  const [enabled, setEnabled] = useState(false)
  const [autoGenerate, setAutoGenerate] = useState(false)
  const [modelSize, setModelSize] = useState('medium')
  const [language, setLanguage] = useState('ja')
  const [task, setTask] = useState('translate')
  const [chunkDuration, setChunkDuration] = useState(30)
  const [beamSize, setBeamSize] = useState(8)
  const [device, setDevice] = useState('auto')
  const [computeType, setComputeType] = useState('auto')
  const [vadFilter, setVadFilter] = useState(true)
  const [suppressNst, setSuppressNst] = useState(true)
  const [cacheSubtitles, setCacheSubtitles] = useState(true)
  const [subtitleFont, setSubtitleFont] = useState('Trebuchet MS')
  const [subtitleFontSize, setSubtitleFontSize] = useState(1.3)
  const [subtitleStyle, setSubtitleStyle] = useState('outline')
  const [subtitleColor, setSubtitleColor] = useState('#ffffff')
  const [subtitleOutlineColor, setSubtitleOutlineColor] = useState('#000000')
  const [subtitleBgOpacity, setSubtitleBgOpacity] = useState(0.75)

  useEffect(() => {
    loadConfig()
  }, [])

  async function loadConfig() {
    try {
      setLoading(true)
      const data = await getWhisperConfig()
      setConfig(data)
      setEnabled(data.enabled || false)
      setAutoGenerate(data.auto_generate || false)
      setModelSize(data.model_size || 'medium')
      setLanguage(data.language || 'ja')
      setTask(data.task || 'translate')
      setChunkDuration(data.chunk_duration || 30)
      setBeamSize(data.beam_size || 8)
      setDevice(data.device || 'auto')
      setComputeType(data.compute_type || 'auto')
      setVadFilter(data.vad_filter !== false)
      setSuppressNst(data.suppress_nst !== false)
      setCacheSubtitles(data.cache_subtitles !== false)
      setSubtitleFont(data.subtitle_font || 'Trebuchet MS')
      setSubtitleFontSize(data.subtitle_font_size || 1.3)
      setSubtitleStyle(data.subtitle_style || 'outline')
      setSubtitleColor(data.subtitle_color || '#ffffff')
      setSubtitleOutlineColor(data.subtitle_outline_color || '#000000')
      setSubtitleBgOpacity(data.subtitle_bg_opacity ?? 0.75)
    } catch (err) {
      console.error('Failed to load whisper config:', err)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    try {
      setSaving(true)
      await updateWhisperConfig({
        enabled,
        auto_generate: autoGenerate,
        model_size: modelSize,
        language,
        task,
        chunk_duration: chunkDuration,
        beam_size: beamSize,
        device,
        compute_type: computeType,
        vad_filter: vadFilter,
        suppress_nst: suppressNst,
        cache_subtitles: cacheSubtitles,
        subtitle_font: subtitleFont,
        subtitle_font_size: subtitleFontSize,
        subtitle_style: subtitleStyle,
        subtitle_color: subtitleColor,
        subtitle_outline_color: subtitleOutlineColor,
        subtitle_bg_opacity: subtitleBgOpacity,
      })
      await loadConfig()
    } catch (err) {
      console.error('Failed to save whisper config:', err)
    } finally {
      setSaving(false)
    }
  }

  function handleReset() {
    if (!config) return
    setEnabled(config.enabled || false)
    setAutoGenerate(config.auto_generate || false)
    setModelSize(config.model_size || 'medium')
    setLanguage(config.language || 'ja')
    setTask(config.task || 'translate')
    setChunkDuration(config.chunk_duration || 30)
    setBeamSize(config.beam_size || 8)
    setDevice(config.device || 'auto')
    setComputeType(config.compute_type || 'auto')
    setVadFilter(config.vad_filter !== false)
    setSuppressNst(config.suppress_nst !== false)
    setCacheSubtitles(config.cache_subtitles !== false)
    setSubtitleFont(config.subtitle_font || 'Trebuchet MS')
    setSubtitleFontSize(config.subtitle_font_size || 1.3)
    setSubtitleStyle(config.subtitle_style || 'outline')
    setSubtitleColor(config.subtitle_color || '#ffffff')
    setSubtitleOutlineColor(config.subtitle_outline_color || '#000000')
    setSubtitleBgOpacity(config.subtitle_bg_opacity ?? 0.75)
  }

  if (loading) {
    return (
      <div className="optical-flow-settings loading">
        <div className="spinner" />
        <span>Loading subtitle settings...</span>
      </div>
    )
  }

  const status = config?.status || {}
  const hasChanges =
    enabled !== (config?.enabled || false) ||
    autoGenerate !== (config?.auto_generate || false) ||
    modelSize !== (config?.model_size || 'medium') ||
    language !== (config?.language || 'ja') ||
    task !== (config?.task || 'translate') ||
    chunkDuration !== (config?.chunk_duration || 30) ||
    beamSize !== (config?.beam_size || 8) ||
    device !== (config?.device || 'auto') ||
    computeType !== (config?.compute_type || 'auto') ||
    vadFilter !== (config?.vad_filter !== false) ||
    suppressNst !== (config?.suppress_nst !== false) ||
    cacheSubtitles !== (config?.cache_subtitles !== false) ||
    subtitleFont !== (config?.subtitle_font || 'Trebuchet MS') ||
    subtitleFontSize !== (config?.subtitle_font_size || 1.3) ||
    subtitleStyle !== (config?.subtitle_style || 'outline') ||
    subtitleColor !== (config?.subtitle_color || '#ffffff') ||
    subtitleOutlineColor !== (config?.subtitle_outline_color || '#000000') ||
    subtitleBgOpacity !== (config?.subtitle_bg_opacity ?? 0.75)

  return (
    <div className="optical-flow-settings">
      <h2>Whisper Subtitles</h2>
      <p className="settings-description">
        Real-time subtitle generation using faster-whisper. Extracts audio in chunks,
        transcribes with Whisper, and displays subtitles over video playback.
      </p>

      {/* Status badges */}
      <div className="backend-status">
        <strong>Status:</strong>
        <span className={`backend-badge ${status.faster_whisper_installed ? 'available' : 'unavailable'}`}>
          faster-whisper: {status.faster_whisper_installed ? 'Installed' : 'Not installed'}
        </span>
        <span className={`backend-badge ${status.cuda_available ? 'available' : 'unavailable'}`}>
          CUDA: {status.cuda_available ? 'Available' : 'CPU only'}
        </span>
      </div>

      {!status.faster_whisper_installed && (
        <div className="optical-flow-status warning">
          <span className="status-icon">!</span>
          <span>faster-whisper is not installed. Run <code>pip install faster-whisper</code> to enable subtitle generation.</span>
        </div>
      )}

      {/* Install status */}
      {!status.faster_whisper_installed && (
        <section className="settings-section">
          <p className="setting-note">
            The CC button is always available in the video player. Clicking it will
            automatically install faster-whisper if needed.
            Press C or click the CC button during video playback to generate subtitles.
          </p>
        </section>
      )}

      {/* Auto-generate */}
      <section className="settings-section">
        <h3>Auto-Generate</h3>
        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={autoGenerate}
              onChange={(e) => setAutoGenerate(e.target.checked)}
            />
            <span>Automatically generate subtitles when opening videos</span>
          </label>
        </div>
        <p className="setting-note">
          When enabled, subtitles will start generating as soon as you open a video.
          Cached subtitles load instantly.
        </p>
      </section>

      {/* Model size */}
      <section className="settings-section">
        <h3>Model Size</h3>
        <div className="setting-row">
          <select
            value={modelSize}
            onChange={(e) => setModelSize(e.target.value)}
            className="quality-select"
          >
            <option value="tiny">Tiny (~75MB, fastest, lowest quality)</option>
            <option value="base">Base (~145MB, fast)</option>
            <option value="small">Small (~483MB, good balance)</option>
            <option value="medium">Medium (~1.5GB, recommended)</option>
            <option value="large-v2">Large V2 (~3GB, best quality)</option>
            <option value="large-v3">Large V3 (~3GB, latest)</option>
          </select>
        </div>
        <p className="setting-note">
          Larger models produce better results but use more memory and are slower.
          Medium is recommended for Japanese audio translation.
        </p>
      </section>

      {/* Language and Task */}
      <section className="settings-section">
        <h3>Language & Task</h3>
        <div className="setting-row">
          <label>
            <span style={{ marginRight: 8, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Source language:</span>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="quality-select"
              style={{ minWidth: 180 }}
            >
              <option value="ja">Japanese</option>
              <option value="en">English</option>
              <option value="zh">Chinese</option>
              <option value="ko">Korean</option>
              <option value="de">German</option>
              <option value="fr">French</option>
              <option value="es">Spanish</option>
              <option value="ru">Russian</option>
              <option value="">Auto-detect</option>
            </select>
          </label>
        </div>
        <div className="setting-row" style={{ marginTop: 8 }}>
          <label>
            <span style={{ marginRight: 8, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Task:</span>
            <select
              value={task}
              onChange={(e) => setTask(e.target.value)}
              className="quality-select"
              style={{ minWidth: 180 }}
            >
              <option value="translate">Translate to English</option>
              <option value="transcribe">Transcribe (keep original language)</option>
            </select>
          </label>
        </div>
      </section>

      {/* Subtitle Appearance */}
      <section className="settings-section">
        <h3>Subtitle Appearance</h3>
        <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 8 }}>
          <label>
            <span style={{ marginRight: 8, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Style:</span>
            <select
              value={subtitleStyle}
              onChange={(e) => setSubtitleStyle(e.target.value)}
              className="quality-select"
              style={{ minWidth: 200 }}
            >
              <option value="outline">Outline (Crunchyroll-style)</option>
              <option value="background">Background box</option>
              <option value="outline_background">Outline + Background</option>
            </select>
          </label>
        </div>
        <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 8, marginTop: 12 }}>
          <label>
            <span style={{ marginRight: 8, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Font:</span>
            <select
              value={subtitleFont}
              onChange={(e) => setSubtitleFont(e.target.value)}
              className="quality-select"
              style={{ minWidth: 200 }}
            >
              <option value="Trebuchet MS">Trebuchet MS (Crunchyroll)</option>
              <option value="Arial">Arial</option>
              <option value="Helvetica">Helvetica</option>
              <option value="Verdana">Verdana</option>
              <option value="Georgia">Georgia</option>
              <option value="Times New Roman">Times New Roman</option>
              <option value="Courier New">Courier New</option>
              <option value="Comic Sans MS">Comic Sans MS</option>
            </select>
          </label>
        </div>
        <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 6, marginTop: 12 }}>
          <label style={{ fontWeight: 500 }}>Font Size</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <input
              type="range"
              min="0.6"
              max="3.0"
              step="0.1"
              value={subtitleFontSize}
              onChange={(e) => setSubtitleFontSize(parseFloat(e.target.value))}
              className="fps-slider"
              style={{ flex: 1 }}
            />
            <span className="fps-value" style={{ minWidth: 50 }}>{subtitleFontSize}rem</span>
          </div>
        </div>
        <div className="setting-row" style={{ display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Text color:</span>
            <input
              type="color"
              value={subtitleColor}
              onChange={(e) => setSubtitleColor(e.target.value)}
              style={{ width: 36, height: 28, cursor: 'pointer', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 4, background: 'transparent' }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Outline color:</span>
            <input
              type="color"
              value={subtitleOutlineColor}
              onChange={(e) => setSubtitleOutlineColor(e.target.value)}
              style={{ width: 36, height: 28, cursor: 'pointer', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 4, background: 'transparent' }}
            />
          </label>
        </div>
        {(subtitleStyle === 'background' || subtitleStyle === 'outline_background') && (
          <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 6, marginTop: 12 }}>
            <label style={{ fontWeight: 500 }}>Background Opacity</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={subtitleBgOpacity}
                onChange={(e) => setSubtitleBgOpacity(parseFloat(e.target.value))}
                className="fps-slider"
                style={{ flex: 1 }}
              />
              <span className="fps-value" style={{ minWidth: 40 }}>{Math.round(subtitleBgOpacity * 100)}%</span>
            </div>
          </div>
        )}
        <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(255,255,255,0.03)', borderRadius: 6, fontSize: '0.85rem' }}>
          <span style={{ fontWeight: 600 }}>Preview: </span>
          <span style={{
            fontFamily: `'${subtitleFont}', Arial, sans-serif`,
            fontSize: `${Math.min(subtitleFontSize, 1.6)}rem`,
            fontWeight: 700,
            color: subtitleColor,
            textShadow: (subtitleStyle === 'outline' || subtitleStyle === 'outline_background')
              ? `-1px -1px 0 ${subtitleOutlineColor}, 1px -1px 0 ${subtitleOutlineColor}, -1px 1px 0 ${subtitleOutlineColor}, 1px 1px 0 ${subtitleOutlineColor}, 0 0 6px ${subtitleOutlineColor}80`
              : 'none',
            background: (subtitleStyle === 'background' || subtitleStyle === 'outline_background')
              ? `rgba(0, 0, 0, ${subtitleBgOpacity})`
              : 'transparent',
            padding: '2px 6px',
            borderRadius: 3,
          }}>
            Sample subtitle text
          </span>
        </div>
      </section>

      {/* Cache */}
      <section className="settings-section">
        <h3>Cache Subtitles</h3>
        <div className="setting-row">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={cacheSubtitles}
              onChange={(e) => setCacheSubtitles(e.target.checked)}
            />
            <span>Save .vtt files alongside videos for instant reuse</span>
          </label>
        </div>
      </section>

      {/* Advanced settings */}
      <section className="settings-section svp-settings">
        <button
          className="toggle-advanced"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? '- ' : '+ '}Advanced Settings
        </button>
        {showAdvanced && (
          <div className="advanced-settings">
            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 6 }}>
              <label style={{ fontWeight: 500 }}>Chunk Duration</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="range"
                  min="10"
                  max="120"
                  step="5"
                  value={chunkDuration}
                  onChange={(e) => setChunkDuration(parseInt(e.target.value))}
                  className="fps-slider"
                  style={{ flex: 1 }}
                />
                <span className="fps-value" style={{ minWidth: 50 }}>{chunkDuration}s</span>
              </div>
              <small>How much audio to process at once. 30s matches mpv whisper-subs defaults.</small>
            </div>

            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 6, marginTop: 16 }}>
              <label style={{ fontWeight: 500 }}>Beam Size</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={beamSize}
                  onChange={(e) => setBeamSize(parseInt(e.target.value))}
                  className="fps-slider"
                  style={{ flex: 1 }}
                />
                <span className="fps-value" style={{ minWidth: 30 }}>{beamSize}</span>
              </div>
              <small>Higher values improve accuracy but slow down transcription.</small>
            </div>

            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 6, marginTop: 16 }}>
              <label style={{ fontWeight: 500 }}>Device</label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                className="quality-select"
                style={{ minWidth: 180 }}
              >
                <option value="auto">Auto (prefer CUDA)</option>
                <option value="cuda">CUDA (GPU)</option>
                <option value="cpu">CPU</option>
              </select>
            </div>

            <div className="setting-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 6, marginTop: 16 }}>
              <label style={{ fontWeight: 500 }}>Compute Type</label>
              <select
                value={computeType}
                onChange={(e) => setComputeType(e.target.value)}
                className="quality-select"
                style={{ minWidth: 180 }}
              >
                <option value="auto">Auto</option>
                <option value="float16">float16 (GPU, fast)</option>
                <option value="int8_float16">int8_float16 (GPU, balanced)</option>
                <option value="int8">int8 (CPU, fast)</option>
              </select>
            </div>

            <div className="setting-row" style={{ marginTop: 16 }}>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={vadFilter}
                  onChange={(e) => setVadFilter(e.target.checked)}
                />
                <span>VAD filter (skip silence)</span>
              </label>
            </div>

            <div className="setting-row">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={suppressNst}
                  onChange={(e) => setSuppressNst(e.target.checked)}
                />
                <span>Suppress non-speech tokens</span>
              </label>
            </div>
          </div>
        )}
      </section>

      {/* Save/Reset */}
      <div className="settings-actions">
        <button
          className="reset-btn"
          onClick={handleReset}
          disabled={saving || !hasChanges}
        >
          Reset
        </button>
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
