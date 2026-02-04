/**
 * Tauri API Compatibility Layer
 *
 * Provides a unified API that works in both Electron and Tauri environments.
 * This module exports a `tauriAPI` object that mirrors the Electron preload API.
 */

// Detect if we're running in Tauri
const isTauri = () => {
  return typeof window !== 'undefined' && window.__TAURI_INTERNALS__ !== undefined
}

// Lazy-load Tauri modules only when needed (and only in Tauri environment)
let tauriInvoke = null
let tauriEvent = null
let tauriDialog = null
let tauriShell = null
let tauriClipboard = null
let tauriWindow = null

async function loadTauriModules() {
  if (!isTauri()) return false

  try {
    const [core, event, dialog, shell, clipboard, windowMod] = await Promise.all([
      import('@tauri-apps/api/core'),
      import('@tauri-apps/api/event'),
      import('@tauri-apps/plugin-dialog'),
      import('@tauri-apps/plugin-shell'),
      import('@tauri-apps/plugin-clipboard-manager'),
      import('@tauri-apps/api/window')
    ])
    tauriInvoke = core.invoke
    tauriEvent = event
    tauriDialog = dialog
    tauriShell = shell
    tauriClipboard = clipboard
    tauriWindow = windowMod
    return true
  } catch (e) {
    console.warn('[TauriAPI] Failed to load Tauri modules:', e)
    return false
  }
}

// Initialize Tauri modules if in Tauri environment
let tauriReady = isTauri() ? loadTauriModules() : Promise.resolve(false)

/**
 * Tauri API that mirrors the Electron API interface
 */
const tauriAPI = {
  // Platform info
  platform: isTauri() ? 'tauri' : null,
  isTauri: isTauri(),
  isElectron: false,

  // Get API URL for backend communication
  getApiUrl: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      const status = await tauriInvoke('backend_status')
      return `http://127.0.0.1:${status.port}`
    } catch (e) {
      console.error('[TauriAPI] getApiUrl failed:', e)
      return 'http://127.0.0.1:8789'
    }
  },

  // Open native folder picker dialog
  addDirectory: async () => {
    await tauriReady
    if (!tauriDialog) return null
    try {
      const result = await tauriDialog.open({
        directory: true,
        multiple: false,
        title: 'Select folder to watch'
      })
      return result || null
    } catch (e) {
      console.error('[TauriAPI] addDirectory failed:', e)
      return null
    }
  },

  // Get backend server status
  getBackendStatus: async () => {
    await tauriReady
    if (!tauriInvoke) return { running: false, port: 8789 }
    try {
      const status = await tauriInvoke('backend_status')
      return {
        running: status.running,
        port: status.port
      }
    } catch (e) {
      console.error('[TauriAPI] getBackendStatus failed:', e)
      return { running: false, port: 8789 }
    }
  },

  // Restart backend server
  restartBackend: async () => {
    await tauriReady
    if (!tauriInvoke) return { success: false }
    try {
      await tauriInvoke('backend_restart')
      return { success: true }
    } catch (e) {
      console.error('[TauriAPI] restartBackend failed:', e)
      return { success: false, error: e.message }
    }
  },

  // Open URL in system browser
  openExternal: async (url) => {
    await tauriReady
    if (!tauriShell) return
    try {
      await tauriShell.open(url)
    } catch (e) {
      console.error('[TauriAPI] openExternal failed:', e)
      // Fallback to window.open
      window.open(url, '_blank')
    }
  },

  // Show file in native file explorer
  showInFolder: async (filePath) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('show_in_folder', { path: filePath })
    } catch (e) {
      console.error('[TauriAPI] showInFolder failed:', e)
    }
  },

  // Auto-updater (not implemented yet for Tauri)
  checkForUpdate: async () => {
    // TODO: Implement Tauri updater
    return null
  },
  downloadUpdate: async () => {
    // TODO: Implement Tauri updater
    return null
  },
  installUpdate: async () => {
    // TODO: Implement Tauri updater
    return null
  },
  getVersion: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('get_app_version')
    } catch (e) {
      console.error('[TauriAPI] getVersion failed:', e)
      return null
    }
  },
  isPortable: async () => {
    await tauriReady
    if (!tauriInvoke) return false
    try {
      const status = await tauriInvoke('backend_status')
      return status.mode === 'portable'
    } catch (e) {
      return false
    }
  },
  onUpdaterStatus: (callback) => {
    // TODO: Implement Tauri updater events
    return () => {}
  },

  // Window controls for custom title bar
  minimizeWindow: async () => {
    await tauriReady
    if (!tauriWindow) return
    try {
      const currentWindow = tauriWindow.getCurrentWindow()
      await currentWindow.minimize()
    } catch (e) {
      console.error('[TauriAPI] minimizeWindow failed:', e)
    }
  },

  maximizeWindow: async () => {
    await tauriReady
    if (!tauriWindow) return false
    try {
      const currentWindow = tauriWindow.getCurrentWindow()
      const isMaximized = await currentWindow.isMaximized()
      if (isMaximized) {
        await currentWindow.unmaximize()
      } else {
        await currentWindow.maximize()
      }
      return await currentWindow.isMaximized()
    } catch (e) {
      console.error('[TauriAPI] maximizeWindow failed:', e)
      return false
    }
  },

  closeWindow: async () => {
    await tauriReady
    if (!tauriWindow) return
    try {
      const currentWindow = tauriWindow.getCurrentWindow()
      await currentWindow.close()
    } catch (e) {
      console.error('[TauriAPI] closeWindow failed:', e)
    }
  },

  isMaximized: async () => {
    await tauriReady
    if (!tauriWindow) return false
    try {
      const currentWindow = tauriWindow.getCurrentWindow()
      return await currentWindow.isMaximized()
    } catch (e) {
      console.error('[TauriAPI] isMaximized failed:', e)
      return false
    }
  },

  // App lifecycle
  quitApp: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('quit_app')
    } catch (e) {
      console.error('[TauriAPI] quitApp failed:', e)
    }
  },

  // Clipboard
  copyImageToClipboard: async (imageUrl) => {
    await tauriReady
    if (!tauriInvoke) return { success: false, error: 'Tauri not available' }
    try {
      return await tauriInvoke('copy_image_to_clipboard', { imageUrl })
    } catch (e) {
      console.error('[TauriAPI] copyImageToClipboard failed:', e)
      return { success: false, error: e.message || String(e) }
    }
  },

  // Context menu for images
  showImageContextMenu: async (options) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('show_image_context_menu', { options })
    } catch (e) {
      console.error('[TauriAPI] showImageContextMenu failed:', e)
    }
  }
}

/**
 * Unified API that works in both Electron and Tauri
 * Falls back to Electron API if available, otherwise uses Tauri API
 */
export function getDesktopAPI() {
  // Check for Electron first
  if (typeof window !== 'undefined' && window.electronAPI) {
    return window.electronAPI
  }

  // Check for Tauri
  if (isTauri()) {
    return tauriAPI
  }

  // No desktop API available (running in browser)
  return null
}

/**
 * Check if running in a desktop environment (Electron or Tauri)
 */
export function isDesktopApp() {
  return (typeof window !== 'undefined' && window.electronAPI) || isTauri()
}

/**
 * Check if running in Tauri specifically
 */
export function isTauriApp() {
  return isTauri()
}

/**
 * Check if running in Electron specifically
 */
export function isElectronApp() {
  return typeof window !== 'undefined' && window.electronAPI?.isElectron
}

// =============================================================================
// Video Control API (GStreamer-based)
// =============================================================================

/**
 * Video control commands for GStreamer-based playback in Tauri
 * These provide native video control as an alternative to HTML5 video
 */
const videoControlAPI = {
  // -------------------------------------------------------------------------
  // VFR (Variable Frame Rate) Video Player Commands
  // -------------------------------------------------------------------------

  /**
   * Analyze a video file for VFR characteristics
   * @param {string} uri - Video file path or URI
   * @returns {Promise<{is_vfr: boolean, average_fps: number, container_fps: number|null}>}
   */
  analyzeVfr: async (uri) => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('video_vfr_analyze', { uri })
    } catch (e) {
      console.error('[VideoAPI] analyzeVfr failed:', e)
      throw e
    }
  },

  /**
   * Initialize the VFR-aware video player
   */
  initVfrPlayer: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('video_vfr_init')
    } catch (e) {
      console.error('[VideoAPI] initVfrPlayer failed:', e)
      throw e
    }
  },

  /**
   * Play a video file with VFR-aware handling
   * @param {string} uri - Video file path or URI
   */
  playVfr: async (uri) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_play', { uri })
    } catch (e) {
      console.error('[VideoAPI] playVfr failed:', e)
      throw e
    }
  },

  /**
   * Pause VFR video playback
   */
  pauseVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_pause')
    } catch (e) {
      console.error('[VideoAPI] pauseVfr failed:', e)
      throw e
    }
  },

  /**
   * Resume VFR video playback
   */
  resumeVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_resume')
    } catch (e) {
      console.error('[VideoAPI] resumeVfr failed:', e)
      throw e
    }
  },

  /**
   * Stop VFR video playback
   */
  stopVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_stop')
    } catch (e) {
      console.error('[VideoAPI] stopVfr failed:', e)
      throw e
    }
  },

  /**
   * Seek to position in seconds
   * @param {number} positionSecs - Position in seconds
   */
  seekVfr: async (positionSecs) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_seek', { positionSecs })
    } catch (e) {
      console.error('[VideoAPI] seekVfr failed:', e)
      throw e
    }
  },

  /**
   * Seek with specific mode
   * @param {number} positionSecs - Position in seconds
   * @param {string} mode - 'keyframe' | 'accurate' | 'snap_to_frame'
   */
  seekVfrWithMode: async (positionSecs, mode) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_seek_with_mode', { positionSecs, mode })
    } catch (e) {
      console.error('[VideoAPI] seekVfrWithMode failed:', e)
      throw e
    }
  },

  /**
   * Step forward one frame (VFR-aware)
   */
  stepFrameVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_step_frame')
    } catch (e) {
      console.error('[VideoAPI] stepFrameVfr failed:', e)
      throw e
    }
  },

  /**
   * Get current playback position in seconds
   * @returns {Promise<number>}
   */
  getPositionVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return 0
    try {
      return await tauriInvoke('video_vfr_get_position')
    } catch (e) {
      console.error('[VideoAPI] getPositionVfr failed:', e)
      return 0
    }
  },

  /**
   * Get video duration in seconds
   * @returns {Promise<number>}
   */
  getDurationVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return 0
    try {
      return await tauriInvoke('video_vfr_get_duration')
    } catch (e) {
      console.error('[VideoAPI] getDurationVfr failed:', e)
      return 0
    }
  },

  /**
   * Get current player state
   * @returns {Promise<string>} 'Idle' | 'Playing' | 'Paused' | 'Buffering' | 'Seeking' | 'EndOfStream' | 'Error'
   */
  getStateVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return 'Idle'
    try {
      return await tauriInvoke('video_vfr_get_state')
    } catch (e) {
      console.error('[VideoAPI] getStateVfr failed:', e)
      return 'Idle'
    }
  },

  /**
   * Get detailed stream information
   * @returns {Promise<{duration_secs: number|null, position_secs: number|null, seekable: boolean, buffering_percent: number|null}>}
   */
  getStreamInfoVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('video_vfr_get_stream_info')
    } catch (e) {
      console.error('[VideoAPI] getStreamInfoVfr failed:', e)
      return null
    }
  },

  /**
   * Get frame rate information
   * @returns {Promise<{is_vfr: boolean, average_fps: number, container_fps: number|null}>}
   */
  getFrameInfoVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('video_vfr_get_frame_info')
    } catch (e) {
      console.error('[VideoAPI] getFrameInfoVfr failed:', e)
      return null
    }
  },

  /**
   * Set the default seek mode
   * @param {string} mode - 'keyframe' | 'accurate' | 'snap_to_frame'
   */
  setSeekModeVfr: async (mode) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_set_seek_mode', { mode })
    } catch (e) {
      console.error('[VideoAPI] setSeekModeVfr failed:', e)
      throw e
    }
  },

  /**
   * Get current seek mode
   * @returns {Promise<string>}
   */
  getSeekModeVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return 'accurate'
    try {
      return await tauriInvoke('video_vfr_get_seek_mode')
    } catch (e) {
      console.error('[VideoAPI] getSeekModeVfr failed:', e)
      return 'accurate'
    }
  },

  /**
   * Set playback rate (speed)
   * @param {number} rate - Playback rate (0.5 = half speed, 2.0 = double speed)
   */
  setRateVfr: async (rate) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_set_rate', { rate })
    } catch (e) {
      console.error('[VideoAPI] setRateVfr failed:', e)
      throw e
    }
  },

  /**
   * Set volume (0.0 to 1.0)
   * @param {number} volume
   */
  setVolumeVfr: async (volume) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_set_volume', { volume })
    } catch (e) {
      console.error('[VideoAPI] setVolumeVfr failed:', e)
      throw e
    }
  },

  /**
   * Get current volume
   * @returns {Promise<number>}
   */
  getVolumeVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return 1.0
    try {
      return await tauriInvoke('video_vfr_get_volume')
    } catch (e) {
      console.error('[VideoAPI] getVolumeVfr failed:', e)
      return 1.0
    }
  },

  /**
   * Set mute state
   * @param {boolean} muted
   */
  setMutedVfr: async (muted) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_set_muted', { muted })
    } catch (e) {
      console.error('[VideoAPI] setMutedVfr failed:', e)
      throw e
    }
  },

  /**
   * Check if muted
   * @returns {Promise<boolean>}
   */
  isMutedVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return false
    try {
      return await tauriInvoke('video_vfr_is_muted')
    } catch (e) {
      console.error('[VideoAPI] isMutedVfr failed:', e)
      return false
    }
  },

  /**
   * Cleanup VFR video player
   */
  cleanupVfr: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_vfr_cleanup')
    } catch (e) {
      console.error('[VideoAPI] cleanupVfr failed:', e)
    }
  },

  // -------------------------------------------------------------------------
  // Transcoding Commands
  // -------------------------------------------------------------------------

  /**
   * Get transcoding capabilities
   * @returns {Promise<{hardware_encoder: string, nvenc_available: boolean, vaapi_available: boolean, quality_presets: string[]}>}
   */
  getTranscodeCapabilities: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('transcode_get_capabilities')
    } catch (e) {
      console.error('[VideoAPI] getTranscodeCapabilities failed:', e)
      return null
    }
  },

  /**
   * Check if a video needs transcoding for browser playback
   * @param {string} videoPath - Path to video file
   * @returns {Promise<boolean>}
   */
  checkTranscodeNeeded: async (videoPath) => {
    await tauriReady
    if (!tauriInvoke) return false
    try {
      return await tauriInvoke('transcode_check_needed', { videoPath })
    } catch (e) {
      console.error('[VideoAPI] checkTranscodeNeeded failed:', e)
      return false
    }
  },

  /**
   * Start transcoding a video
   * @param {string} sourcePath - Source video path
   * @param {string} quality - Quality preset: 'low' | 'medium' | 'high' | 'original'
   * @param {number} [startPosition] - Optional start position in seconds
   * @returns {Promise<{stream_id: string, playlist_path: string, encoder: string}>}
   */
  startTranscode: async (sourcePath, quality, startPosition) => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('transcode_start', {
        request: {
          source_path: sourcePath,
          quality,
          start_position: startPosition
        }
      })
    } catch (e) {
      console.error('[VideoAPI] startTranscode failed:', e)
      throw e
    }
  },

  /**
   * Get transcoding progress
   * @param {string} streamId
   * @returns {Promise<{state: string, progress_percent: number, position_secs: number, duration_secs: number, segments_ready: number, encoding_speed: number, eta_secs: number|null, encoder: string}>}
   */
  getTranscodeProgress: async (streamId) => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('transcode_get_progress', { streamId })
    } catch (e) {
      console.error('[VideoAPI] getTranscodeProgress failed:', e)
      return null
    }
  },

  /**
   * Check if stream is ready for playback
   * @param {string} streamId
   * @returns {Promise<boolean>}
   */
  isTranscodeReady: async (streamId) => {
    await tauriReady
    if (!tauriInvoke) return false
    try {
      return await tauriInvoke('transcode_is_ready', { streamId })
    } catch (e) {
      console.error('[VideoAPI] isTranscodeReady failed:', e)
      return false
    }
  },

  /**
   * Get HLS playlist path for a stream
   * @param {string} streamId
   * @returns {Promise<string>}
   */
  getTranscodePlaylist: async (streamId) => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('transcode_get_playlist', { streamId })
    } catch (e) {
      console.error('[VideoAPI] getTranscodePlaylist failed:', e)
      return null
    }
  },

  /**
   * Stop a transcoding session
   * @param {string} streamId
   */
  stopTranscode: async (streamId) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('transcode_stop', { streamId })
    } catch (e) {
      console.error('[VideoAPI] stopTranscode failed:', e)
    }
  },

  /**
   * Stop all transcoding sessions
   */
  stopAllTranscode: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('transcode_stop_all')
    } catch (e) {
      console.error('[VideoAPI] stopAllTranscode failed:', e)
    }
  },

  // -------------------------------------------------------------------------
  // Interpolation Commands
  // -------------------------------------------------------------------------

  /**
   * Detect available interpolation backends
   * @returns {Promise<Array<{backend: string, available: boolean, status: string, gpu_available: boolean}>>}
   */
  detectInterpolationBackends: async () => {
    await tauriReady
    if (!tauriInvoke) return []
    try {
      return await tauriInvoke('interpolation_detect_backends')
    } catch (e) {
      console.error('[VideoAPI] detectInterpolationBackends failed:', e)
      return []
    }
  },

  /**
   * Get current interpolation configuration
   * @returns {Promise<{enabled: boolean, backend: string, preset: string, target_fps: number}>}
   */
  getInterpolationConfig: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('interpolation_get_config')
    } catch (e) {
      console.error('[VideoAPI] getInterpolationConfig failed:', e)
      return null
    }
  },

  /**
   * Update interpolation configuration
   * @param {Object} config
   */
  setInterpolationConfig: async (config) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('interpolation_set_config', { config })
    } catch (e) {
      console.error('[VideoAPI] setInterpolationConfig failed:', e)
      throw e
    }
  },

  /**
   * Enable or disable interpolation
   * @param {boolean} enabled
   */
  setInterpolationEnabled: async (enabled) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('interpolation_set_enabled', { enabled })
    } catch (e) {
      console.error('[VideoAPI] setInterpolationEnabled failed:', e)
      throw e
    }
  },

  /**
   * Set interpolation backend
   * @param {string} backend - 'svp' | 'rife_ncnn' | 'ffmpeg' | 'none'
   */
  setInterpolationBackend: async (backend) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('interpolation_set_backend', { backend })
    } catch (e) {
      console.error('[VideoAPI] setInterpolationBackend failed:', e)
      throw e
    }
  },

  /**
   * Set target FPS
   * @param {number} fps
   */
  setInterpolationTargetFps: async (fps) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('interpolation_set_target_fps', { fps })
    } catch (e) {
      console.error('[VideoAPI] setInterpolationTargetFps failed:', e)
      throw e
    }
  },

  /**
   * Set quality preset
   * @param {string} preset - 'fast' | 'balanced' | 'quality' | 'max' | 'animation' | 'film'
   */
  setInterpolationPreset: async (preset) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('interpolation_set_preset', { preset })
    } catch (e) {
      console.error('[VideoAPI] setInterpolationPreset failed:', e)
      throw e
    }
  },

  /**
   * Get available presets
   * @returns {Promise<Array<{id: string, name: string, description: string}>>}
   */
  getInterpolationPresets: async () => {
    await tauriReady
    if (!tauriInvoke) return []
    try {
      return await tauriInvoke('interpolation_get_presets')
    } catch (e) {
      console.error('[VideoAPI] getInterpolationPresets failed:', e)
      return []
    }
  },

  /**
   * Check if interpolation is available on this system
   * @returns {Promise<boolean>}
   */
  isInterpolationAvailable: async () => {
    await tauriReady
    if (!tauriInvoke) return false
    try {
      return await tauriInvoke('interpolation_is_available')
    } catch (e) {
      console.error('[VideoAPI] isInterpolationAvailable failed:', e)
      return false
    }
  },

  /**
   * Get recommended backend for given video dimensions
   * @param {number} width
   * @param {number} height
   * @param {boolean} preferQuality
   * @returns {Promise<string|null>}
   */
  recommendInterpolationBackend: async (width, height, preferQuality = false) => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('interpolation_recommend_backend', { width, height, preferQuality })
    } catch (e) {
      console.error('[VideoAPI] recommendInterpolationBackend failed:', e)
      return null
    }
  },

  /**
   * Get GStreamer system information
   * @returns {Promise<string>}
   */
  getSystemInfo: async () => {
    await tauriReady
    if (!tauriInvoke) return null
    try {
      return await tauriInvoke('video_get_system_info')
    } catch (e) {
      console.error('[VideoAPI] getSystemInfo failed:', e)
      return null
    }
  },

  // -------------------------------------------------------------------------
  // Event Streaming
  // -------------------------------------------------------------------------

  /**
   * Start video event streaming
   * Events will be emitted as 'video_event' from Tauri
   * @param {number} [intervalMs=100] - Polling interval in milliseconds
   */
  startEventStream: async (intervalMs = 100) => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_events_start', { intervalMs })
    } catch (e) {
      console.error('[VideoAPI] startEventStream failed:', e)
      throw e
    }
  },

  /**
   * Stop video event streaming
   */
  stopEventStream: async () => {
    await tauriReady
    if (!tauriInvoke) return
    try {
      await tauriInvoke('video_events_stop')
    } catch (e) {
      console.error('[VideoAPI] stopEventStream failed:', e)
    }
  },

  /**
   * Check if event streaming is active
   * @returns {Promise<boolean>}
   */
  isEventStreamActive: async () => {
    await tauriReady
    if (!tauriInvoke) return false
    try {
      return await tauriInvoke('video_events_is_active')
    } catch (e) {
      console.error('[VideoAPI] isEventStreamActive failed:', e)
      return false
    }
  },

  /**
   * Subscribe to video events
   * @param {function} callback - Function to call with each event
   * @returns {function} Unsubscribe function
   */
  subscribeToEvents: async (callback) => {
    await tauriReady
    if (!tauriEvent) return () => {}
    try {
      const unlisten = await tauriEvent.listen('video_event', (event) => {
        callback(event.payload)
      })
      return unlisten
    } catch (e) {
      console.error('[VideoAPI] subscribeToEvents failed:', e)
      return () => {}
    }
  }
}

// Export the Tauri API and detection functions
export { tauriAPI, isTauri, videoControlAPI }
export default getDesktopAPI
