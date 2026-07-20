import { useCallback, useState, useEffect, useRef } from 'react'
import { isTauri, videoControlAPI } from '../../../tauriAPI'

/**
 * Hook for managing video interpolation through Tauri/GStreamer backend
 *
 * This provides frame interpolation for smoother video playback.
 * Features:
 * - Multiple backends: SVP, RIFE-NCNN, FFmpeg minterpolate
 * - Quality presets
 * - Target FPS configuration
 * - GPU acceleration detection
 */
export function useTauriInterpolation() {
  // State
  const [backends, setBackends] = useState([])
  const [config, setConfig] = useState(null)
  const [presets, setPresets] = useState([])
  const [isAvailable, setIsAvailable] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Refs
  const isMountedRef = useRef(true)

  // Check if Tauri is available
  const isTauriAvailable = isTauri()

  // Load available backends
  const loadBackends = useCallback(async () => {
    if (!isTauriAvailable) return []

    try {
      const detectedBackends = await videoControlAPI.detectInterpolationBackends()
      if (isMountedRef.current) {
        setBackends(detectedBackends)

        // Check if any backend is available
        const hasAvailable = detectedBackends.some(b => b.available)
        setIsAvailable(hasAvailable)
      }
      return detectedBackends
    } catch (e) {
      console.error('[TauriInterpolation] loadBackends error:', e)
      return []
    }
  }, [isTauriAvailable])

  // Load current configuration
  const loadConfig = useCallback(async () => {
    if (!isTauriAvailable) return null

    try {
      const currentConfig = await videoControlAPI.getInterpolationConfig()
      if (isMountedRef.current && currentConfig) {
        setConfig(currentConfig)
      }
      return currentConfig
    } catch (e) {
      console.error('[TauriInterpolation] loadConfig error:', e)
      return null
    }
  }, [isTauriAvailable])

  // Load available presets
  const loadPresets = useCallback(async () => {
    if (!isTauriAvailable) return []

    try {
      const availablePresets = await videoControlAPI.getInterpolationPresets()
      if (isMountedRef.current) {
        setPresets(availablePresets)
      }
      return availablePresets
    } catch (e) {
      console.error('[TauriInterpolation] loadPresets error:', e)
      return []
    }
  }, [isTauriAvailable])

  // Update configuration
  const updateConfig = useCallback(async (newConfig) => {
    if (!isTauriAvailable) return

    setIsLoading(true)
    setError(null)

    try {
      await videoControlAPI.setInterpolationConfig(newConfig)
      setConfig(prev => ({ ...prev, ...newConfig }))
    } catch (e) {
      setError(e.message || 'Failed to update configuration')
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [isTauriAvailable])

  // Enable/disable interpolation
  const setEnabled = useCallback(async (enabled) => {
    if (!isTauriAvailable) return

    setIsLoading(true)
    setError(null)

    try {
      await videoControlAPI.setInterpolationEnabled(enabled)
      setConfig(prev => prev ? { ...prev, enabled } : null)
    } catch (e) {
      setError(e.message || 'Failed to toggle interpolation')
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [isTauriAvailable])

  // Set backend
  const setBackend = useCallback(async (backend) => {
    if (!isTauriAvailable) return

    setIsLoading(true)
    setError(null)

    try {
      await videoControlAPI.setInterpolationBackend(backend)
      setConfig(prev => prev ? { ...prev, backend } : null)
    } catch (e) {
      setError(e.message || 'Failed to set backend')
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [isTauriAvailable])

  // Set target FPS
  const setTargetFps = useCallback(async (fps) => {
    if (!isTauriAvailable) return

    setIsLoading(true)
    setError(null)

    try {
      await videoControlAPI.setInterpolationTargetFps(fps)
      setConfig(prev => prev ? { ...prev, target_fps: fps } : null)
    } catch (e) {
      setError(e.message || 'Failed to set target FPS')
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [isTauriAvailable])

  // Set quality preset
  const setPreset = useCallback(async (preset) => {
    if (!isTauriAvailable) return

    setIsLoading(true)
    setError(null)

    try {
      await videoControlAPI.setInterpolationPreset(preset)
      setConfig(prev => prev ? { ...prev, preset } : null)
    } catch (e) {
      setError(e.message || 'Failed to set preset')
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [isTauriAvailable])

  // Get recommended backend for video dimensions
  const getRecommendedBackend = useCallback(async (width, height, preferQuality = false) => {
    if (!isTauriAvailable) return null

    try {
      return await videoControlAPI.recommendInterpolationBackend(width, height, preferQuality)
    } catch (e) {
      console.error('[TauriInterpolation] getRecommendedBackend error:', e)
      return null
    }
  }, [isTauriAvailable])

  // Check if a specific backend is available
  const isBackendAvailable = useCallback((backendName) => {
    const backend = backends.find(b => b.backend === backendName)
    return backend?.available || false
  }, [backends])

  // Get backend info
  const getBackendInfo = useCallback((backendName) => {
    return backends.find(b => b.backend === backendName) || null
  }, [backends])

  // Get best available backend
  const getBestBackend = useCallback(() => {
    const available = backends.filter(b => b.available && b.backend !== 'none')
    if (available.length === 0) return null

    // Sort by performance tier (higher is better)
    available.sort((a, b) => (b.performance_tier || 0) - (a.performance_tier || 0))
    return available[0]
  }, [backends])

  // Load all data on mount
  useEffect(() => {
    if (isTauriAvailable) {
      Promise.all([
        loadBackends(),
        loadConfig(),
        loadPresets()
      ]).catch(console.error)
    }
  }, [isTauriAvailable, loadBackends, loadConfig, loadPresets])

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  return {
    // State
    isTauriAvailable,
    isAvailable,
    isLoading,
    error,
    backends,
    config,
    presets,

    // Functions
    loadBackends,
    loadConfig,
    loadPresets,
    updateConfig,
    setEnabled,
    setBackend,
    setTargetFps,
    setPreset,
    getRecommendedBackend,
    isBackendAvailable,
    getBackendInfo,
    getBestBackend
  }
}

export default useTauriInterpolation
