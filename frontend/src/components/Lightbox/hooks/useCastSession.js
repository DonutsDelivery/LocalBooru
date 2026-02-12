/**
 * Hook for managing Chromecast/DLNA casting from the Lightbox.
 *
 * Handles device discovery, starting/stopping cast sessions,
 * SSE status subscription, and local↔remote video position sync.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import {
  getCastDevices,
  refreshCastDevices,
  castPlay,
  castControl,
  castStop,
  subscribeToCastEvents,
  getCastConfig,
} from '../../../api'

export function useCastSession(mediaRef, image) {
  const [castConfig, setCastConfig] = useState(null)
  const [devices, setDevices] = useState([])
  const [isCasting, setIsCasting] = useState(false)
  const [castStatus, setCastStatus] = useState(null) // {state, current_time, duration, volume, title}
  const [castError, setCastError] = useState(null)
  const [showDevicePicker, setShowDevicePicker] = useState(false)
  const [devicesLoading, setDevicesLoading] = useState(false)

  const sseCleanupRef = useRef(null)
  const localPositionRef = useRef(0) // Position when cast started (to resume local)

  // Load cast config on mount
  useEffect(() => {
    getCastConfig()
      .then(setCastConfig)
      .catch(() => {})
  }, [])

  // Clean up SSE on unmount
  useEffect(() => {
    return () => {
      if (sseCleanupRef.current) {
        sseCleanupRef.current()
        sseCleanupRef.current = null
      }
    }
  }, [])

  // Stop casting when image changes
  useEffect(() => {
    if (isCasting) {
      handleStopCasting()
    }
  }, [image?.id])

  // Subscribe to cast events
  const subscribeToCast = useCallback(() => {
    if (sseCleanupRef.current) {
      sseCleanupRef.current()
    }

    sseCleanupRef.current = subscribeToCastEvents((event) => {
      if (event.type === 'cast_status') {
        setCastStatus(event.data)
      } else if (event.type === 'cast_disconnected') {
        setIsCasting(false)
        setCastStatus(null)
        // Resume local video at cast position
        if (mediaRef.current && event.data?.current_time) {
          mediaRef.current.currentTime = event.data.current_time
          mediaRef.current.play().catch(() => {})
        }
        if (sseCleanupRef.current) {
          sseCleanupRef.current()
          sseCleanupRef.current = null
        }
      } else if (event.type === 'cast_error') {
        setCastError(event.data?.error || 'Cast error')
        setTimeout(() => setCastError(null), 5000)
      }
    })
  }, [mediaRef])

  // Load devices when picker opens
  const handleToggleDevicePicker = useCallback(async () => {
    if (showDevicePicker) {
      setShowDevicePicker(false)
      return
    }

    if (!castConfig?.enabled) return

    setDevicesLoading(true)
    setShowDevicePicker(true)
    try {
      const result = await getCastDevices()
      setDevices(result.devices || [])
    } catch (e) {
      console.error('[Cast] Failed to get devices:', e)
    }
    setDevicesLoading(false)
  }, [showDevicePicker, castConfig?.enabled])

  // Refresh device list
  const handleRefreshDevices = useCallback(async () => {
    setDevicesLoading(true)
    try {
      const result = await refreshCastDevices()
      setDevices(result.devices || [])
    } catch (e) {
      console.error('[Cast] Failed to refresh devices:', e)
    }
    setDevicesLoading(false)
  }, [])

  // Start casting to a device
  const handleStartCasting = useCallback(async (deviceId) => {
    if (!image) return

    // Save local position and pause local video
    if (mediaRef.current) {
      localPositionRef.current = mediaRef.current.currentTime
      mediaRef.current.pause()
    }

    setShowDevicePicker(false)
    setCastError(null)

    try {
      const result = await castPlay(
        deviceId,
        image.file_path,
        image.id,
        image.directory_id
      )

      if (result.success) {
        setIsCasting(true)
        subscribeToCast()
      } else {
        setCastError(result.error || 'Failed to start casting')
        setTimeout(() => setCastError(null), 5000)
        // Resume local video
        if (mediaRef.current) {
          mediaRef.current.play().catch(() => {})
        }
      }
    } catch (e) {
      console.error('[Cast] Failed to start casting:', e)
      setCastError(e.message || 'Failed to start casting')
      setTimeout(() => setCastError(null), 5000)
      if (mediaRef.current) {
        mediaRef.current.play().catch(() => {})
      }
    }
  }, [image, mediaRef, subscribeToCast])

  // Stop casting
  const handleStopCasting = useCallback(async () => {
    try {
      await castStop()
    } catch (e) {
      console.error('[Cast] Failed to stop casting:', e)
    }
    setIsCasting(false)

    if (sseCleanupRef.current) {
      sseCleanupRef.current()
      sseCleanupRef.current = null
    }

    // Resume local video at cast position
    if (mediaRef.current && castStatus?.current_time) {
      mediaRef.current.currentTime = castStatus.current_time
      mediaRef.current.play().catch(() => {})
    }
    setCastStatus(null)
  }, [mediaRef, castStatus])

  // Remote control functions
  const handleCastPause = useCallback(async () => {
    try {
      await castControl('pause')
    } catch (e) {
      console.error('[Cast] Pause error:', e)
    }
  }, [])

  const handleCastResume = useCallback(async () => {
    try {
      await castControl('resume')
    } catch (e) {
      console.error('[Cast] Resume error:', e)
    }
  }, [])

  const handleCastSeek = useCallback(async (position) => {
    try {
      await castControl('seek', position)
    } catch (e) {
      console.error('[Cast] Seek error:', e)
    }
  }, [])

  const handleCastVolume = useCallback(async (level) => {
    try {
      await castControl('volume', level)
    } catch (e) {
      console.error('[Cast] Volume error:', e)
    }
  }, [])

  return {
    castConfig,
    devices,
    isCasting,
    castStatus,
    castError,
    showDevicePicker,
    devicesLoading,
    toggleDevicePicker: handleToggleDevicePicker,
    refreshDevices: handleRefreshDevices,
    startCasting: handleStartCasting,
    stopCasting: handleStopCasting,
    castPause: handleCastPause,
    castResume: handleCastResume,
    castSeek: handleCastSeek,
    castVolume: handleCastVolume,
  }
}
