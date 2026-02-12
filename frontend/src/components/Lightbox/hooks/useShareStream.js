import { useState, useCallback, useRef, useEffect } from 'react'
import { createShareSession, stopShareSession, syncShareState } from '../../../api'

/**
 * Hook for managing share stream (host side).
 * Creates a session, syncs host state on play/pause/seek.
 */
export function useShareStream(mediaRef, { imageId, directoryId, isVideoFile }) {
  const [isSharing, setIsSharing] = useState(false)
  const [shareToken, setShareToken] = useState(null)
  const [shareUrl, setShareUrl] = useState(null)
  const syncTimerRef = useRef(null)
  const lastSyncRef = useRef({ playing: false, position: 0, speed: 1 })

  // Send host state to server (debounced)
  const sendSync = useCallback((state) => {
    if (!shareToken) return

    // Debounce: don't send more than once per 200ms
    if (syncTimerRef.current) clearTimeout(syncTimerRef.current)
    syncTimerRef.current = setTimeout(() => {
      syncShareState(shareToken, state).catch(() => {})
      lastSyncRef.current = { ...lastSyncRef.current, ...state }
    }, 200)
  }, [shareToken])

  // Start sharing
  const startSharing = useCallback(async () => {
    if (!imageId || !isVideoFile) return
    try {
      const data = await createShareSession(imageId, directoryId)
      setShareToken(data.token)
      setShareUrl(data.share_url)
      setIsSharing(true)
    } catch (e) {
      console.error('[Share] Failed to create session:', e)
    }
  }, [imageId, isVideoFile])

  // Stop sharing
  const stopSharing = useCallback(async () => {
    if (shareToken) {
      try {
        await stopShareSession(shareToken)
      } catch (e) { /* ignore */ }
    }
    setShareToken(null)
    setShareUrl(null)
    setIsSharing(false)
  }, [shareToken])

  // Sync on play/pause events
  useEffect(() => {
    if (!isSharing || !mediaRef.current) return
    const video = mediaRef.current

    const onPlay = () => sendSync({ playing: true, position: video.currentTime })
    const onPause = () => sendSync({ playing: false, position: video.currentTime })
    const onSeeked = () => sendSync({ position: video.currentTime, playing: !video.paused })
    const onRateChange = () => sendSync({ speed: video.playbackRate })

    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('seeked', onSeeked)
    video.addEventListener('ratechange', onRateChange)

    // Periodic position sync every 5s while playing
    const interval = setInterval(() => {
      if (!video.paused) {
        syncShareState(shareToken, {
          playing: true,
          position: video.currentTime,
          speed: video.playbackRate,
        }).catch(() => {})
      }
    }, 5000)

    return () => {
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('ratechange', onRateChange)
      clearInterval(interval)
    }
  }, [isSharing, mediaRef, sendSync, shareToken])

  // Cleanup on unmount or image change
  useEffect(() => {
    return () => {
      if (shareToken) {
        stopShareSession(shareToken).catch(() => {})
      }
      if (syncTimerRef.current) clearTimeout(syncTimerRef.current)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    isSharing,
    shareToken,
    shareUrl,
    startSharing,
    stopSharing,
  }
}
