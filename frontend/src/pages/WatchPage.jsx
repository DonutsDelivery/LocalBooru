import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { getShareInfo, subscribeToShareEvents, getShareHlsUrl } from '../api'
import Hls from 'hls.js'
import './WatchPage.css'

export default function WatchPage() {
  const { token } = useParams()
  const videoRef = useRef(null)
  const hlsRef = useRef(null)
  const [sessionInfo, setSessionInfo] = useState(null)
  const [error, setError] = useState(null)
  const [synced, setSynced] = useState(true)
  const [disconnected, setDisconnected] = useState(false)
  const lastSyncRef = useRef(null)
  const isSeeking = useRef(false)

  // Load session info
  useEffect(() => {
    let cancelled = false

    async function loadInfo() {
      try {
        const info = await getShareInfo(token)
        if (!cancelled) {
          setSessionInfo(info)
        }
      } catch (e) {
        if (!cancelled) {
          setError('Session not found or expired')
        }
      }
    }

    loadInfo()
    return () => { cancelled = true }
  }, [token])

  // Set up HLS playback
  useEffect(() => {
    if (!sessionInfo?.stream_id || !videoRef.current) return

    const hlsUrl = getShareHlsUrl(token)

    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
      })
      hls.loadSource(hlsUrl)
      hls.attachMedia(videoRef.current)
      hlsRef.current = hls

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          console.error('[Watch] HLS fatal error:', data)
          if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            hls.startLoad()
          }
        }
      })

      return () => {
        hls.destroy()
        hlsRef.current = null
      }
    } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
      videoRef.current.src = hlsUrl
    }
  }, [sessionInfo, token])

  // Apply sync state from host
  const applySync = useCallback((state) => {
    const video = videoRef.current
    if (!video || isSeeking.current) return

    const { playing, position, speed } = state.data || state

    // Match playback speed
    if (speed !== undefined && video.playbackRate !== speed) {
      video.playbackRate = speed
    }

    // Seek if diverged > 2 seconds
    if (position !== undefined) {
      const drift = Math.abs(video.currentTime - position)
      if (drift > 2) {
        isSeeking.current = true
        video.currentTime = position
        setTimeout(() => { isSeeking.current = false }, 500)
      }
    }

    // Match play/pause state
    if (playing !== undefined) {
      if (playing && video.paused) {
        video.play().catch(() => {})
        setSynced(true)
      } else if (!playing && !video.paused) {
        video.pause()
        setSynced(true)
      }
    }
  }, [])

  // Subscribe to SSE sync events
  useEffect(() => {
    if (!sessionInfo) return

    const unsub = subscribeToShareEvents(token, (event) => {
      if (event.type === 'sync') {
        lastSyncRef.current = event
        applySync(event)
        setSynced(true)
      } else if (event.type === 'disconnected') {
        setDisconnected(true)
      }
    })

    return unsub
  }, [sessionInfo, token, applySync])

  if (error) {
    return (
      <div className="watch-page">
        <div className="watch-error">
          <h2>Session Unavailable</h2>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  if (!sessionInfo) {
    return (
      <div className="watch-page">
        <div className="watch-loading">Connecting...</div>
      </div>
    )
  }

  return (
    <div className="watch-page">
      <div className="watch-header">
        <span className="watch-title">{sessionInfo.original_filename}</span>
        <div className="watch-badges">
          {disconnected ? (
            <span className="watch-badge disconnected">Host disconnected</span>
          ) : synced ? (
            <span className="watch-badge synced">Synced</span>
          ) : (
            <span className="watch-badge buffering">Buffering...</span>
          )}
        </div>
      </div>
      <div className="watch-video-container">
        <video
          ref={videoRef}
          className="watch-video"
          controls
          autoPlay
          playsInline
        />
      </div>
    </div>
  )
}
