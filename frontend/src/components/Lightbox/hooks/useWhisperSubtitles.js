import { useCallback, useState, useRef, useEffect } from 'react'
import {
  getWhisperConfig,
  installWhisper,
  generateSubtitles,
  stopSubtitles,
  subscribeToSubtitleEvents,
} from '../../../api'
import { isVideo } from '../utils/helpers'

/**
 * Hook for managing whisper-based subtitle generation and display.
 *
 * Uses SSE events to add VTTCue objects dynamically to a TextTrack,
 * avoiding <track src> reloads which reset playback state.
 */
export function useWhisperSubtitles(mediaRef, image) {
  const [whisperConfig, setWhisperConfig] = useState(null)
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [streamId, setStreamId] = useState(null)
  const [vttUrl, setVttUrl] = useState(null)
  const [completed, setCompleted] = useState(false)

  // Per-session language/task overrides (initialized from config)
  const [subtitleLanguage, setSubtitleLanguage] = useState(null)
  const [subtitleTask, setSubtitleTask] = useState(null)

  const unsubscribeRef = useRef(null)
  const trackRef = useRef(null) // Reference to the TextTrack
  const installPollRef = useRef(null) // Poll interval for install status
  const hadActiveStreamRef = useRef(false) // Track if generation was ever started (for cleanup)

  // Load whisper config on mount
  useEffect(() => {
    async function loadConfig() {
      try {
        const config = await getWhisperConfig()
        setWhisperConfig(config)
        // Initialize per-session overrides from config defaults
        if (subtitleLanguage === null) setSubtitleLanguage(config.language || 'ja')
        if (subtitleTask === null) setSubtitleTask(config.task || 'translate')
      } catch (err) {
        console.error('[Whisper] Failed to load config:', err)
      }
    }
    loadConfig()
  }, [])

  // Apply subtitle appearance settings via dynamic <style> element
  useEffect(() => {
    if (!whisperConfig) return

    const font = whisperConfig.subtitle_font || 'Trebuchet MS'
    const fontSize = whisperConfig.subtitle_font_size || 1.3
    const style = whisperConfig.subtitle_style || 'outline'
    const color = whisperConfig.subtitle_color || '#ffffff'
    const outlineColor = whisperConfig.subtitle_outline_color || '#000000'
    const bgOpacity = whisperConfig.subtitle_bg_opacity ?? 0.75

    let background = 'transparent'
    let textShadow = 'none'

    if (style === 'outline' || style === 'outline_background') {
      textShadow = [
        `-1px -1px 0 ${outlineColor}`,
        `1px -1px 0 ${outlineColor}`,
        `-1px 1px 0 ${outlineColor}`,
        `1px 1px 0 ${outlineColor}`,
        `0 0 6px ${outlineColor}80`,
      ].join(', ')
    }
    if (style === 'background' || style === 'outline_background') {
      background = `rgba(0, 0, 0, ${bgOpacity})`
    }

    const css = `video::cue {
  background: ${background};
  color: ${color};
  font-size: ${fontSize}rem;
  line-height: 1.4;
  font-family: '${font}', 'Arial', sans-serif;
  font-weight: 700;
  text-shadow: ${textShadow};
}`

    let styleEl = document.getElementById('whisper-subtitle-style')
    if (!styleEl) {
      styleEl = document.createElement('style')
      styleEl.id = 'whisper-subtitle-style'
      document.head.appendChild(styleEl)
    }
    styleEl.textContent = css

    return () => {
      const el = document.getElementById('whisper-subtitle-style')
      if (el) el.remove()
    }
  }, [whisperConfig])

  // Auto-dismiss error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [error])

  // Cleanup SSE on image change (close stale EventSource before new one opens)
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current()
        unsubscribeRef.current = null
      }
    }
  }, [image?.id])

  // Cleanup SSE, install poll, and backend stream on unmount
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current()
        unsubscribeRef.current = null
      }
      if (installPollRef.current) {
        clearInterval(installPollRef.current)
        installPollRef.current = null
      }
      // Only stop backend generation if we actually started one
      if (hadActiveStreamRef.current) {
        stopSubtitles().catch(() => {})
      }
    }
  }, [])

  // Helper: get or create a TextTrack on the video element
  const getOrCreateTrack = useCallback(() => {
    const video = mediaRef.current
    if (!video) return null

    // Look for existing track we created
    for (let i = 0; i < video.textTracks.length; i++) {
      if (video.textTracks[i].label === 'Whisper Subtitles') {
        return video.textTracks[i]
      }
    }

    // Create new track via addTextTrack (no <track> element needed)
    const track = video.addTextTrack('subtitles', 'Whisper Subtitles', 'en')
    track.mode = 'showing'
    return track
  }, [mediaRef])

  // Helper: clear all cues from track
  const clearTrackCues = useCallback(() => {
    const video = mediaRef.current
    if (!video || !video.textTracks) return

    for (let i = 0; i < video.textTracks.length; i++) {
      const track = video.textTracks[i]
      if (track.label === 'Whisper Subtitles') {
        // Remove all cues
        while (track.cues && track.cues.length > 0) {
          track.removeCue(track.cues[0])
        }
      }
    }
    trackRef.current = null
  }, [mediaRef])

  // Load cached VTT into track element
  const loadCachedVtt = useCallback((url) => {
    const video = mediaRef.current
    if (!video) return

    // For cached VTTs, use a <track> element with src
    // Use the URL directly (not getMediaUrl) because <track> elements require
    // same-origin. The /api/... path goes through the Vite proxy in dev mode.
    // First remove any existing whisper track elements
    const existingTracks = video.querySelectorAll('track[data-whisper]')
    existingTracks.forEach(t => t.remove())

    const trackEl = document.createElement('track')
    trackEl.kind = 'subtitles'
    trackEl.label = 'Whisper Subtitles'
    trackEl.srclang = 'en'
    trackEl.src = url
    trackEl.default = true
    trackEl.dataset.whisper = 'true'
    video.appendChild(trackEl)

    // Set mode to showing after adding
    setTimeout(() => {
      for (let i = 0; i < video.textTracks.length; i++) {
        if (video.textTracks[i].label === 'Whisper Subtitles') {
          video.textTracks[i].mode = 'showing'
        }
      }
    }, 100)
  }, [mediaRef])

  // Start subtitle generation
  const startSubtitles = useCallback(async (langOverride, taskOverride) => {
    if (!image || !isVideo(image.filename)) return
    if (generating) return

    const lang = langOverride !== undefined ? langOverride : subtitleLanguage
    const task = taskOverride !== undefined ? taskOverride : subtitleTask

    setGenerating(true)
    setError(null)
    setProgress(0)
    setCompleted(false)

    try {
      // Start from current playback position (like mpv whisper-subs)
      const currentTime = mediaRef.current?.currentTime || 0
      const result = await generateSubtitles(image.file_path, lang, task, currentTime)

      if (!result.success) {
        setError(result.error || 'Failed to generate subtitles')
        setGenerating(false)
        return
      }

      setStreamId(result.stream_id)
      setVttUrl(result.vtt_url)
      hadActiveStreamRef.current = true

      if (result.cached && result.completed) {
        // Cached VTT - load directly
        loadCachedVtt(result.vtt_url)
        setCompleted(true)
        setGenerating(false)
        setSubtitlesEnabled(true)
        return
      }

      // Live generation - subscribe to SSE events
      const track = getOrCreateTrack()
      trackRef.current = track

      const unsubscribe = subscribeToSubtitleEvents(result.stream_id, (event) => {
        switch (event.type) {
          case 'subtitle_cue': {
            // Add cue dynamically
            const { start, end, text } = event.data
            if (track && text) {
              try {
                const cue = new VTTCue(start, end, text)
                track.addCue(cue)
              } catch (e) {
                console.error('[Whisper] Failed to add cue:', e)
              }
            }
            break
          }
          case 'subtitle_progress': {
            setProgress(event.data.progress || 0)
            break
          }
          case 'subtitle_completed': {
            setCompleted(true)
            setGenerating(false)
            break
          }
          case 'subtitle_error': {
            setError(event.data.error || 'Subtitle generation failed')
            setGenerating(false)
            break
          }
        }
      })

      unsubscribeRef.current = unsubscribe
      setSubtitlesEnabled(true)

    } catch (err) {
      console.error('[Whisper] Generation error:', err)
      setError(err.message || 'Failed to start subtitle generation')
      setGenerating(false)
    }
  }, [image, generating, subtitleLanguage, subtitleTask, mediaRef, getOrCreateTrack, loadCachedVtt])

  // Install faster-whisper and then generate subtitles
  const installAndGenerate = useCallback(async () => {
    if (installing) return

    setInstalling(true)
    setError(null)

    try {
      const result = await installWhisper()

      if (result.message === 'faster-whisper is already installed') {
        // Already installed, just generate
        setInstalling(false)
        // Refresh config to update status
        const config = await getWhisperConfig()
        setWhisperConfig(config)
        await startSubtitles()
        return
      }

      if (!result.success) {
        setError(result.error || 'Failed to start installation')
        setInstalling(false)
        return
      }

      // Poll until install completes
      installPollRef.current = setInterval(async () => {
        try {
          const config = await getWhisperConfig()
          setWhisperConfig(config)

          if (!config.installing) {
            clearInterval(installPollRef.current)
            installPollRef.current = null
            setInstalling(false)

            if (config.status?.faster_whisper_installed) {
              // Install succeeded - auto-generate
              await startSubtitles()
            } else {
              setError(config.install_progress || 'Installation failed')
            }
          }
        } catch (err) {
          // Keep polling on transient errors
        }
      }, 2000)

      // Safety timeout after 10 minutes
      setTimeout(() => {
        if (installPollRef.current) {
          clearInterval(installPollRef.current)
          installPollRef.current = null
          setInstalling(false)
          setError('Installation timed out')
        }
      }, 600000)

    } catch (err) {
      console.error('[Whisper] Install error:', err)
      setError(err.message || 'Failed to install faster-whisper')
      setInstalling(false)
    }
  }, [installing, startSubtitles])

  // Stop subtitle generation and hide subtitles
  const stopSubtitlesStream = useCallback(async () => {
    // Unsubscribe from SSE
    if (unsubscribeRef.current) {
      unsubscribeRef.current()
      unsubscribeRef.current = null
    }

    // Clear track cues
    clearTrackCues()

    // Remove any <track> elements we added
    const video = mediaRef.current
    if (video) {
      const tracks = video.querySelectorAll('track[data-whisper]')
      tracks.forEach(t => t.remove())
      // Also hide any text tracks
      for (let i = 0; video.textTracks && i < video.textTracks.length; i++) {
        if (video.textTracks[i].label === 'Whisper Subtitles') {
          video.textTracks[i].mode = 'disabled'
        }
      }
    }

    // Stop backend generation
    if (generating || streamId) {
      try {
        await stopSubtitles()
      } catch (err) {
        console.error('[Whisper] Failed to stop:', err)
      }
    }

    setSubtitlesEnabled(false)
    setGenerating(false)
    setStreamId(null)
    setVttUrl(null)
    setProgress(0)
    setCompleted(false)
    setError(null)
  }, [generating, streamId, clearTrackCues, mediaRef])

  // Toggle subtitles on/off (with auto-install if needed)
  const toggleSubtitles = useCallback(async () => {
    if (subtitlesEnabled) {
      await stopSubtitlesStream()
    } else if (!whisperConfig?.status?.faster_whisper_installed || whisperConfig?.status?.error) {
      // Not installed or broken - install/repair first, then generate
      await installAndGenerate()
    } else {
      await startSubtitles()
    }
  }, [subtitlesEnabled, whisperConfig, stopSubtitlesStream, startSubtitles, installAndGenerate])

  // Restart subtitles with new language/task settings
  const restartWithSettings = useCallback(async (language, task) => {
    setSubtitleLanguage(language)
    setSubtitleTask(task)

    // If currently showing subtitles, stop and restart with new settings
    if (subtitlesEnabled || generating) {
      await stopSubtitlesStream()
      // Small delay to ensure cleanup completes
      await new Promise(r => setTimeout(r, 100))
    }
    await startSubtitles(language, task)
  }, [subtitlesEnabled, generating, stopSubtitlesStream, startSubtitles])

  // Auto-generate if enabled in config (requires faster-whisper to be installed)
  const autoGenerate = useCallback(async () => {
    if (whisperConfig?.auto_generate && whisperConfig?.status?.faster_whisper_installed && image && isVideo(image.filename)) {
      await startSubtitles()
    }
  }, [whisperConfig, image, startSubtitles])

  return {
    whisperConfig,
    subtitlesEnabled,
    generating,
    installing,
    progress,
    error,
    completed,
    subtitleLanguage,
    subtitleTask,
    setSubtitleLanguage,
    setSubtitleTask,
    toggleSubtitles,
    stopSubtitlesStream,
    autoGenerate,
    startSubtitles,
    restartWithSettings,
  }
}
