/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { getMediaUrl, fetchRelatedMusic } from '../../api'
import { nativeVideoAPI, videoControlAPI, isTauri } from '../../tauriAPI'
import { advanceMusicSession, appendRelatedTracks, musicTrackKey, nextMusicTrack, orderedAlbumTracks, removeExplicitQueueTrack, upcomingAlbumTracks } from './musicQueue'
import MusicLightbox from './MusicLightbox'
import './Music.css'

const MusicPlayerContext = createContext(null)

const emptyBrowse = () => ({ mode: 'albums', byMode: {
  albums: { query: '', artist: '', album: '', genre: '', year: '', favorites: false, folder: '', collection: '', library: '', scroll: 0, page: 1 },
  songs: { query: '', artist: '', album: '', genre: '', year: '', favorites: false, folder: '', collection: '', library: '', scroll: 0, page: 1 },
} })

function initialBrowse() {
  try {
    const stored = JSON.parse(sessionStorage.getItem('dmc_music_browse'))
    if (stored?.byMode?.albums && stored?.byMode?.songs) return stored
  } catch { /* use defaults */ }
  return emptyBrowse()
}

export function MusicPlayerProvider({ children }) {
  const [browse, setBrowse] = useState(initialBrowse)
  const [session, setSession] = useState(null)
  const [viewer, setViewer] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.8)
  const audioRef = useRef(null)
  const currentTrack = session?.current
  const sessionRef = useRef(session)
  const relatedRequestRef = useRef(null)
  const waitingForNextRef = useRef(false)
  const generationRef = useRef(0)
  useEffect(() => { sessionRef.current = session }, [session])

  useEffect(() => { sessionStorage.setItem('dmc_music_browse', JSON.stringify(browse)) }, [browse])

  const updateBrowse = useCallback((mode, patch) => {
    setBrowse(previous => ({ ...previous, byMode: {
      ...previous.byMode,
      [mode]: { ...previous.byMode[mode], ...patch },
    } }))
  }, [])

  const startAlbum = useCallback((album, tracks, selectedTrack = null) => {
    const ordered = orderedAlbumTracks(tracks)
    const start = selectedTrack || ordered[0]
    if (!start) return
    const albumCursor = Math.max(0, ordered.findIndex(track => musicTrackKey(track) === musicTrackKey(start)))
    generationRef.current += 1
    waitingForNextRef.current = false
    relatedRequestRef.current = null
    setSession({
      kind: 'album', album, albumTracks: ordered, albumCursor,
      seed: start, current: start, relatedShuffle: false,
      explicitQueue: [], recommendations: [], history: [],
      playedKeys: [musicTrackKey(start)], relatedExhausted: false, loadingRelated: false, notice: null, playbackError: null,
    })
    setPosition(0)
    setDuration(Number(start.duration) || 0)
    setPlaying(true)
    setViewer({ kind: 'session' })
  }, [])

  const startSong = useCallback(track => {
    generationRef.current += 1
    waitingForNextRef.current = false
    relatedRequestRef.current = null
    setSession({
      kind: 'mix', album: null, albumTracks: [], albumCursor: -1,
      seed: track, current: track, relatedShuffle: true,
      explicitQueue: [], recommendations: [], history: [],
      playedKeys: [musicTrackKey(track)], relatedExhausted: false, loadingRelated: false, notice: null, playbackError: null,
    })
    setPosition(0)
    setDuration(Number(track.duration) || 0)
    setPlaying(true)
    setViewer({ kind: 'session' })
  }, [])

  const advance = useCallback(() => {
    const current = session
    const choice = nextMusicTrack(current)
    if (choice) {
      waitingForNextRef.current = false
      setSession(previous => advanceMusicSession(previous, choice))
      setPosition(0)
      setDuration(Number(choice.track.duration) || 0)
      setPlaying(true)
    } else if (current && !current.relatedExhausted) {
      // Prefetch starts in an effect, so Next can arrive before loadingRelated
      // turns true. Wait for that first result instead of declaring exhaustion.
      waitingForNextRef.current = true
    } else {
      waitingForNextRef.current = false
      setPlaying(false)
      setSession(previous => previous ? { ...previous, notice: previous.notice || 'No further related songs in this library.' } : previous)
    }
  }, [session])

  const previous = useCallback(() => {
    if (audioRef.current?.currentTime > 3) {
      audioRef.current.currentTime = 0
      return
    }
    const current = sessionRef.current
    if (!current?.history.length) {
      if (audioRef.current) audioRef.current.currentTime = 0
      return
    }
    const track = current.history.at(-1)
    setSession({ ...current, current: track, history: current.history.slice(0, -1),
      explicitQueue: [current.current, ...current.explicitQueue] })
    setPosition(0)
    setDuration(Number(track.duration) || 0)
    setPlaying(true)
  }, [])

  const queueTrack = useCallback(track => {
    setSession(previous => previous ? { ...previous, explicitQueue: [...previous.explicitQueue, track], notice: null } : previous)
  }, [])

  const removeQueuedTrack = useCallback(index => {
    setSession(previous => removeExplicitQueueTrack(previous, index))
  }, [])

  const reportPlaybackError = useCallback((track, error) => {
    // Pausing for video or replacing a source can reject an in-flight play()
    // with AbortError even though the audio file itself is healthy.
    if (error?.name === 'AbortError') return
    if (!track || !sessionRef.current?.current ||
      musicTrackKey(sessionRef.current.current) !== musicTrackKey(track)) return
    const message = error?.name === 'NotAllowedError'
      ? 'Playback was blocked. Press Play to try again.'
      : `Could not play “${track.title || 'this song'}”. The file may be missing or unsupported.`
    setPlaying(false)
    setSession(previous => previous && musicTrackKey(previous.current) === musicTrackKey(track)
      ? { ...previous, playbackError: message }
      : previous)
  }, [])

  const setRelatedShuffle = useCallback(enabled => {
    setSession(previous => previous?.kind === 'album' ? { ...previous, relatedShuffle: enabled } : previous)
  }, [])

  const openAlbum = useCallback((album, tracks) => setViewer({ kind: 'album', album, tracks: orderedAlbumTracks(tracks) }), [])
  const openSession = useCallback(() => { if (sessionRef.current) setViewer({ kind: 'session' }) }, [])
  const closeViewer = useCallback(() => setViewer(null), [])
  const seek = useCallback(time => {
    if (!audioRef.current || !Number.isFinite(time)) return
    audioRef.current.currentTime = Math.max(0, Math.min(time, audioRef.current.duration || time))
    setPosition(audioRef.current.currentTime)
  }, [])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    audio.volume = volume
  }, [volume])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !currentTrack) return
    audio.src = getMediaUrl(currentTrack.stream_url)
    audio.load()
  }, [currentTrack])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (playing && currentTrack) audio.play().catch(error => reportPlaybackError(currentTrack, error))
    else audio.pause()
  }, [playing, currentTrack, reportPlaybackError])

  // Video playback pauses music at the actual play event, not on gallery open.
  useEffect(() => {
    const onMediaPlay = event => {
      if (event.target?.tagName === 'VIDEO') setPlaying(false)
    }
    document.addEventListener('play', onMediaPlay, true)
    return () => document.removeEventListener('play', onMediaPlay, true)
  }, [])

  useEffect(() => {
    if (!isTauri()) return
    let disposed = false
    let cleanupNative = () => {}
    let cleanupVfr = () => {}
    nativeVideoAPI.subscribe({
      onSnapshot: snapshot => {
        if (snapshot?.presentation === 'native_video' && snapshot.playback?.paused === false) setPlaying(false)
      },
    }).then(cleanup => { if (disposed) cleanup(); else cleanupNative = cleanup }).catch(() => {})
    videoControlAPI.subscribeToEvents(event => {
      if (event?.type === 'state_changed' && event.state === 'Playing') setPlaying(false)
    }).then(cleanup => { if (disposed) cleanup(); else cleanupVfr = cleanup }).catch(() => {})
    return () => { disposed = true; cleanupNative(); cleanupVfr() }
  }, [])

  useEffect(() => {
    if (!session || session.loadingRelated || session.relatedExhausted || session.recommendations.length >= 5) return
    const generation = generationRef.current
    const requestKey = `${generation}:${session.playedKeys.join(',')}:${session.recommendations.map(musicTrackKey).join(',')}`
    if (relatedRequestRef.current === requestKey) return
    relatedRequestRef.current = requestKey
    // The session tracks request state so the queue can show pending results.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSession(previous => previous ? { ...previous, loadingRelated: true } : previous)
    const excluded = [...session.albumTracks, ...session.explicitQueue, ...session.recommendations, session.current,
      ...session.history].filter(track => track?.library_id === session.seed.library_id).map(track => track.id)
    fetchRelatedMusic(session.seed, [...new Set(excluded)], 20).then(result => {
      if (generationRef.current !== generation) return
      setSession(previous => {
        const updated = appendRelatedTracks(previous, result.tracks || [])
        const added = updated.recommendations.length - previous.recommendations.length
        return { ...updated, loadingRelated: false,
          relatedExhausted: added === 0,
          notice: added === 0 && !updated.recommendations.length ? (result.reason || 'No further related songs in this library.') : null }
      })
    }).catch(() => {
      if (generationRef.current !== generation) return
      setSession(previous => previous ? { ...previous, loadingRelated: false,
        relatedExhausted: true, notice: 'Related songs are unavailable.' } : previous)
    })
  }, [session])

  useEffect(() => {
    if (!waitingForNextRef.current || !session || session.loadingRelated) return
    // Resolve a Next request that arrived while recommendations were loading.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    advance()
  }, [session, advance])

  const value = useMemo(() => ({
    browse, setBrowse, updateBrowse, session, viewer, playing, position, duration, volume,
    setPlaying, setVolume, startAlbum, startSong, advance, previous, queueTrack, removeQueuedTrack,
    setRelatedShuffle, openAlbum, openSession, closeViewer, seek,
  }), [browse, session, viewer, playing, position, duration, volume, updateBrowse,
    startAlbum, startSong, advance, previous, queueTrack, removeQueuedTrack, setRelatedShuffle,
    openAlbum, openSession, closeViewer, seek])

  return <MusicPlayerContext.Provider value={value}>
    {children}
    <audio ref={audioRef} preload="metadata"
      onTimeUpdate={event => setPosition(event.currentTarget.currentTime)}
      onDurationChange={event => setDuration(event.currentTarget.duration || 0)}
      onEnded={advance}
      onPlaying={() => setSession(previous => previous?.playbackError ? { ...previous, playbackError: null } : previous)}
      onError={event => reportPlaybackError(sessionRef.current?.current, event.currentTarget.error)}
    />
    {viewer && <MusicLightbox />}
  </MusicPlayerContext.Provider>
}

export function useMusicPlayer() {
  const value = useContext(MusicPlayerContext)
  if (!value) throw new Error('MusicPlayerProvider is required')
  return value
}

export function PersistentMusicPlayer() {
  const { session, viewer, playing, setPlaying, advance, previous, openSession } = useMusicPlayer()
  if (!session || viewer) return null
  const track = session.current
  return <div className="music-mini-player" role="region" aria-label="Music player">
    <button className="music-mini-track" onClick={openSession} aria-label="Open music player">
      {track.artwork_url ? <img src={getMediaUrl(track.artwork_url)} alt="" /> : <span className="music-art-fallback">♫</span>}
      <span><strong>{track.title}</strong><small>{track.artist || 'Unknown artist'}</small>
        {session.playbackError && <small className="music-mini-error" role="alert">Playback error · Open player</small>}
      </span>
    </button>
    <div className="music-mini-controls">
      <button onClick={previous} aria-label="Previous track">⏮</button>
      <button onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause music' : 'Play music'}>{playing ? '⏸' : '▶'}</button>
      <button onClick={advance} aria-label="Next track">⏭</button>
    </div>
  </div>
}

export { upcomingAlbumTracks }
