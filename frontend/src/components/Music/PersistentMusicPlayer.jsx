import { getMediaUrl } from '../../api'
import { useMusicPlayer } from './MusicPlayer'

const formatTime = seconds => {
  if (!Number.isFinite(seconds)) return '0:00'
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

export default function PersistentMusicPlayer({ placement = 'sidebar' }) {
  const { session, viewer, playing, position, duration, volume, setPlaying, setVolume, seek, advance, previous, openSession } = useMusicPlayer()
  if (!session || viewer) return null
  const track = session.current
  const artwork = track.artwork_url || session.album?.artwork_url

  return <div className={`music-mini-player music-mini-player--${placement}`} role="region" aria-label="Music player">
    <button className="music-mini-artwork" onClick={openSession} aria-label="Open music player">
      {artwork ? <img src={getMediaUrl(artwork)} alt="" /> : <span className="music-art-fallback">♫</span>}
    </button>
    <button className="music-mini-track" onClick={openSession} aria-label="Open music player">
      <strong>{track.title || 'Untitled track'}</strong>
      <small>{track.artist || 'Unknown artist'}</small>
      {session.playbackError && <small className="music-mini-error" role="alert">Playback error · Open player</small>}
    </button>
    <div className="music-mini-progress">
      <input type="range" min="0" max={Math.max(duration, 1)} step="0.1"
        value={Math.min(position, Math.max(duration, 1))} onChange={event => seek(Number(event.target.value))}
        aria-label="Seek music" disabled={!duration} />
      <div className="music-mini-times"><span>{formatTime(position)}</span><span>{formatTime(duration)}</span></div>
    </div>
    <div className="music-mini-controls">
      <button onClick={previous} aria-label="Previous track">⏮</button>
      <button onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause music' : 'Play music'}>{playing ? '⏸' : '▶'}</button>
      <button onClick={advance} aria-label="Next track">⏭</button>
    </div>
    <label className="music-mini-volume">
      <span>Volume</span>
      <input type="range" min="0" max="1" step="0.01" value={volume}
        onChange={event => setVolume(Number(event.target.value))} aria-label="Music volume" />
    </label>
  </div>
}
