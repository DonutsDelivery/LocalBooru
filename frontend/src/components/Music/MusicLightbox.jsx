import { useEffect } from 'react'
import { getMediaUrl } from '../../api'
import { useMusicPlayer, upcomingAlbumTracks } from './MusicPlayer'

const formatTime = seconds => {
  if (!Number.isFinite(seconds)) return '0:00'
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

function Artwork({ url, label, className = '' }) {
  return url
    ? <img className={className} src={getMediaUrl(url)} alt={label || 'Album artwork'} />
    : <div className={`music-art-fallback ${className}`} role="img" aria-label="No artwork">♫</div>
}

function TrackRow({ track, index, current, onPlay, onQueue, onRemove }) {
  return <div className={`music-track-row ${current ? 'current' : ''}`}>
    <button onClick={onPlay} className="music-track-main" aria-label={`Play ${track.title}`}>
      <span className="music-track-index">{Number(track.disc_number) > 1 ? `${track.disc_number}.${track.track_number || index + 1}` : track.track_number || index + 1}</span>
      <span><strong>{track.title || 'Untitled track'}</strong><small>{track.artist || 'Unknown artist'}</small></span>
      <span className="music-track-duration">{formatTime(track.duration)}</span>
    </button>
    {onQueue && <button className="music-queue-action" onClick={onQueue} aria-label={`Queue ${track.title}`} title="Play next">＋</button>}
    {onRemove && <button className="music-queue-remove" onClick={onRemove} aria-label={`Remove ${track.title} from queue`} title="Remove from queue">✕</button>}
  </div>
}

export default function MusicLightbox() {
  const {
    session, viewer, playing, position, duration, volume, setPlaying, setVolume,
    startAlbum, startSong, queueTrack, removeQueuedTrack, advance, previous, setRelatedShuffle,
    openSession, closeViewer, seek,
  } = useMusicPlayer()
  const isPreview = viewer?.kind === 'album'
  const album = isPreview ? viewer.album : session?.album
  const albumTitle = album?.display_title || album?.title
  const tracks = isPreview ? viewer.tracks : session?.albumTracks || []
  const current = session?.current
  const upcomingAlbum = upcomingAlbumTracks(session)
  const upcoming = [...(session?.explicitQueue || []), ...upcomingAlbum, ...(session?.recommendations || [])]

  useEffect(() => {
    const onKey = event => {
      const target = event.target
      if (target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (event.key === 'Escape') { closeViewer(); return }
      if (!session) return
      if (event.key === ' ') { event.preventDefault(); setPlaying(!playing) }
      if (event.key === 'ArrowLeft') { event.preventDefault(); seek(position - 5) }
      if (event.key === 'ArrowRight') { event.preventDefault(); seek(position + 5) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [closeViewer, playing, position, seek, session, setPlaying])

  return <div className="music-lightbox" role="dialog" aria-modal="true" aria-label="Music player"
    onClick={event => { if (event.target === event.currentTarget) closeViewer() }}>
    <div className="music-lightbox-shell">
      <header className="music-lightbox-header">
        <div><span className="music-eyebrow">Music</span><h2>{isPreview ? albumTitle || 'Album' : session?.kind === 'mix' ? `Mix based on ${session.seed.title}` : albumTitle || 'Now playing'}</h2></div>
        <div className="music-lightbox-header-actions">
          {isPreview && session && <button onClick={openSession}>Now playing</button>}
          <button onClick={closeViewer} aria-label="Close music player">✕</button>
        </div>
      </header>
      <div className="music-lightbox-body">
        <section className="music-now-playing">
          <Artwork url={isPreview ? album?.artwork_url : current?.artwork_url} label={isPreview ? albumTitle : current?.title} className="music-large-art" />
          <div className="music-now-playing-meta">
            <span className="music-eyebrow">{isPreview ? 'Album' : session?.kind === 'mix' ? 'Related songs' : 'Album playback'}</span>
            <h3>{isPreview ? albumTitle : current?.title}</h3>
            <p>{isPreview ? album?.artist : current?.artist || 'Unknown artist'}</p>
            {isPreview && <button className="music-primary" onClick={() => startAlbum(album, tracks)}>▶ Play album</button>}
          </div>
          {!isPreview && session && <div className="music-playback-controls">
            <div className="music-seek-row"><span>{formatTime(position)}</span><input type="range" min="0" max={Math.max(duration, 1)} step="0.1" value={Math.min(position, Math.max(duration, 1))} onChange={event => seek(Number(event.target.value))} aria-label="Seek music" /><span>{formatTime(duration)}</span></div>
            <div className="music-transport">
              <button onClick={previous} aria-label="Previous track">⏮</button>
              <button className="music-play-button" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause music' : 'Play music'}>{playing ? '⏸' : '▶'}</button>
              <button onClick={advance} aria-label="Next track">⏭</button>
            </div>
            <label className="music-volume">Volume <input type="range" min="0" max="1" step="0.01" value={volume} onChange={event => setVolume(Number(event.target.value))} /></label>
            {session.kind === 'album' && <div className="music-shuffle-switch">
              <label><input type="checkbox" checked={session.relatedShuffle} onChange={event => setRelatedShuffle(event.target.checked)} /> Shuffle related songs</label>
              <small>{session.relatedShuffle ? 'Related songs after this track' : 'Album order · Related songs afterward'}</small>
            </div>}
            {session.kind === 'mix' && <p className="music-mix-caption">Mix based on {session.seed.title}</p>}
            {session.playbackError && <p className="music-playback-error" role="alert">{session.playbackError}</p>}
          </div>}
        </section>
        <section className="music-list-panel">
          {tracks.length > 0 && <div className="music-list-section">
            <h3>Album tracks</h3>
            {tracks.map((track, index) => <TrackRow key={`${track.library_id}:${track.id}`} track={track} index={index}
              current={!isPreview && current?.id === track.id && current?.library_id === track.library_id}
              onPlay={() => startAlbum(album, tracks, track)}
              onQueue={session ? () => queueTrack(track) : null} />)}
          </div>}
          {!isPreview && <div className="music-list-section">
            <h3>Up next</h3>
            {upcoming.length ? upcoming.map((track, index) => <TrackRow key={`${track.library_id}:${track.id}:${index}`} track={track} index={index}
              onPlay={() => startSong(track)} onQueue={null}
              onRemove={index < session.explicitQueue.length ? () => removeQueuedTrack(index) : null} />) : <p className="music-queue-empty">{session?.loadingRelated ? 'Finding related songs…' : session?.notice || 'Finding what plays next…'}</p>}
            {session?.notice && upcoming.length > 0 && <p className="music-queue-note">{session.notice}</p>}
          </div>}
        </section>
      </div>
    </div>
  </div>
}
