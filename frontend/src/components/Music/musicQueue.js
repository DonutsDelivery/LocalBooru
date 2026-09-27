export const musicTrackKey = (track) => `${track.library_id || ''}:${track.id}`

export function orderedAlbumTracks(tracks) {
  return [...tracks].sort((a, b) =>
    (Number(a.disc_number) || 1) - (Number(b.disc_number) || 1) ||
    (Number(a.track_number) || 0) - (Number(b.track_number) || 0) ||
    a.id - b.id
  )
}

export function upcomingAlbumTracks(session) {
  if (session?.kind !== 'album' || session.relatedShuffle) return []
  const unavailable = new Set([
    ...session.playedKeys,
    ...session.explicitQueue.map(musicTrackKey),
  ])
  return session.albumTracks.slice(session.albumCursor + 1)
    .filter(track => !unavailable.has(musicTrackKey(track)))
}

export function nextMusicTrack(session) {
  if (!session) return null
  if (session.explicitQueue.length) {
    return { track: session.explicitQueue[0], source: 'queued' }
  }
  const albumNext = upcomingAlbumTracks(session)[0]
  if (albumNext) return { track: albumNext, source: 'album' }
  if (session.recommendations.length) {
    return { track: session.recommendations[0], source: 'related' }
  }
  return null
}

export function advanceMusicSession(session, choice) {
  if (!session || !choice) return session
  const { track, source } = choice
  const explicitQueue = source === 'queued' ? session.explicitQueue.slice(1) : session.explicitQueue
  const recommendations = source === 'related' ? session.recommendations.slice(1) : session.recommendations
  const albumCursor = source === 'album'
    ? session.albumTracks.findIndex(item => musicTrackKey(item) === musicTrackKey(track))
    : session.albumCursor
  return {
    ...session,
    current: track,
    albumCursor,
    explicitQueue,
    recommendations,
    history: [...session.history, session.current].filter(Boolean),
    playedKeys: [...session.playedKeys, musicTrackKey(track)],
    notice: null,
  }
}

export function appendRelatedTracks(session, tracks) {
  if (!session) return session
  const seen = new Set([
    ...session.playedKeys,
    ...session.explicitQueue.map(musicTrackKey),
    ...session.recommendations.map(musicTrackKey),
    ...session.albumTracks.map(musicTrackKey),
  ])
  const additions = tracks.filter(track => {
    const key = musicTrackKey(track)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
  return { ...session, recommendations: [...session.recommendations, ...additions] }
}
