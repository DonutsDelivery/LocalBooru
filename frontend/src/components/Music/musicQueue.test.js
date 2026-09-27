import test from 'node:test'
import assert from 'node:assert/strict'

import {
  advanceMusicSession, appendRelatedTracks, nextMusicTrack, orderedAlbumTracks,
} from './musicQueue.js'

const track = (id, artist = 'Artist A', disc_number = 1, track_number = id) => ({
  id, artist, disc_number, track_number, library_id: 'library-a', title: `Track ${id}`,
})

function albumSession(tracks, startIndex = 0) {
  return {
    kind: 'album', albumTracks: tracks, albumCursor: startIndex,
    current: tracks[startIndex], seed: tracks[startIndex], relatedShuffle: false,
    explicitQueue: [], recommendations: [], history: [],
    playedKeys: [`library-a:${tracks[startIndex].id}`],
  }
}

test('album order follows disc and track numbers, then continues into related songs', () => {
  const tracks = orderedAlbumTracks([
    track(6, 'Artist A', 2, 2), track(4, 'Artist A', 1, 4),
    track(5, 'Artist A', 2, 1), track(3, 'Artist A', 1, 3),
  ])
  assert.deepEqual(tracks.map(item => item.id), [3, 4, 5, 6])
  let session = albumSession(tracks, 1)
  session.recommendations = [track(90, 'Artist B')]
  const played = [session.current.id]
  for (let i = 0; i < 3; i++) {
    session = advanceMusicSession(session, nextMusicTrack(session))
    played.push(session.current.id)
  }
  assert.deepEqual(played, [4, 5, 6, 90])
  assert.equal(session.seed.id, 4)
})

test('related shuffle changes the next choice without replacing the current track or seed', () => {
  const session = albumSession([track(4), track(5), track(6)])
  session.recommendations = [track(90, 'Artist B')]
  assert.equal(nextMusicTrack(session).track.id, 5)
  const shuffled = { ...session, relatedShuffle: true }
  assert.equal(shuffled.current.id, 4)
  assert.equal(shuffled.seed.id, 4)
  assert.equal(nextMusicTrack(shuffled).track.id, 90)
  assert.equal(nextMusicTrack({ ...shuffled, relatedShuffle: false }).track.id, 5)
})

test('explicit queue takes precedence and recommendations do not repeat played or queued tracks', () => {
  let session = albumSession([track(4), track(5)])
  session.explicitQueue = [track(80, 'Artist C')]
  session.recommendations = [track(90, 'Artist B')]
  assert.equal(nextMusicTrack(session).track.id, 80)
  session = advanceMusicSession(session, nextMusicTrack(session))
  assert.equal(session.current.id, 80)
  assert.equal(session.seed.id, 4)
  assert.equal(nextMusicTrack(session).track.id, 5)
  session = appendRelatedTracks(session, [track(4), track(80), track(90), track(91, 'Artist D')])
  assert.deepEqual(session.recommendations.map(item => item.id), [90, 91])
})

test('an explicitly queued album track plays once and remaining album tracks stay in order', () => {
  const tracks = [track(4), track(5), track(6), track(7), track(8)]
  const playFromQueue = queued => {
    let session = albumSession(tracks)
    session.explicitQueue = queued
    const played = [session.current.id]
    while (nextMusicTrack(session)) {
      session = advanceMusicSession(session, nextMusicTrack(session))
      played.push(session.current.id)
    }
    return played
  }
  assert.deepEqual(playFromQueue([tracks[1]]), [4, 5, 6, 7, 8])
  assert.deepEqual(playFromQueue([tracks[3], tracks[1]]), [4, 7, 5, 6, 8])
})

test('song mix keeps its selected seed while advancing through recommendations', () => {
  const selected = track(1)
  let session = {
    kind: 'mix', albumTracks: [], albumCursor: -1, current: selected, seed: selected,
    relatedShuffle: true, explicitQueue: [], recommendations: [track(2), track(3)],
    history: [], playedKeys: ['library-a:1'],
  }
  session = advanceMusicSession(session, nextMusicTrack(session))
  session = advanceMusicSession(session, nextMusicTrack(session))
  assert.equal(session.current.id, 3)
  assert.equal(session.seed.id, 1)
  assert.deepEqual(session.history.map(item => item.id), [1, 2])
})
