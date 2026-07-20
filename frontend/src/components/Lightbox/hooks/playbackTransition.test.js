import test from 'node:test'
import assert from 'node:assert/strict'

import {
  capturePlaybackIntent,
  createPlaybackTransitionOwner,
  createVideoBackendCleanupTracker,
  nextPlaybackSourceRevision,
} from './playbackTransition.js'

const video = ({ paused = false, ended = false } = {}) => ({ paused, ended })

test('only the newest playback transition owns completion', () => {
  // AC: @reliable-stream-transitions ac-final-source-owner
  // AC: @svp-single-player ac-final-transition-owner
  const media = video()
  const owner = createPlaybackTransitionOwner()
  const initial = capturePlaybackIntent(media, 73, 'image-1')
  const first = owner.begin(initial)
  const final = owner.begin(capturePlaybackIntent(media, 0, 'image-1'))

  assert.equal(first.signal.aborted, true)
  assert.equal(owner.isCurrent(first), false)
  assert.equal(owner.isCurrent(final), true)
  assert.equal(final.intent.position, 73)
})

test('transition generations preserve request order across player owners', () => {
  // AC: @reliable-stream-transitions ac-stop-superseded-producer
  const media = video()
  const firstOwner = createPlaybackTransitionOwner()
  const secondOwner = createPlaybackTransitionOwner()
  const first = firstOwner.begin(capturePlaybackIntent(media, 10, 'image-1'))
  const second = secondOwner.begin(capturePlaybackIntent(media, 20, 'image-2'))

  assert.ok(second.generation > first.generation)
  assert.ok(firstOwner.invalidate() > second.generation)
})

test('a transition preserves playing and paused intent at the absolute position', () => {
  // AC: @reliable-stream-transitions ac-preserve-playback-intent
  // AC: @svp-single-player ac-preserve-playback-intent
  const playing = capturePlaybackIntent(video(), 73, 'playing')
  const paused = capturePlaybackIntent(video({ paused: true }), 41, 'paused')

  assert.deepEqual(
    { position: playing.position, shouldPlay: playing.shouldPlay, ended: playing.ended },
    { position: 73, shouldPlay: true, ended: false },
  )
  assert.deepEqual(
    { position: paused.position, shouldPlay: paused.shouldPlay, ended: paused.ended },
    { position: 41, shouldPlay: false, ended: false },
  )
})

test('ended playback never becomes an autoplay intent', () => {
  // AC: @svp-single-player ac-ended-transition
  const intent = capturePlaybackIntent(video({ ended: true }), 120, 'ended')
  assert.equal(intent.ended, true)
  assert.equal(intent.shouldPlay, false)
})

test('invalidating a transition aborts producer startup and image navigation ownership', () => {
  // AC: @reliable-stream-transitions ac-stop-superseded-producer
  // AC: @svp-single-player ac-disable-during-start
  // AC: @svp-single-player ac-image-transition-owner
  const media = video()
  const transitions = createPlaybackTransitionOwner()
  const starting = transitions.begin(capturePlaybackIntent(media, 73, 'image-1'))

  transitions.invalidate()

  assert.equal(starting.signal.aborted, true)
  assert.equal(transitions.isCurrent(starting), false)
})

test('ordinary images never request video backend cleanup', () => {
  // AC: @ordinary-lightbox-media-rendering ac-3
  const cleanup = createVideoBackendCleanupTracker()

  assert.equal(cleanup.replace(false), false)
  assert.equal(cleanup.disable(false), false)
  assert.equal(cleanup.unmount(false), false)
})

test('video backend cleanup is claimed once when playback is replaced or disabled', () => {
  // AC: @ordinary-lightbox-media-rendering ac-3
  // AC: @svp-single-player ac-idempotent-stop
  const replaced = createVideoBackendCleanupTracker()
  assert.equal(replaced.replace(true), true)
  assert.equal(replaced.disable(false), false)
  assert.equal(replaced.unmount(false), false)

  const disabled = createVideoBackendCleanupTracker()
  assert.equal(disabled.disable(true), true)
  assert.equal(disabled.unmount(false), false)
})

test('forced replacement can revise an unchanged playback address', () => {
  // AC: @reliable-stream-transitions ac-reload-restarted-source
  const url = '/api/settings/transcode/stream/current/playlist.m3u8'
  const previous = { url, revision: 4 }
  const replacement = { url, revision: nextPlaybackSourceRevision(previous.revision) }

  assert.equal(replacement.url, previous.url)
  assert.notEqual(replacement.revision, previous.revision)
})
