import test from 'node:test'
import assert from 'node:assert/strict'

import { captureWebPlaybackHandoff, initialNativeVideoView, nativeNavigationAction, nativeViewFlags, reduceNativeVideoView, snapshotMatchesTarget } from './nativeVideoLifecycle.js'

test('ignores snapshots from stale native generations', () => {
  const current = reduceNativeVideoView(initialNativeVideoView, {
    type: 'snapshot',
    snapshot: { generation: 4, presentation: 'native_video', position: 12, item_id: 9 },
  })
  assert.equal(reduceNativeVideoView(current, {
    type: 'snapshot',
    snapshot: { generation: 3, presentation: 'web_fallback', position: 1, item_id: 8 },
  }), current)
})

test('derives preparing, visible, and preserved fallback state', () => {
  const playback = { generation: 1, position: 18.5, volume: 0.4, muted: true, speed: 1.5 }
  const preparing = reduceNativeVideoView(initialNativeVideoView, {
    type: 'snapshot',
    snapshot: { generation: 1, presentation: 'preparing_native', position: 0, item_id: 7, playback },
  })
  assert.deepEqual(nativeViewFlags(preparing), {
    useNative: true,
    preparing: true,
    visible: false,
    fallbackPosition: 0,
    playback,
  })

  const fallback = reduceNativeVideoView(preparing, {
    type: 'snapshot',
    snapshot: { generation: 1, presentation: 'web_fallback', position: 18.5, item_id: 7 },
  })
  assert.deepEqual(nativeViewFlags(fallback), {
    useNative: false,
    preparing: false,
    visible: false,
    fallbackPosition: 18.5,
    playback,
  })
})

test('runtime errors select web fallback without losing position', () => {
  const visible = {
    generation: 2,
    presentation: 'native_video',
    position: 6.25,
    itemId: 4,
    error: null,
  }
  const failed = reduceNativeVideoView(visible, { type: 'runtime-error', error: 'helper exited' })
  assert.equal(failed.presentation, 'web_fallback')
  assert.equal(failed.position, 6.25)
  assert.equal(failed.error, 'helper exited')
  assert.equal(
    reduceNativeVideoView(failed, {
      type: 'snapshot',
      snapshot: { generation: 2, presentation: 'native_video', position: 7 },
    }),
    failed,
  )
  assert.equal(
    reduceNativeVideoView(failed, { type: 'runtime-error', generation: 1 }),
    failed,
  )
})

test('target selection switches presentation ownership synchronously', () => {
  const native = reduceNativeVideoView(initialNativeVideoView, { type: 'target-native' })
  assert.equal(native.presentation, 'preparing_native')
  assert.equal(
    reduceNativeVideoView(native, { type: 'target-web', isVideo: true }).presentation,
    'web_fallback',
  )
})

test('accepts navigation only from the active native generation', () => {
  assert.equal(nativeNavigationAction({ type: 'navigate_next', generation: 7 }, 7), 1)
  assert.equal(nativeNavigationAction({ type: 'navigate_previous', generation: 7 }, 7), -1)
  assert.equal(nativeNavigationAction({ type: 'close_requested', generation: 7 }, 7), 'close')
  assert.equal(nativeNavigationAction({ type: 'navigate_next', generation: 6 }, 7), null)
})

test('filters subscription snapshots against current playback ownership', () => {
  assert.equal(snapshotMatchesTarget({ presentation: 'native_video' }, 'web-fallback'), false)
  assert.equal(snapshotMatchesTarget({ presentation: 'react_image' }, 'native-local'), false)
  assert.equal(snapshotMatchesTarget({ presentation: 'web_fallback' }, 'native-local'), true)
  assert.equal(snapshotMatchesTarget({ presentation: 'native_video' }, 'native-local'), true)
})

test('captures and bounds the canonical web-to-native handoff fields', () => {
  assert.deepEqual(captureWebPlaybackHandoff({
    currentTime: 18.25,
    paused: true,
    volume: 0.4,
    muted: true,
    playbackRate: 1.5,
  }), {
    position: 18.25,
    paused: true,
    volume: 0.4,
    muted: true,
    speed: 1.5,
  })
  assert.deepEqual(captureWebPlaybackHandoff({
    currentTime: Number.NaN,
    paused: false,
    volume: 3,
    muted: false,
    playbackRate: 9,
  }, 7), {
    position: 7,
    paused: false,
    volume: 1,
    muted: false,
    speed: 2,
  })
})
