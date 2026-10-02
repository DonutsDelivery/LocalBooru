import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import { svpVideoHostOwnsEvent } from './svpVideoHost.js'

// Execute the production event callbacks with disposable video/timer objects.
// The JSX component is not evaluated and no desktop/profile API is invoked.
const source = fs.readFileSync(new URL('../components/Lightbox/Lightbox.jsx', import.meta.url), 'utf8')
const filterStart = source.indexOf('onFilterChanged: ') + 'onFilterChanged: '.length
const pausedStart = source.indexOf('onPaused: ', filterStart) + 'onPaused: '.length
const filter = source.slice(filterStart, source.indexOf(',\n      onPaused:', filterStart))
const paused = source.slice(pausedStart, source.indexOf(',\n    }).then', pausedStart))

function fixture() {
  const ref = current => ({ current })
  const owner = { hostId: 'localbooru-svp-host-new', hostEpoch: 2, hostRevision: 5 }
  let generation = 2, pauses = 0, timer
  const video = { currentSrc: 'synthetic://video', readyState: 2, currentTime: 4, paused: false, seeking: false,
    pause() { pauses++ }, play() { return Promise.resolve() } }
  const context = {
    svpVideoHostOwnsEvent, svpHostOwnerRef: ref(owner), svpPathEnabledRef: ref(true),
    activeImageKeyRef: ref('synthetic-media'), svpInteractionReadyRef: ref(true),
    svpFilterActiveRef: ref(true), mediaRef: ref(video), svpResumeRef: ref(null),
    svpTransitionRef: ref({ active: false, token: 0, timer: null }),
    setSvpStartupReady() {}, setVideoFrameReadyKey() {}, setSvpConnectionIssue() {},
    isVideoMediaElement: value => value === video, clearTimeout() {}, setTimeout(callback) { timer = callback; return 1 },
    setSvpPipelineGeneration(callback) { generation = callback(generation) },
  }
  return { context, owner, filter: vm.runInNewContext(`(${filter})`, context), paused: vm.runInNewContext(`(${paused})`, context),
    receipt: () => ({ generation, pauses }), settle: () => timer?.() }
}

// AC: @svp-platform-routing ac-linux-route
test('old same-media host, phase and document events cannot remount or pause the successor', () => {
  for (const stale of [{ hostId: 'localbooru-svp-host-old' }, { hostRevision: 4 }, { hostEpoch: 1 }]) {
    const f = fixture()
    const event = { ...f.owner, ...stale, mediaKey: 'synthetic-media' }
    f.filter({ ...event, enabled: true })
    f.paused({ ...event, paused: true })
    f.settle()
    assert.deepEqual(f.receipt(), { generation: 2, pauses: 0 })
  }
})

test('the current graph event still remounts once after settling', () => {
  const f = fixture()
  f.filter({ ...f.owner, mediaKey: 'synthetic-media', enabled: true })
  assert.equal(f.receipt().generation, 2)
  f.settle()
  assert.equal(f.receipt().generation, 3)
})

test('a graph timer cannot remount a changed host after a valid event', () => {
  const f = fixture()
  f.filter({ ...f.owner, mediaKey: 'synthetic-media', enabled: true })
  f.context.svpHostOwnerRef.current = { ...f.owner, hostRevision: 6 }
  f.settle()
  assert.equal(f.receipt().generation, 2)
})

test('current pause works and unleased non-Linux desktop events retain compatibility', () => {
  const f = fixture()
  f.paused({ ...f.owner, paused: true, mediaKey: 'synthetic-media' })
  assert.equal(f.receipt().pauses, 1)
  assert.equal(svpVideoHostOwnsEvent({ hostEpoch: 0 }, { paused: true }), true)
  assert.equal(svpVideoHostOwnsEvent({ hostEpoch: null }, { paused: true }), false)
})
