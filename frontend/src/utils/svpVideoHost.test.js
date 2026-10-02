import assert from 'node:assert/strict'
import test from 'node:test'
import { createSVPVideoHostId, nextSVPVideoHostRevision, svpVideoHostUpdate, publishSVPVideoHost } from './svpVideoHost.js'

// AC: @svp-platform-routing ac-linux-route
test('physical video instances receive distinct safe registration IDs', () => {
  const first = createSVPVideoHostId()
  const second = createSVPVideoHostId()
  assert.match(first, /^localbooru-svp-host-[a-zA-Z0-9-]+$/)
  assert.notEqual(first, second)
})

test('phase changes preserve ordering and backend document epoch', () => {
  const owner = { hostId: createSVPVideoHostId(), hostEpoch: 4, hostRevision: nextSVPVideoHostRevision() }
  const disable = { ...owner, hostRevision: nextSVPVideoHostRevision() }
  assert.ok(disable.hostRevision > owner.hostRevision)
  const stale = svpVideoHostUpdate(owner, true, { width: 1920, hostEpoch: 999, enabled: false })
  assert.equal(stale.hostEpoch, 4)
  assert.equal(stale.enabled, true)
  assert.deepEqual(svpVideoHostUpdate(disable, false), { ...disable, enabled: false })
})

const missingHost = new Error('Native SVP video host is not registered; matching WebKit runtime is required')

test('a transient registration race receives bounded startup grace', async () => {
  let calls = 0, waits = 0
  const result = await publishSVPVideoHost(async () => { if (++calls < 3) throw missingHost }, {}, () => true, async delay => { assert.equal(delay, 250); waits++ })
  assert.equal(result, true)
  assert.equal(calls, 3)
  assert.equal(waits, 2)
})

test('permanently missing runtime reports its detail after five attempts', async () => {
  let calls = 0
  await assert.rejects(publishSVPVideoHost(async () => { calls++; throw missingHost }, {}, () => true, async () => {}), missingHost)
  assert.equal(calls, 5)
})

// AC: @local-decoded-video-resolution ac-geometry
test('missing pre-graph geometry receives bounded grace before any Manager activation', async () => {
  let calls = 0
  const notReady = new Error('Native video geometry is not ready; matching scaler runtime is required')
  assert.equal(await publishSVPVideoHost(async () => { if (++calls < 3) throw notReady }, {}, () => true, async () => {}), true)
  assert.equal(calls, 3)
})

test('ownership change cancels retries and unrelated errors are not retried', async () => {
  let current = true, calls = 0
  assert.equal(await publishSVPVideoHost(async () => { calls++; throw missingHost }, {}, () => current, async () => { current = false }), false)
  assert.equal(calls, 1)
  const unrelated = new Error('synthetic filesystem failure')
  await assert.rejects(publishSVPVideoHost(async () => { throw unrelated }, {}, () => true, async () => { assert.fail('must not retry unrelated failure') }), unrelated)
})

// AC: @local-decoded-video-resolution ac-ownership
test('a successful deferred acknowledgement cannot complete a stale owner', async () => {
  let current = true, release
  const result = publishSVPVideoHost(() => new Promise(resolve => { release = resolve }), {}, () => current)
  current = false
  release()
  assert.equal(await result, false)
})
