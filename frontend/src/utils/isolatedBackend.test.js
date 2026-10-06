import assert from 'node:assert/strict'
import test from 'node:test'
import { getIsolatedInstanceId, verifyIsolatedHealth } from './isolatedBackend.js'

const own = 'a'.repeat(64)
test('isolated native and browser launches select their respective expected identity', () => {
  assert.equal(getIsolatedInstanceId({ __TAURI_INTERNALS__: {}, __LOCALBOORU_ISOLATED_INSTANCE_ID__: own }, 'b'.repeat(64)), own)
  assert.equal(getIsolatedInstanceId({}, own), own)
  assert.equal(getIsolatedInstanceId({}, undefined), null)
  assert.throws(() => getIsolatedInstanceId({}, 'invalid'), /Invalid isolated backend identity/)
})

test('a successful foreign health response cannot authorize test library access', () => {
  for (const response of [null, {}, { status: 'healthy' }, { status: 'healthy', isolated_instance_id: 'b'.repeat(64) }, { status: 'starting', isolated_instance_id: own }]) {
    assert.throws(() => verifyIsolatedHealth(response, own), /refusing access/)
  }
  assert.deepEqual(verifyIsolatedHealth({ status: 'healthy', isolated_instance_id: own }, own), { status: 'healthy', isolated_instance_id: own })
})
