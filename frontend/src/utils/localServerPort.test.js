import assert from 'node:assert/strict'
import test from 'node:test'
import { getLocalServerPort } from './localServerPort.js'

test('native process port overrides the baked build port', () => {
  assert.equal(getLocalServerPort({ __TAURI_INTERNALS__: {}, __LOCALBOORU_PORT__: 18790 }, '8790'), '18790')
  assert.equal(getLocalServerPort({ __TAURI_INTERNALS__: {}, __LOCALBOORU_PORT__: '18790' }, '5211'), '18790')
})

test('browser and SSR ignore a native port property and preserve build/default behavior', () => {
  assert.equal(getLocalServerPort({ __LOCALBOORU_PORT__: 18790 }, '19000'), '19000')
  assert.equal(getLocalServerPort({}, undefined), '8790')
  assert.equal(getLocalServerPort(undefined, '18790'), '18790')
})

test('port selection rejects malformed, fractional, and out-of-range values', () => {
  for (const invalid of [0, -1, 65536, 8790.5, NaN, Infinity, null, {}, '18790/api', '127.0.0.1:18790', '1e4', '']) {
    assert.throws(() => getLocalServerPort({ __TAURI_INTERNALS__: {}, __LOCALBOORU_PORT__: invalid }, '19000'), /refusing to connect/)
    assert.equal(getLocalServerPort({}, invalid), '8790')
  }
  assert.equal(getLocalServerPort({ __TAURI_INTERNALS__: {}, __LOCALBOORU_PORT__: 65535 }), '65535')
})

test('missing native runtime identity fails closed before Tauri globals initialize', () => {
  assert.throws(() => getLocalServerPort({ __TAURI_INTERNALS__: {} }, '8790'), /refusing to connect/)
  assert.throws(() => getLocalServerPort({ location: { protocol: 'tauri:' } }), /refusing to connect/)
  assert.throws(() => getLocalServerPort({ location: { hostname: 'tauri.localhost' } }), /refusing to connect/)
})
