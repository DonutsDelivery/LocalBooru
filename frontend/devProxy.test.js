import assert from 'node:assert/strict'
import test from 'node:test'
import { getDevProxyTarget } from './devProxy.js'
import config from './vite.config.js'

test('a development preview cannot default to another app backend', () => {
  for (const value of [undefined, '', '0', '80', '65536', '18790/api', '1e4', '18790.5']) {
    assert.throws(() => getDevProxyTarget(value), /explicitly select the backend/)
  }
})

test('an explicitly selected isolated backend is the proxy destination', () => {
  assert.equal(getDevProxyTarget('18790'), 'http://127.0.0.1:18790')
  assert.equal(getDevProxyTarget('18793'), 'http://127.0.0.1:18793')
})

test('Vite serving refuses an unspecified backend while production builds remain independent', () => {
  const previous = process.env.VITE_LOCALBOORU_PORT
  try {
    delete process.env.VITE_LOCALBOORU_PORT
    assert.throws(() => config({ command: 'serve' }), /explicitly select the backend/)
    assert.doesNotThrow(() => config({ command: 'build' }))
    process.env.VITE_LOCALBOORU_PORT = '18793'
    const serving = config({ command: 'serve' })
    assert.equal(serving.server.proxy['/api'].target, 'http://127.0.0.1:18793')
    assert.equal(serving.server.proxy['/thumbnails'].target, 'http://127.0.0.1:18793')
  } finally {
    if (previous === undefined) delete process.env.VITE_LOCALBOORU_PORT
    else process.env.VITE_LOCALBOORU_PORT = previous
  }
})
