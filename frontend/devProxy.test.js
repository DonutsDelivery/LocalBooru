import assert from 'node:assert/strict'
import test from 'node:test'
import { getDevProxyTarget } from './devProxy.js'
import config from './vite.config.js'
import { createServer as createHttpServer } from 'node:http'
import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createServer as createViteServer } from 'vite'

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
    assert.equal(serving.server.proxy['/health'].target, 'http://127.0.0.1:18793')
    assert.equal(serving.server.proxy['/api'].target, 'http://127.0.0.1:18793')
    assert.equal(serving.server.proxy['/thumbnails'].target, 'http://127.0.0.1:18793')
  } finally {
    if (previous === undefined) delete process.env.VITE_LOCALBOORU_PORT
    else process.env.VITE_LOCALBOORU_PORT = previous
  }
})

test('a development preview forwards identity health checks to its explicitly selected synthetic backend', async () => {
  const root = await mkdtemp(join(tmpdir(), 'dmc-proxy-identity-'))
  const previous = process.env.VITE_LOCALBOORU_PORT
  const requests = []
  const identity = 'a'.repeat(64)
  const backend = createHttpServer((req, res) => {
    requests.push(req.url)
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ status: 'healthy', isolated_instance_id: identity }))
  })
  let preview
  try {
    await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve))
    process.env.VITE_LOCALBOORU_PORT = String(backend.address().port)
    await writeFile(join(root, 'index.html'), '<p>Synthetic frontend fallback</p>')
    const serving = config({ command: 'serve' })
    preview = await createViteServer({ configFile: false, root, logLevel: 'silent', server: { host: '127.0.0.1', port: 0, proxy: serving.server.proxy }, plugins: [] })
    await preview.listen()
    const response = await fetch(`http://127.0.0.1:${preview.httpServer.address().port}/health`)
    assert.equal(response.status, 200)
    assert.deepEqual(await response.json(), { status: 'healthy', isolated_instance_id: identity })
    assert.deepEqual(requests, ['/health'])
  } finally {
    await preview?.close()
    await new Promise(resolve => backend.close(resolve))
    if (previous === undefined) delete process.env.VITE_LOCALBOORU_PORT
    else process.env.VITE_LOCALBOORU_PORT = previous
    await rm(root, { recursive: true, force: true })
  }
})
