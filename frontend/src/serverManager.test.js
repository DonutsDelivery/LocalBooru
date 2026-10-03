import assert from 'node:assert/strict'
import test from 'node:test'
import { pairingUrls, serverFromQrHandshake, probeServer } from './serverManager.js'

const qr = { local: 'http://192.168.1.10:8790', tailscale: 'http://100.64.1.10:8790' }
const handshake = { success: true, serverId: 'synthetic-server', token: 'synthetic-credential' }

test('pairing retains Tailscale when using LAN and LAN when using Tailscale', () => {
  const lan = serverFromQrHandshake(qr, qr.local, handshake)
  const tail = serverFromQrHandshake(qr, qr.tailscale, handshake)
  assert.equal(lan.fallbackUrl, qr.tailscale)
  assert.equal(tail.fallbackUrl, qr.local)
  assert.equal(tail.id, handshake.serverId)
  assert.equal(tail.token, handshake.token)
})

test('legacy QR public fallback remains compatible; duplicate and invalid addresses are omitted', () => {
  assert.deepEqual(pairingUrls({ ...qr, public: qr.tailscale + '/' }), [qr.local, qr.tailscale])
  assert.deepEqual(pairingUrls({ local: qr.local, tailscale: 'file:///tmp/fixture', public: null }), [qr.local])
  assert.equal(serverFromQrHandshake({ local: qr.local, public: 'https://example.test' }, qr.local, handshake).fallbackUrl, 'https://example.test')
  assert.equal(serverFromQrHandshake({ local: qr.local }, qr.local, handshake).fallbackUrl, null)
})

test('outside LAN probes the fallback with the existing pairing credential', async t => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, config) => {
    calls.push({ url, auth: config.headers.Authorization })
    if (url.startsWith(qr.local)) throw new TypeError('Unreachable LAN')
    return { ok: true, status: 200 }
  })
  const result = await probeServer({ ...serverFromQrHandshake(qr, qr.local, handshake) })
  assert.equal(result.success, true)
  assert.equal(result.url, qr.tailscale)
  assert.equal(result.usedFallback, true)
  assert.deepEqual(calls, [
    { url: qr.local + '/api', auth: 'Bearer synthetic-credential' },
    { url: qr.tailscale + '/api', auth: 'Bearer synthetic-credential' },
  ])
})

test('auth rejection does not switch addresses; both offline addresses stay offline', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => ({ ok: false, status: 401 }))
  const server = serverFromQrHandshake(qr, qr.local, handshake)
  assert.equal((await probeServer(server)).networkFailure, false)
  assert.equal(fetch.mock.callCount(), 1)
  fetch.mock.mockImplementation(async () => { throw new TypeError('Offline') })
  const result = await probeServer(server)
  assert.equal(result.success, false)
  assert.equal(result.networkFailure, true)
})
