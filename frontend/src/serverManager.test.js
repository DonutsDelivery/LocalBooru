import assert from 'node:assert/strict'
import test from 'node:test'
import { pairingUrls, serverFromQrHandshake, probeServer, discoveredAddressUpdates, serverConnectionUrls, learnServerAddresses, getServers, saveServers, updateServer } from './serverManager.js'

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

const advertisement = { server_id: 'synthetic-server', server_port: 8790,
  tailscale_url: qr.tailscale, local_url: qr.local }

test('connected discovery fills missing fallback and rotates only learned fallbacks', () => {
  const server = { id: 'legacy-random-id', url: qr.local }
  const learned = { ...server, ...discoveredAddressUpdates(server, advertisement, qr.local) }
  assert.equal(learned.fallbackUrl, qr.tailscale)
  assert.equal(learned.advertisedServerId, advertisement.server_id)
  assert.equal(discoveredAddressUpdates(learned, advertisement, qr.local), null)
  const rotated = { ...advertisement, tailscale_url: 'http://100.64.1.11:8790' }
  assert.equal(discoveredAddressUpdates(learned, rotated, qr.local).fallbackUrl, rotated.tailscale_url)
  const manual = { ...learned, fallbackUrl: 'https://private.example.test/library' }
  const update = discoveredAddressUpdates(manual, rotated, qr.local)
  assert.equal(update.fallbackUrl, undefined)
  assert.deepEqual(serverConnectionUrls({ ...manual, ...update }),
    [qr.local, manual.fallbackUrl, rotated.tailscale_url])
})

test('Tailscale primary learns a LAN fallback; legacy advertisements use the connected port', () => {
  assert.equal(discoveredAddressUpdates({ url: qr.tailscale }, advertisement, qr.tailscale).fallbackUrl, qr.local)
  const legacy = discoveredAddressUpdates({ url: qr.local }, { all_local_ips: ['192.168.1.10', '100.65.1.10'] }, 'http://192.168.1.10:18000')
  assert.equal(legacy.tailscaleUrl, 'http://100.65.1.10:18000')
  assert.deepEqual(serverConnectionUrls({ url: 'example.test/library/', fallbackUrl: 'http://example.test/library' }), ['http://example.test/library'])
})

test('discovery rejects different identities, malformed addresses, and non-Tailscale hosts', () => {
  const server = { url: qr.local, advertisedServerId: 'synthetic-server' }
  for (const data of [null, {}, { ...advertisement, server_id: 'different-server' },
    { ...advertisement, server_port: 0 }, { ...advertisement, server_port: '8790' },
    { ...advertisement, tailscale_url: 'http://100.128.1.10:8790' },
    { ...advertisement, tailscale_url: 'http://100.64.1.10:8790/other' },
    { ...advertisement, tailscale_url: 'http://user:password@100.64.1.10:8790' }]) {
    assert.equal(discoveredAddressUpdates(server, data, qr.local), null)
  }
})

test('learned addresses persist for reconnect without losing credentials or concurrent manual edits', async t => {
  const storage = new Map()
  const oldWindow = globalThis.window
  const oldStorage = globalThis.localStorage
  globalThis.window = {}
  globalThis.localStorage = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) }
  t.after(() => { globalThis.window = oldWindow; globalThis.localStorage = oldStorage })
  await saveServers([{ id: 'fixture', url: qr.local, token: 'synthetic-token', password: 'synthetic-password' }])
  await Promise.all([
    updateServer('fixture', { name: 'Edited name', fallbackUrl: 'https://manual.example.test' }),
    learnServerAddresses('fixture', advertisement, qr.local),
  ])
  const [restored] = await getServers()
  assert.equal(restored.name, 'Edited name')
  assert.equal(restored.fallbackUrl, 'https://manual.example.test')
  assert.equal(restored.tailscaleUrl, qr.tailscale)
  assert.equal(restored.token, 'synthetic-token')
  assert.equal(restored.password, 'synthetic-password')
  const fetch = t.mock.method(globalThis, 'fetch', async url => {
    if (!url.startsWith(qr.tailscale)) throw new TypeError('Offline')
    return { ok: true, status: 200 }
  })
  assert.equal((await probeServer(restored)).url, qr.tailscale)
  assert.equal(fetch.mock.callCount(), 3)
  assert.equal(await learnServerAddresses('fixture', { ...advertisement, tailscale_url: 'http://100.65.1.11:8790' }, qr.local, () => false), null)
  assert.equal((await getServers())[0].tailscaleUrl, qr.tailscale)
})
