import test from 'node:test'
import assert from 'node:assert/strict'
import { webcrypto } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { formatPairingFingerprint, isEligiblePairingServerUrl, isLoopbackPairingUrl, localPairingApiBaseForPort, pairingServerEndpoints, selectDesktopCallbackUrls, validateDesktopPairingRequest } from './devicePairing.js'

globalThis.crypto ||= webcrypto
globalThis.btoa ||= value => Buffer.from(value, 'binary').toString('base64')
globalThis.atob ||= value => Buffer.from(value, 'base64').toString('binary')

function bytesToBase64(bytes) {
  return Buffer.from(bytes).toString('base64')
}

function bytesToBase64Url(bytes) {
  return Buffer.from(bytes).toString('base64url')
}

async function request(overrides = {}) {
  const [signingKeys, encryptionKeys, callbackKeys] = await Promise.all([
    crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify']),
    crypto.subtle.generateKey({ name: 'RSA-OAEP', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' }, true, ['encrypt', 'decrypt']),
    crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify']),
  ])
  const signingPublicKeySpki = bytesToBase64(await crypto.subtle.exportKey('spki', signingKeys.publicKey))
  const encryptionPublicKeySpki = bytesToBase64(await crypto.subtle.exportKey('spki', encryptionKeys.publicKey))
  const callbackPublicKeySpki = bytesToBase64(await crypto.subtle.exportKey('spki', callbackKeys.publicKey))
  const fingerprintInput = new TextEncoder().encode(`${signingPublicKeySpki}\0${encryptionPublicKeySpki}`)
  return {
    type: 'localbooru-desktop-pairing',
    version: 1,
    sessionId: 'session',
    sessionSecret: 'secret',
    displayName: 'Bedroom Desktop',
    signingPublicKeySpki,
    encryptionPublicKeySpki,
    publicKeyFingerprint: bytesToBase64Url(await crypto.subtle.digest('SHA-256', fingerprintInput)),
    callbackPublicKeySpki,
    callbackUrls: ['http://192.168.1.2:8790'],
    expiresAt: Math.floor(Date.now() / 1000) + 60,
    requestedScope: 'local_network_write',
    ...overrides,
  }
}

test('accepts live local-network desktop authorization requests', async () => {
  assert.equal(await validateDesktopPairingRequest(await request()), true)
  assert.equal(await validateDesktopPairingRequest(await request({ callbackUrls: ['http://100.80.1.2:8790'] })), true)
})

test('rejects expired, forged, or public callback desktop requests', async () => {
  assert.equal(await validateDesktopPairingRequest(await request({ expiresAt: 1 })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ callbackUrls: ['https://attacker.example'] })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ callbackUrls: ['http://desktop.local:8790'] })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ callbackUrls: ['http://192.168.1.2@attacker.example'] })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ callbackUrls: ['http://192.168.1.2:8790/internal'] })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ callbackUrls: ['http://192.168.1.2:8790?target=internal'] })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ requestedScope: 'admin' })), false)
  assert.equal(await validateDesktopPairingRequest(await request({ publicKeyFingerprint: 'forged' })), false)
})

test('formats fingerprints for cross-device comparison', () => {
  assert.equal(formatPairingFingerprint('abcdefghijklmnopqrstuvwx'), 'abcd efgh ijkl mnop qrst uvwx')
})

test('uses the embedded server runtime port for desktop pairing', () => {
  assert.equal(localPairingApiBaseForPort(18790), 'http://127.0.0.1:18790/api/device-pairing')
  assert.throws(() => localPairingApiBaseForPort(80), /invalid port/)
  assert.equal(isLoopbackPairingUrl('http://127.0.0.1:18790/api/device-pairing'), true)
  assert.equal(isLoopbackPairingUrl('http://localhost:8790/api/device-pairing'), true)
  assert.equal(isLoopbackPairingUrl('https://192.168.1.2:8790/api/device-pairing'), false)
})

test('native HTTP capability admits configured LAN and Tailscale ports', () => {
  const capabilities = JSON.parse(readFileSync(new URL('../../src-tauri/capabilities/default.json', import.meta.url)))
  const http = capabilities.permissions.find(permission => permission.identifier === 'http:default')
  assert.deepEqual(http.allow, [
    { url: 'http://*:*' },
    { url: 'https://*:*' },
  ])
})

test('does not replace a real desktop callback failure with the phone loopback result', () => {
  assert.deepEqual(
    selectDesktopCallbackUrls(['http://192.168.1.2:8790/', 'http://127.0.0.1:8790']),
    ['http://192.168.1.2:8790'],
  )
  assert.deepEqual(selectDesktopCallbackUrls(['http://127.0.0.1:8790/']), ['http://127.0.0.1:8790'])
})

test('issues pairing grants through the live endpoint while retaining the alternate route', () => {
  const server = {
    url: 'http://192.168.1.20:8790',
    fallbackUrl: 'http://100.80.1.2:8790',
  }
  assert.deepEqual(pairingServerEndpoints(server, server.url), {
    baseUrl: server.url,
    fallbackUrl: server.fallbackUrl,
  })
  assert.deepEqual(pairingServerEndpoints(server, server.fallbackUrl), {
    baseUrl: server.fallbackUrl,
    fallbackUrl: server.url,
  })
  assert.deepEqual(pairingServerEndpoints({ ...server, fallbackUrl: 'http://public.example' }, server.url), {
    baseUrl: server.url,
    fallbackUrl: null,
  })
})

test('allows authenticated private-LAN servers without allowing public HTTP', () => {
  assert.equal(isEligiblePairingServerUrl('https://booru.example'), true)
  assert.equal(isEligiblePairingServerUrl('http://192.168.1.20:8790'), true)
  assert.equal(isEligiblePairingServerUrl('http://10.0.0.4:8790'), true)
  assert.equal(isEligiblePairingServerUrl('http://100.80.1.2:8790'), true)
  assert.equal(isEligiblePairingServerUrl('http://203.0.113.10:8790'), false)
  assert.equal(isEligiblePairingServerUrl('http://booru.example:8790'), false)
  assert.equal(isEligiblePairingServerUrl('http://192.168.1.20:8790/private'), false)
})
