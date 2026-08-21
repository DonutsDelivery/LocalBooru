import test from 'node:test'
import assert from 'node:assert/strict'
import { serverFromQrHandshake } from './serverManager.js'

test('QR handshake carries the stable server identity into protected storage', () => {
  const server = serverFromQrHandshake(
    { name: 'QR name', cert_fingerprint: 'sha256:test' },
    'https://booru.example',
    { success: true, token: 'device-token', serverId: 'stable-server-id', serverName: 'Server name' },
  )

  assert.equal(server.id, 'stable-server-id')
  assert.equal(server.name, 'Server name')
  assert.equal(server.url, 'https://booru.example')
  assert.equal(server.token, 'device-token')
  assert.equal(server.certFingerprint, 'sha256:test')
})

test('QR handshake fails before storage when the stable identity is absent', () => {
  assert.throws(
    () => serverFromQrHandshake({}, 'https://booru.example', { success: true, token: 'device-token' }),
    /stable identity/,
  )
})

test('QR handshake preserves the server rejection detail', () => {
  assert.throws(
    () => serverFromQrHandshake({}, 'https://booru.example', { success: false, error: 'Handshake nonce is invalid' }),
    /Handshake nonce is invalid/,
  )
})