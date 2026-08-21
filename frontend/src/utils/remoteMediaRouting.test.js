import test from 'node:test'
import assert from 'node:assert/strict'
import { remoteMediaProxyUrl } from './remoteMediaRouting.js'

test('routes remote media through the embedded proxy on every Tauri platform', () => {
  const options = {
    tauri: true,
    remoteServerUrl: 'http://192.168.1.20:8790',
    localServerBase: 'http://127.0.0.1:8790',
  }
  assert.equal(
    remoteMediaProxyUrl('/api/images/7/thumbnail?directory_id=2', options),
    'http://127.0.0.1:8790/remote/api/images/7/thumbnail?directory_id=2',
  )
  assert.equal(
    remoteMediaProxyUrl('api/images/7/file?directory_id=2', options),
    'http://127.0.0.1:8790/remote/api/images/7/file?directory_id=2',
  )
})

test('does not proxy local-server or non-Tauri media paths', () => {
  assert.equal(remoteMediaProxyUrl('/api/images/7/file', {
    tauri: true,
    remoteServerUrl: null,
    localServerBase: 'http://127.0.0.1:8790',
  }), null)
  assert.equal(remoteMediaProxyUrl('/api/images/7/file', {
    tauri: false,
    remoteServerUrl: 'http://192.168.1.20:8790',
    localServerBase: '',
  }), null)
})
