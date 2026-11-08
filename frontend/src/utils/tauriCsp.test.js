import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const config = JSON.parse(readFileSync(new URL('../../../src-tauri/tauri.conf.json', import.meta.url), 'utf8'))
const directives = Object.fromEntries(
  config.app.security.csp.split(';').map(part => part.trim().split(/\s+/)).filter(p => p[0]).map(([name, ...sources]) => [name, sources]),
)

test('hls.js can start its blob transmux worker instead of falling back to the UI thread', () => {
  // hls.js builds its worker from a blob: URL. Without worker-src the browser
  // falls back to script-src 'self', blocks it, and every HLS segment is
  // demuxed on the main thread of the (Android) WebView.
  assert.ok(directives['worker-src'], 'worker-src must be declared')
  assert.ok(directives['worker-src'].includes('blob:'))
})

test('workers do not widen script loading', () => {
  assert.deepEqual(directives['script-src'], ["'self'"])
})
