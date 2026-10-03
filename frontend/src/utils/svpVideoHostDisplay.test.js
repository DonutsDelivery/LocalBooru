import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { transformSync } from 'esbuild'

const source = fs.readFileSync(new URL('../components/Lightbox/Lightbox.jsx', import.meta.url), 'utf8')
const gridStart = source.indexOf('const showVideoLoadingGrid = ') + 'const showVideoLoadingGrid = '.length
const grid = source.slice(gridStart, source.indexOf('\n  const videoTitle', gridStart))
const statusEnd = source.indexOf('</span>}', source.indexOf('className="svp-starting-label"')) + '</span>'.length
const statusStart = source.lastIndexOf('{(', statusEnd)
const status = source.slice(statusStart + 1, statusEnd)
// Transform and render the actual production status JSX, using no app/profile.
const compiled = transformSync(`(${status})`, { loader: 'jsx', format: 'cjs' }).code

function display({ ready = true, failed = false, controls = false, issue = null } = {}) {
  const context = { React, playback: { playbackError: null }, videoFrameReadyKey: ready ? 'synthetic-media-0' : null, videoMediaKey: 'synthetic-media-0',
    localRawPlayback: false, currentQuality: 'original', localResolutionVerified: null, localResolutionReadyKey: 'synthetic-host',
    svpPathEnabled: true, svpFilterActiveRef: { current: false }, svpFailOpenRef: { current: failed },
    svpControlsReady: controls, svpConnectionIssueText: issue, svpStartupCancelPending: false }
  const overlay = vm.runInNewContext(grid, context)
  const element = vm.runInNewContext(compiled, context)
  return { overlay, status: element ? renderToStaticMarkup(element) : '' }
}

test('unavailable native host reveals ready original video and renders its error with ordinary controls', () => {
  const result = display({ failed: true, controls: true, issue: 'Native SVP video host is not registered' })
  assert.equal(result.overlay, false)
  assert.match(result.status, /role="status"/)
  assert.match(result.status, /Native SVP video host is not registered/)
})

test('healthy native startup still waits for its graph and displays startup status', () => {
  const result = display()
  assert.equal(result.overlay, true)
  assert.match(result.status, /SVP starting/)
})

test('fail-open does not reveal original video before a frame is ready', () => {
  assert.equal(display({ ready: false, failed: true, controls: true }).overlay, true)
})
