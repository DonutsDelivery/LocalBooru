import assert from 'node:assert/strict'
import test from 'node:test'

import { createUnavailableLibraryToastGate, createRemoteConnectionToastGate, isRemoteConnectionFailure, shouldSuppressOptionalNotFound } from './apiErrors.js'

// AC: @identity-safe-timeline-previews ac-optional-failure
test('only request-scoped optional 404 responses suppress error toasts', () => {
  assert.equal(shouldSuppressOptionalNotFound({ suppressErrorToast: true }, 404), true)
  assert.equal(shouldSuppressOptionalNotFound({}, 404), false)
  assert.equal(shouldSuppressOptionalNotFound({ suppressErrorToast: true }, 500), false)
})

test('repeated unavailable-library reads notify once per library within the cooldown', () => {
  const suppress = createUnavailableLibraryToastGate()
  const get = { method: 'get', url: '/images?page=1' }
  const offline = "Library 'lib-a' not found or not mounted"
  assert.equal(suppress(get, 404, offline, 1000), false)
  assert.equal(suppress({ ...get, url: '/images?page=2' }, 404, offline, 1001), true)
  assert.equal(suppress(get, 404, "Library 'lib-b' not found or not mounted", 1001), false)
  assert.equal(suppress(get, 404, offline, 601001), false)
})

test('unrelated errors and explicit mutations are never throttled', () => {
  const suppress = createUnavailableLibraryToastGate()
  const offline = "Library 'lib-a' not found or not mounted"
  assert.equal(suppress({ method: 'post' }, 404, offline, 1000), false)
  assert.equal(suppress({ method: 'get' }, 500, offline, 1000), false)
  assert.equal(suppress({ method: 'get' }, 404, 'Image not found', 1000), false)
  assert.equal(suppress({ method: 'get' }, 404, offline, 1000), false)
})

test('one remote outage covers repeated stats and gallery errors', () => {
  const gate = createRemoteConnectionToastGate()
  const failure = 'Proxy error: error sending request for url (http://example.invalid/api/images)'
  assert.equal(gate({ method: 'get', url: '/library/stats' }, 502, failure, 1000), false)
  assert.equal(gate({ method: 'get', url: '/images' }, 502, failure, 1001), true)
  assert.equal(gate({ method: 'get' }, 502, failure, 601001), false)
})

test('both failed addresses count as one outage; real server and mutation errors stay visible', () => {
  const gate = createRemoteConnectionToastGate()
  const failure = 'Proxy error (primary + fallback): first / second'
  assert.equal(isRemoteConnectionFailure(502, { detail: failure }), true)
  assert.equal(gate({ method: 'get' }, 502, failure, 1000), false)
  assert.equal(gate({ method: 'post' }, 502, failure, 1001), false)
  assert.equal(gate({ method: 'get' }, 502, 'Bad Gateway', 1001), false)
  assert.equal(gate({ method: 'get' }, 500, failure, 1001), false)
  assert.equal(gate({ method: 'get' }, 401, failure, 1001), false)
  assert.equal(gate({ method: 'get' }, 502, failure, 1001), true)
  assert.equal(createRemoteConnectionToastGate()({ method: 'get' }, 502, failure, 1001), false)
})
