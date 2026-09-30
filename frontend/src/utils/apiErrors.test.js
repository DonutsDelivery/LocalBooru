import assert from 'node:assert/strict'
import test from 'node:test'

import { createUnavailableLibraryToastGate, shouldSuppressOptionalNotFound } from './apiErrors.js'

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
