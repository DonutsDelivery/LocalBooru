import assert from 'node:assert/strict'
import test from 'node:test'

import { shouldSuppressOptionalNotFound } from './apiErrors.js'

// AC: @identity-safe-timeline-previews ac-optional-failure
test('only request-scoped optional 404 responses suppress error toasts', () => {
  assert.equal(shouldSuppressOptionalNotFound({ suppressErrorToast: true }, 404), true)
  assert.equal(shouldSuppressOptionalNotFound({}, 404), false)
  assert.equal(shouldSuppressOptionalNotFound({ suppressErrorToast: true }, 500), false)
})
