import assert from 'node:assert/strict'
import test from 'node:test'
import { svpClientTransition, svpPlaybackError } from './svpPlayback.js'

test('SVP starts and cleanup share a client identity and preserve their order', () => {
  // AC: @svp-single-player ac-final-transition-owner
  const start = svpClientTransition(100)
  const stop = svpClientTransition(101)
  assert.ok(start.client_session_id)
  assert.equal(start.client_session_id, stop.client_session_id)
  assert.equal(start.client_transition_id, 100)
  assert.equal(stop.client_transition_id, 101)
})

test('independent loaded clients do not share a session identity', async () => {
  // AC: @svp-single-player ac-idempotent-stop
  const independent = await import(`./svpPlayback.js?client=second`)
  assert.notEqual(svpClientTransition(100).client_session_id,
    independent.svpClientTransition(1).client_session_id)
})

test('SVP errors expose upstream failure details and safely handle structured validation errors', () => {
  assert.equal(svpPlaybackError({ message: 'Request failed with status code 500',
    response: { data: { detail: 'SVP plugins not found' } } }), 'SVP plugins not found')
  assert.equal(svpPlaybackError({ message: 'Request failed with status code 422',
    response: { data: { detail: [{ msg: 'Invalid path' }] } } }), 'Request failed with status code 422')
  assert.equal(svpPlaybackError({ message: 'Network Error' }), 'Network Error')
})
