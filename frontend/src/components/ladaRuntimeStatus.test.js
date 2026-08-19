import assert from 'node:assert/strict'
import test from 'node:test'

import { formatLadaReadiness, normalizeAddonStatus } from './ladaRuntimeStatus.js'

// AC: @lada-runtime-readiness ac-actionable-status
for (const [status, label] of [
  ['unsupported_platform', 'Unsupported platform'],
  ['accelerator_unavailable', 'Accelerator unavailable'],
  ['incompatible_driver', 'Incompatible driver'],
  ['downloading', 'Downloading'],
  ['installing', 'Installing'],
  ['probing', 'Verifying accelerator'],
  ['repair_required', 'Repair required'],
  ['update_available', 'Update available'],
  ['runtime_failure', 'Runtime failure'],
]) {
  test(`renders ${status} as an actionable state`, () => {
    assert.deepEqual(
      formatLadaReadiness({ status, configured_backend: 'auto', reason: 'Details' }),
      {
        key: status,
        status: label,
        reason: 'Details',
        configuredBackend: 'auto',
        activeBackend: null,
        transitional: ['downloading', 'installing', 'probing'].includes(status),
        badge: ['downloading', 'installing', 'probing'].includes(status) ? 'starting' : 'error',
      },
    )
  })
}

test('renders missing managed deployment as not installed', () => {
  assert.deepEqual(
    formatLadaReadiness({
      status: 'not_installed',
      configured_backend: 'auto',
      reason: 'Install the add-on',
    }),
    {
      key: 'not_installed',
      status: 'Not installed',
      reason: 'Install the add-on',
      configuredBackend: 'auto',
      activeBackend: null,
      transitional: false,
      badge: 'not-installed',
    },
  )
})

// AC: @lada-runtime-readiness ac-active-backend-evidence
test('shows an active backend only for proven ready evidence', () => {
  assert.equal(formatLadaReadiness({
    status: 'ready',
    configured_backend: 'auto',
    active_backend: 'cuda',
  }).activeBackend, 'cuda')

  assert.deepEqual(
    formatLadaReadiness({
      status: 'runtime_failure',
      configured_backend: 'cuda',
      active_backend: 'cuda',
    }),
    {
      key: 'runtime_failure',
      status: 'Runtime failure',
      reason: null,
      configuredBackend: 'cuda',
      activeBackend: null,
      transitional: false,
      badge: 'error',
    },
  )
})

test('normalizes structured Rust lifecycle errors', () => {
  assert.deepEqual(normalizeAddonStatus({ error: 'Probe failed' }), {
    key: 'error',
    reason: 'Probe failed',
  })
})
