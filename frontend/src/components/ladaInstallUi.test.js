import test from 'node:test'
import assert from 'node:assert/strict'

import {
  canCancelManagedInstall,
  canInstallManagedAddon,
  getInstallProgressPercent,
  shouldOfferManagedInstallCancellation,
} from './ladaInstallUi.js'

// AC: @lada-managed-install ac-verified-activation
test('managed installation remains disabled until the license is explicitly accepted', () => {
  assert.equal(canInstallManagedAddon({ acceptedLicense: false }), false)
  assert.equal(canInstallManagedAddon({ acceptedLicense: true }), true)
  assert.equal(canInstallManagedAddon({ acceptedLicense: true, busy: 'install' }), false)
  assert.equal(canInstallManagedAddon({ acceptedLicense: true, transitional: true }), false)
})

// AC: @lada-managed-install ac-atomic-rollback
test('install and repair operations expose cancellation while settled operations do not', () => {
  assert.equal(canCancelManagedInstall('install'), true)
  assert.equal(canCancelManagedInstall('repair'), true)
  assert.equal(canCancelManagedInstall('cancelling'), false)
  assert.equal(canCancelManagedInstall(undefined), false)
  assert.equal(shouldOfferManagedInstallCancellation(undefined, { stage: 'downloading' }), true)
  assert.equal(shouldOfferManagedInstallCancellation('cancelling', { stage: 'downloading' }), true)
  assert.equal(shouldOfferManagedInstallCancellation('install', null), false)
})

test('installation progress is bounded and supports indeterminate metadata loading', () => {
  assert.equal(getInstallProgressPercent(null), null)
  assert.equal(getInstallProgressPercent({ completed_bytes: 1, total_bytes: 0 }), null)
  assert.equal(getInstallProgressPercent({ completed_bytes: 25, total_bytes: 100 }), 25)
  assert.equal(getInstallProgressPercent({ completed_bytes: 150, total_bytes: 100 }), 100)
  assert.equal(getInstallProgressPercent({ completed_bytes: -10, total_bytes: 100 }), 0)
})
