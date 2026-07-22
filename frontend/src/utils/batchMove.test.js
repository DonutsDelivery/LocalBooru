import assert from 'node:assert/strict'
import test from 'node:test'

import { loadMoveDirectoryOptions } from './batchMove.js'

// AC: @batch-move-dialog-stability ac-directory-envelope
test('Move loads destinations from the directory response envelope', async () => {
  const directories = [{ id: 1, name: 'Pictures', path: 'C:\\Pictures' }]

  assert.deepEqual(
    await loadMoveDirectoryOptions(async () => ({ directories })),
    directories
  )
})

// AC: @batch-move-dialog-stability ac-fetch-failure
test('Move rejects malformed and failed directory responses without rendering them', async () => {
  await assert.rejects(
    loadMoveDirectoryOptions(async () => ({ id: 1 })),
    /invalid response/
  )
  await assert.rejects(
    loadMoveDirectoryOptions(async () => { throw new Error('offline') }),
    /offline/
  )
})
