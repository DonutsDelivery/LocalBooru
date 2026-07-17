import test from 'node:test'
import assert from 'node:assert/strict'
import { getCurationProgress, loadCurationGoal, recordCurated, undoRecordedCurated } from './curationProgress.js'

function storage() {
  const values = new Map()
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) }
}

test('records and undoes daily progress', () => {
  const store = storage()
  const now = new Date(2026, 6, 16, 12)
  const day = recordCurated(now, store)
  assert.equal(getCurationProgress({ cadence: 'daily' }, now, store), 1)
  undoRecordedCurated(day, store)
  assert.equal(getCurationProgress({ cadence: 'daily' }, now, store), 0)
})

test('weekly progress starts Monday', () => {
  const store = storage()
  recordCurated(new Date(2026, 6, 13, 12), store)
  recordCurated(new Date(2026, 6, 16, 12), store)
  assert.equal(getCurationProgress({ cadence: 'weekly' }, new Date(2026, 6, 19), store), 2)
  assert.deepEqual(loadCurationGoal(store), { enabled: false, cadence: 'daily', target: 50 })
})
