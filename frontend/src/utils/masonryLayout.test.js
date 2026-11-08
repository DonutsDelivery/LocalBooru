import assert from 'node:assert/strict'
import test from 'node:test'

import { distributeToColumns, tileLayoutKey } from './masonryLayout.js'

const image = (id, width = 400, height = 300) => ({ id, directory_id: 1, library_id: 'primary', width, height })
const columnIds = columns => columns.map(column => column.items.map(item => item.id))

test('first layout balances columns by estimated height', () => {
  const { columns } = distributeToColumns([image(1, 100, 400), image(2), image(3), image(4)], 2)
  assert.deepEqual(columnIds(columns), [[1], [2, 3, 4]])
})

test('prepended tiles never move a known tile to another column', () => {
  const initial = Array.from({ length: 12 }, (_, index) => image(12 - index))
  const first = distributeToColumns(initial, 4)
  const placement = new Map(first.columns.flatMap((column, index) => column.items.map(item => [item.id, index])))

  const withNew = [image(15), image(14), image(13), ...initial]
  const second = distributeToColumns(withNew, 4, first.layout)

  for (const [id, index] of placement) {
    assert.ok(second.columns[index].items.some(item => item.id === id), `tile ${id} stayed in column ${index}`)
  }
  // New tiles sit on top and are spread across columns, not stacked in one.
  assert.deepEqual(second.columns.map(column => column.items.length).sort(), [3, 4, 4, 4])
  assert.deepEqual(
    second.columns.map(column => column.items[0].id).filter(id => id > 12).sort(),
    [13, 14, 15],
  )
})

test('appended tiles go to the shortest column and keep existing placement', () => {
  const initial = [image(3), image(2), image(1)]
  const first = distributeToColumns(initial, 2)
  const second = distributeToColumns([...initial, image(0)], 2, first.layout)
  assert.deepEqual(columnIds(second.columns), [[3, 1], [2, 0]])
})

test('a column count change or an unrelated list lays out from scratch', () => {
  const list = Array.from({ length: 6 }, (_, index) => image(index + 1))
  const first = distributeToColumns(list, 3)
  assert.deepEqual(distributeToColumns(list, 2, first.layout).columns, distributeToColumns(list, 2).columns)
  const other = Array.from({ length: 6 }, (_, index) => image(index + 100))
  assert.deepEqual(distributeToColumns(other, 3, first.layout).columns, distributeToColumns(other, 3).columns)
})

test('folder tiles are keyed by path', () => {
  assert.equal(tileLayoutKey({ _isFolder: true, path: '/a' }), 'folder-/a')
})
