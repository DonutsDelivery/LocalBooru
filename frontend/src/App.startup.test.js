import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const appSource = await readFile(new URL('./App.jsx', import.meta.url), 'utf8')

test('initializes curation before creating callbacks that read it', () => {
  const curationHook = appSource.indexOf('const curation = useCurationGame(')
  const sidebarTouchCallback = appSource.indexOf('const handleTouchEnd = useCallback(')

  assert.notEqual(curationHook, -1)
  assert.notEqual(sidebarTouchCallback, -1)
  assert.ok(curationHook < sidebarTouchCallback)
})
