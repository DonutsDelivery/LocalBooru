import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

import {
  clearNativeVideoCompositing,
  setNativeVideoCompositing,
} from './Lightbox/utils/nativeVideoCompositing.js'

function element() {
  const values = new Set()
  return {
    values,
    classList: {
      toggle(name, force) {
        if (force) values.add(name)
        else values.delete(name)
      },
      remove(name) { values.delete(name) },
    },
  }
}

test('explicitly opens and closes the document compositing aperture', () => {
  const root = element()
  setNativeVideoCompositing(root, true)
  assert.equal(root.values.has('native-video-compositing'), true)
  setNativeVideoCompositing(root, false)
  assert.equal(root.values.has('native-video-compositing'), false)
})

test('cleanup always removes the compositing aperture class', () => {
  const root = element()
  setNativeVideoCompositing(root, true)
  clearNativeVideoCompositing(root)
  assert.deepEqual([...root.values], [])
})

test('native Android video only makes the WebView transparent while playback is active', async () => {
  const source = await readFile(
    new URL('../../../src-tauri/gen/android/app/src/main/java/com/localbooru/app/NativeVideoController.kt', import.meta.url),
    'utf8',
  )
  const attach = source.slice(source.indexOf('private fun attachBehindWebView()'), source.indexOf('private fun ensurePlayer()'))
  const open = source.slice(source.indexOf('fun open('), source.indexOf('@JavascriptInterface\n  fun close('))
  const close = source.slice(source.indexOf('private fun closeActivePlayer()'), source.indexOf('private fun withGeneration('))

  assert.doesNotMatch(attach, /setBackgroundColor\(Color\.TRANSPARENT\)/)
  assert.match(open, /setBackgroundColor\(Color\.TRANSPARENT\)/)
  assert.match(close, /surfaceView\.visibility = View\.GONE/)
  assert.match(close, /setLayerType\(View\.LAYER_TYPE_HARDWARE/)
  assert.match(close, /setBackgroundColor\(Color\.BLACK\)/)
})
