import assert from 'node:assert/strict'
import test from 'node:test'

import { isWindowsOrMacDesktopApp } from '../serverManager.js'

const withPlatform = async (userAgent, tauri, callback) => {
  const windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')
  const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: tauri ? { __TAURI_INTERNALS__: {} } : {}
  })
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { userAgent }
  })
  try {
    await callback()
  } finally {
    if (windowDescriptor) Object.defineProperty(globalThis, 'window', windowDescriptor)
    else delete globalThis.window
    if (navigatorDescriptor) Object.defineProperty(globalThis, 'navigator', navigatorDescriptor)
    else delete globalThis.navigator
  }
}

// AC: @svp-single-player ac-platform-routing
test('routes Windows and macOS desktop apps to desktop SVP playback', async () => {
  await withPlatform('Mozilla/5.0 (Windows NT 10.0; Win64; x64)', true, () => {
    assert.equal(isWindowsOrMacDesktopApp(), true)
  })
  await withPlatform('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', true, () => {
    assert.equal(isWindowsOrMacDesktopApp(), true)
  })
})

// AC: @svp-single-player ac-platform-routing
test('does not route browsers, Linux, or mobile apps to desktop MSE', async () => {
  await withPlatform('Mozilla/5.0 (Windows NT 10.0; Win64; x64)', false, () => {
    assert.equal(isWindowsOrMacDesktopApp(), false)
  })
  await withPlatform('Mozilla/5.0 (X11; Linux x86_64)', true, () => {
    assert.equal(isWindowsOrMacDesktopApp(), false)
  })
  await withPlatform('Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)', true, () => {
    assert.equal(isWindowsOrMacDesktopApp(), false)
  })
})
