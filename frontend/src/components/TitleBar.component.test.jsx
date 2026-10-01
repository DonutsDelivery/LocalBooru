import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'

const platform = vi.hoisted(() => ({ mobile: false, desktop: true }))
vi.mock('../serverManager', () => ({
  isMobileApp: () => platform.mobile,
  getServers: async () => [],
}))
vi.mock('../tauriAPI', () => ({
  isTauri: () => platform.desktop,
  getDesktopAPI: () => null,
}))
vi.mock('./UpdateBanner', () => ({ default: () => null }))
import TitleBar from './TitleBar'

beforeEach(() => { platform.mobile = false; platform.desktop = true })
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  document.documentElement.classList.remove('desktop-app')
  document.documentElement.style.removeProperty('--title-bar-height')
})

function setPlatform(value) {
  vi.stubGlobal('navigator', { platform: value })
}

test('macOS leaves window chrome to the native traffic lights without duplicate controls', async () => {
  setPlatform('MacIntel')
  const { container } = render(<TitleBar />)
  expect(container.querySelector('.title-bar')).toBeNull()
  expect(screen.queryByTitle('Close')).toBeNull()
  expect(document.querySelector('.window-resize-handle')).toBeNull()
  await waitFor(() => expect(document.documentElement.style.getPropertyValue('--title-bar-height')).toBe('0px'))
  expect(document.documentElement.classList.contains('desktop-app')).toBe(false)
})

test('Linux keeps the custom window controls and their titlebar offset', async () => {
  setPlatform('Linux x86_64')
  render(<TitleBar />)
  expect(screen.getByTitle('Close')).toBeTruthy()
  expect(screen.getByTitle('Maximize')).toBeTruthy()
  expect(screen.getByTitle('Minimize to tray')).toBeTruthy()
  await waitFor(() => expect(document.documentElement.style.getPropertyValue('--title-bar-height')).toBe('32px'))
})

test('iOS retains its mobile server button without desktop window controls', () => {
  setPlatform('MacIntel')
  platform.mobile = true
  render(<TitleBar />)
  expect(screen.getByTitle('Switch Server')).toBeTruthy()
  expect(screen.queryByTitle('Close')).toBeNull()
})
