import { afterEach, expect, test, vi } from 'vitest'

const { invoke, open } = vi.hoisted(() => ({ invoke: vi.fn(), open: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke }))
vi.mock('@tauri-apps/api/event', () => ({}))
vi.mock('@tauri-apps/plugin-dialog', () => ({ open }))
vi.mock('@tauri-apps/plugin-shell', () => ({}))
vi.mock('@tauri-apps/api/window', () => ({}))

afterEach(() => {
  delete window.__TAURI_INTERNALS__
  delete window.__DMC_EDITION_CAPABILITIES__
  vi.unstubAllGlobals()
  vi.resetModules()
  invoke.mockReset()
  open.mockReset()
})

// AC: @library-management — Android native local directory selection.
test('Android uses permission-aware media folders instead of the unsupported desktop picker', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.stubGlobal('navigator', { userAgent: 'Android' })
  const directory = { path: '/storage/emulated/0/Music/日本語 album', show_music: true, show_images: false, show_videos: false }
  invoke.mockResolvedValue({ directory })
  const { tauriAPI } = await import('./tauriAPI')
  expect(await tauriAPI.addDirectory()).toEqual(directory)
  expect(invoke).toHaveBeenCalledWith('android_pick_media_directory')
  expect(open).not.toHaveBeenCalled()
})

test('Android cancellation returns no directory and permission failures remain visible', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.stubGlobal('navigator', { userAgent: 'Android' })
  invoke.mockResolvedValueOnce({ directory: null }).mockRejectedValueOnce(new Error('Media access denied'))
  const { tauriAPI } = await import('./tauriAPI')
  expect(await tauriAPI.addDirectory()).toBeNull()
  await expect(tauriAPI.addDirectory()).rejects.toThrow('Media access denied')
})

test('native Android permission rejection strings provide a readable error message', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.stubGlobal('navigator', { userAgent: 'Android' })
  invoke.mockRejectedValue('Media access was not granted. Allow access and try again.')
  const { tauriAPI } = await import('./tauriAPI')
  await expect(tauriAPI.addDirectory()).rejects.toMatchObject({
    message: 'Media access was not granted. Allow access and try again.'
  })
})

test('Mac keeps the native desktop folder picker and filenames with spaces', async () => {
  window.__TAURI_INTERNALS__ = {}
  vi.stubGlobal('navigator', { userAgent: 'Macintosh' })
  open.mockResolvedValue('/synthetic/Users/Person/Music/日本語 album')
  const { tauriAPI } = await import('./tauriAPI')
  expect(await tauriAPI.addDirectory()).toBe('/synthetic/Users/Person/Music/日本語 album')
  expect(open).toHaveBeenCalledWith({ directory: true, multiple: false, title: 'Select folder to watch' })
  expect(invoke).not.toHaveBeenCalled()
})

test('Mac Lite authorizes folders through the native bookmark picker', async () => {
  window.__TAURI_INTERNALS__ = {}
  window.__DMC_EDITION_CAPABILITIES__ = { edition: 'store-lite' }
  vi.stubGlobal('navigator', { userAgent: 'Macintosh' })
  invoke.mockResolvedValue('/synthetic/Music/日本語 album')
  const { tauriAPI } = await import('./tauriAPI')
  expect(await tauriAPI.addDirectory()).toBe('/synthetic/Music/日本語 album')
  expect(invoke).toHaveBeenCalledWith('pick_persistent_media_folder')
  expect(open).not.toHaveBeenCalled()
})

test('Mac Lite preserves cancellation and displays failed permission persistence', async () => {
  window.__TAURI_INTERNALS__ = {}
  window.__DMC_EDITION_CAPABILITIES__ = { edition: 'store-lite' }
  vi.stubGlobal('navigator', { userAgent: 'Macintosh' })
  invoke.mockResolvedValueOnce(null).mockRejectedValueOnce('Folder access could not be saved')
  const { tauriAPI } = await import('./tauriAPI')
  expect(await tauriAPI.addDirectory()).toBeNull()
  await expect(tauriAPI.addDirectory()).rejects.toThrow('Folder access could not be saved')
  expect(open).not.toHaveBeenCalled()
})
