import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  getMediaUrl: vi.fn(path => path), getAssetUrl: vi.fn(path => `asset://${path}`),
  isUsingLocalServer: vi.fn(() => true),
  getSVPConfig: vi.fn(async () => ({ enabled: true })),
  playVideoInterpolated: vi.fn(), stopInterpolatedStream: vi.fn(async () => {}),
  playVideoSVP: vi.fn(async () => ({ success: true, stream_url: '/svp/stream.m3u8', duration: 90 })),
  stopSVPStream: vi.fn(async () => {}),
  playVideoTranscode: vi.fn(), stopTranscodeStream: vi.fn(async () => {}),
  openSVPProcessingSession: vi.fn(), getSVPProcessingEvents: vi.fn(), fetchSVPProcessingSegment: vi.fn(),
  acknowledgeSVPInitSegment: vi.fn(), acknowledgeSVPMediaSegment: vi.fn(), pauseSVPProcessingSession: vi.fn(),
  resumeSVPProcessingSession: vi.fn(), seekSVPProcessingSession: vi.fn(), stopSVPProcessingSession: vi.fn(),
}))
const controllers = vi.hoisted(() => [])
const mseOpen = vi.hoisted(() => ({ impl: async () => {} }))
vi.mock('../../../api', () => api)
vi.mock('../../../serverManager', () => ({
  isLinuxDesktopApp: () => false,
  isMobileApp: () => false,
  isWindowsOrMacDesktopApp: () => true,
}))
vi.mock('../utils/MSESessionController', () => ({
  MSESessionController: class {
    constructor(video, client, options) {
      this.options = options
      this.open = vi.fn(() => mseOpen.impl())
      this.seek = vi.fn(async () => {})
      this.close = vi.fn(async () => {})
      this.pause = vi.fn(async () => {})
      this.resume = vi.fn(async () => {})
      controllers.push(this)
    }
  },
}))
vi.mock('hls.js', () => ({ default: { isSupported: () => false } }))
vi.mock('./useAudioNormalization', () => ({ useAudioNormalization: () => ({
  applyNormalization: () => {}, resetGain: () => {}, setOutputVolume: () => {}, setOutputMuted: () => {},
}) }))
import { useVideoStreaming } from './useVideoStreaming'

const image = { filename: 'synthetic.mp4', original_filename: 'synthetic.mp4', file_path: '/synthetic.mp4', url: '/synthetic-original.mp4' }
let video
beforeEach(() => {
  vi.clearAllMocks()
  controllers.length = 0
  mseOpen.impl = async () => {}
  api.isUsingLocalServer.mockReturnValue(true)
  vi.stubGlobal('MediaSource', class {})
  video = document.createElement('video')
  video.currentTime = 37
  video.pause = vi.fn()
  video.play = vi.fn(async () => {})
  video.load = vi.fn()
  video.canPlayType = vi.fn(() => 'maybe')
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// The auto-start effect opens SVP playback once the config has loaded.
async function mountAndStart(quality = '720p') {
  const mediaRef = { current: video }
  const hook = renderHook(
    ({ currentQuality }) => useVideoStreaming(mediaRef, image, currentQuality, { svpInstalled: true }),
    { initialProps: { currentQuality: quality } },
  )
  await waitFor(() => expect(hook.result.current.svpStreamUrl ?? hook.result.current.svpError).toBeTruthy())
  return hook
}

test('a Mac/Windows client on a remote server uses the HTTP SVP path, not an MSE session', async () => {
  api.isUsingLocalServer.mockReturnValue(false)
  const { result } = await mountAndStart('720p')

  expect(controllers).toHaveLength(0)
  expect(api.openSVPProcessingSession).not.toHaveBeenCalled()
  expect(api.playVideoSVP).toHaveBeenCalledTimes(1)
  expect(api.playVideoSVP.mock.calls[0].slice(0, 3)).toEqual(['/synthetic.mp4', 37, '720p'])
  expect(result.current.svpStreamUrl).toBe('/svp/stream.m3u8')
})

test('a Mac/Windows client on its local server still uses an MSE session', async () => {
  const { result } = await mountAndStart('720p')

  expect(controllers).toHaveLength(1)
  expect(controllers[0].open).toHaveBeenCalledWith('/synthetic.mp4', 37, 1, { targetResolution: '720p' })
  expect(api.playVideoSVP).not.toHaveBeenCalled()
  expect(result.current.svpStreamUrl).toBe('mse://svp-session')
})

test('a remote client changes quality and restarts through the HTTP SVP path', async () => {
  api.isUsingLocalServer.mockReturnValue(false)
  const { result, rerender } = await mountAndStart('720p')
  api.playVideoSVP.mockClear()

  await act(async () => { await result.current.handleQualityChange('480p') })
  rerender({ currentQuality: '480p' })
  expect(controllers).toHaveLength(0)
  expect(api.playVideoSVP).toHaveBeenCalledTimes(1)
  expect(api.playVideoSVP.mock.calls[0].slice(0, 3)).toEqual(['/synthetic.mp4', 37, '480p'])

  api.playVideoSVP.mockClear()
  await act(async () => { await result.current.restartSVPFromPosition(120) })
  expect(api.playVideoSVP.mock.calls[0].slice(0, 3)).toEqual(['/synthetic.mp4', 120, '480p'])
})

test.each([
  ['a 503 status', Object.assign(new Error('Request failed with status code 503'), { response: { status: 503 } })],
  ['the missing-graph message', new Error('No trusted SVP Manager graph is available')],
])('an MSE session open failing with %s falls back to the HTTP SVP path', async (_name, error) => {
  mseOpen.impl = async () => { throw error }
  const { result } = await mountAndStart('720p')

  expect(controllers).toHaveLength(1)
  expect(controllers[0].close).toHaveBeenCalled()
  expect(api.playVideoSVP).toHaveBeenCalledTimes(1)
  expect(api.playVideoSVP.mock.calls[0].slice(0, 3)).toEqual(['/synthetic.mp4', 37, '720p'])
  expect(result.current.svpStreamUrl).toBe('/svp/stream.m3u8')
  expect(result.current.svpError).toBeNull()
  // The original video must not have been substituted by the failure path.
  expect(video.src).not.toContain('synthetic-original')

})

test('an unrelated MSE open failure still falls back to the original video', async () => {
  mseOpen.impl = async () => { throw new Error('boom') }
  const { result } = await mountAndStart('720p')

  expect(api.playVideoSVP).not.toHaveBeenCalled()
  expect(result.current.svpStreamUrl).toBeNull()
  expect(result.current.svpError).toBe('boom')
})
