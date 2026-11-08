import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  getMediaUrl: vi.fn(path => path), getAssetUrl: vi.fn(path => `asset://${path}`),
  isUsingLocalServer: vi.fn(() => true),
  getSVPConfig: vi.fn(async () => ({ enabled: true })),
  playVideoInterpolated: vi.fn(), stopInterpolatedStream: vi.fn(async () => {}),
  playVideoSVP: vi.fn(), stopSVPStream: vi.fn(async () => {}),
  playVideoTranscode: vi.fn(), stopTranscodeStream: vi.fn(async () => {}),
  openSVPProcessingSession: vi.fn(), getSVPProcessingEvents: vi.fn(), fetchSVPProcessingSegment: vi.fn(),
  acknowledgeSVPInitSegment: vi.fn(), acknowledgeSVPMediaSegment: vi.fn(), pauseSVPProcessingSession: vi.fn(),
  resumeSVPProcessingSession: vi.fn(), seekSVPProcessingSession: vi.fn(), stopSVPProcessingSession: vi.fn(),
}))
const controllers = vi.hoisted(() => [])
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
      this.open = vi.fn(async () => {})
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
  vi.stubGlobal('MediaSource', class {})
  video = document.createElement('video')
  video.currentTime = 37
  video.pause = vi.fn()
  video.play = vi.fn(async () => {})
  video.load = vi.fn()
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

async function mountWithMseSession(quality) {
  const mediaRef = { current: video }
  const hook = renderHook(
    ({ currentQuality }) => useVideoStreaming(mediaRef, image, currentQuality, { svpInstalled: true }),
    { initialProps: { currentQuality: quality } },
  )
  await waitFor(() => expect(hook.result.current.svpConfigLoaded).toBe(true))
  await act(async () => { await hook.result.current.startSVPStream(37, null, true) })
  await waitFor(() => expect(hook.result.current.svpStreamUrl).toBe('mse://svp-session'))
  return hook
}

test('opens the MSE session at the selected quality', async () => {
  await mountWithMseSession('720p')
  expect(controllers).toHaveLength(1)
  expect(controllers[0].open).toHaveBeenCalledWith('/synthetic.mp4', 37, 1, { targetResolution: '720p' })
})

test('original quality opens the MSE session without a target resolution', async () => {
  await mountWithMseSession('original')
  expect(controllers[0].open).toHaveBeenCalledWith('/synthetic.mp4', 37, 1, { targetResolution: null })
})

test('a quality change reopens the MSE session at the current position instead of seeking', async () => {
  video.currentTime = 52
  const { result } = await mountWithMseSession('1080p')
  const controller = controllers[0]
  controller.open.mockClear()

  await act(async () => { await result.current.handleQualityChange('480p') })

  expect(controller.seek).not.toHaveBeenCalled()
  expect(controller.open).toHaveBeenCalledTimes(1)
  expect(controller.open).toHaveBeenCalledWith('/synthetic.mp4', 52, 1, { targetResolution: '480p' })
  expect(controllers).toHaveLength(1)
  expect(result.current.svpStreamUrl).toBe('mse://svp-session')
})

test('a reduced session does not replace the known source resolution', async () => {
  const { result } = await mountWithMseSession('480p')
  act(() => result.current.setSourceResolution({ width: 3840, height: 2160 }))
  act(() => controllers[0].options.onMetadata({ source_duration: 60, width: 854, height: 480 }))
  expect(result.current.sourceResolution).toEqual({ width: 3840, height: 2160 })
})
