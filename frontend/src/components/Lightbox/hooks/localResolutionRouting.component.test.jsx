import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'

const state = vi.hoisted(() => ({ linux: true, mobile: false, embedded: true, svp: false }))
const api = vi.hoisted(() => ({
  getMediaUrl: vi.fn(path => path), getAssetUrl: vi.fn(path => `asset://${path}`),
  isUsingLocalServer: vi.fn(() => state.embedded),
  getSVPConfig: vi.fn(async () => ({ enabled: state.svp })),
  playVideoInterpolated: vi.fn(), stopInterpolatedStream: vi.fn(async () => {}),
  playVideoSVP: vi.fn(async () => ({ success: true, stream_url: '/synthetic-svp.m3u8' })),
  stopSVPStream: vi.fn(async () => {}),
  playVideoTranscode: vi.fn(async () => ({ success: true, stream_url: '/synthetic-transcode.m3u8' })),
  stopTranscodeStream: vi.fn(async () => {}),
  openSVPProcessingSession: vi.fn(), getSVPProcessingEvents: vi.fn(), fetchSVPProcessingSegment: vi.fn(),
  acknowledgeSVPInitSegment: vi.fn(), acknowledgeSVPMediaSegment: vi.fn(), pauseSVPProcessingSession: vi.fn(),
  resumeSVPProcessingSession: vi.fn(), seekSVPProcessingSession: vi.fn(), stopSVPProcessingSession: vi.fn(),
}))
vi.mock('../../../api', () => api)
vi.mock('../../../serverManager', () => ({
  isLinuxDesktopApp: () => state.linux,
  isMobileApp: () => state.mobile,
  isWindowsOrMacDesktopApp: () => false,
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
  Object.assign(state, { linux: true, mobile: false, embedded: true, svp: false })
  video = document.createElement('video')
  video.currentTime = 37
  video.pause = vi.fn()
  video.play = vi.fn(async () => {})
  video.load = vi.fn()
  video.canPlayType = vi.fn(() => 'probably')
})
afterEach(cleanup)
function mount(quality = '720p') {
  const mediaRef = { current: video }
  return renderHook(() => useVideoStreaming(mediaRef, image, quality, { svpInstalled: true }))
}

// AC: @local-decoded-video-resolution ac-local-raw
test.each([false, true])('local lower resolution with SVP=%s retains original source without an encoder', async enabled => {
  state.svp = enabled
  video.src = image.url
  const { result } = mount()
  await waitFor(() => expect(result.current.svpConfigLoaded).toBe(true))
  await act(async () => result.current.handleQualityChange('480p'))
  expect(result.current.nativeSvpPlayback).toBe(true)
  expect(api.playVideoTranscode).not.toHaveBeenCalled()
  expect(api.playVideoSVP).not.toHaveBeenCalled()
  expect(video.getAttribute('src')).toBe(image.url)
  expect(video.load).not.toHaveBeenCalled()
})

// AC: @local-decoded-video-resolution ac-remote
test.each([{ linux: true, mobile: false }, { linux: false, mobile: true }])('remote %j lower quality retains encoded streaming', async platform => {
  Object.assign(state, platform, { embedded: false })
  const { result } = mount('original')
  await waitFor(() => expect(result.current.svpConfigLoaded).toBe(true))
  await act(async () => result.current.handleQualityChange('720p'))
  expect(result.current.nativeSvpPlayback).toBe(false)
  expect(api.playVideoTranscode).toHaveBeenCalledWith(image.file_path, 37, '720p', expect.any(AbortSignal))
  expect(result.current.transcodeStreamUrl).toBe('/synthetic-transcode.m3u8')
})

test('Android Original restores the original network URL rather than a lower encoded quality', async () => {
  Object.assign(state, { linux: false, mobile: true, embedded: false })
  const { result } = mount('original')
  await waitFor(() => expect(result.current.svpConfigLoaded).toBe(true))
  await act(async () => result.current.handleQualityChange('original'))
  expect(video.getAttribute('src')).toBe(image.url)
  expect(api.playVideoTranscode).not.toHaveBeenCalled()
  expect(api.getAssetUrl).not.toHaveBeenCalled()
})

test('Android SVP keeps the server encoded interpolation transport', async () => {
  Object.assign(state, { linux: false, mobile: true, embedded: false, svp: true })
  const { result } = mount('720p')
  await waitFor(() => expect(result.current.svpStreamUrl).toBe('/synthetic-svp.m3u8'))
  expect(api.playVideoSVP).toHaveBeenCalled()
  expect(api.playVideoTranscode).not.toHaveBeenCalled()
})

// AC: @local-decoded-video-resolution ac-remote
// AC: @local-decoded-video-resolution ac-ownership
test.each([false, true])('switching remote encoded playback to embedded clears the old stream, SVP=%s', async svp => {
  Object.assign(state, { linux: true, embedded: false, svp })
  const { result, rerender } = mount('720p')
  await waitFor(() => expect(svp ? result.current.svpStreamUrl : result.current.transcodeStreamUrl)
    .toBe(svp ? '/synthetic-svp.m3u8' : '/synthetic-transcode.m3u8'))
  const encodedStarts = api.playVideoTranscode.mock.calls.length + api.playVideoSVP.mock.calls.length
  state.embedded = true
  rerender()
  await waitFor(() => {
    expect(result.current.nativeSvpPlayback).toBe(true)
    expect(result.current.transcodeStreamUrl).toBeNull()
    expect(result.current.svpStreamUrl).toBeNull()
  })
  expect(api.playVideoTranscode.mock.calls.length + api.playVideoSVP.mock.calls.length).toBe(encodedStarts)
})
