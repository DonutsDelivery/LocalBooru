import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useVideoPlayback } from './useVideoPlayback'

vi.mock('../../../api', () => ({ savePlaybackPosition: vi.fn(() => Promise.resolve()) }))

class DelayedVideo extends EventTarget {
  constructor() {
    super()
    this.src = 'http://synthetic.invalid/original.mp4'
    this.currentTime = 30
    this.duration = 120
    this.paused = false
    this.seeking = false
    this.requests = []
    this.play = vi.fn(() => Promise.resolve())
    this.fastSeek = vi.fn(time => { this.requests.push(time); this.seeking = true })
    this.buffered = { length: 0 }
  }
  complete(time = this.requests.at(-1)) {
    this.currentTime = time
    this.seeking = false
    this.dispatchEvent(new Event('seeked'))
  }
}

function setup(overrides = {}) {
  const video = new DelayedVideo()
  const mediaRef = { current: video }
  const stream = {
    streamTransitioningRef: { current: false },
    getCurrentAbsoluteTime: () => mediaRef.current.currentTime,
    setSvpPendingSeek: vi.fn(),
    restartSVPFromPosition: vi.fn(),
    restartTranscodeFromPosition: vi.fn(),
    ...overrides,
  }
  const hook = renderHook(() => useVideoPlayback(mediaRef, stream))
  act(() => hook.result.current.handleLoadedMetadata())
  return { ...hook, video, mediaRef, stream }
}

let frames
const flushFrame = () => act(() => {
  const scheduled = [...frames.values()]
  frames.clear()
  scheduled.forEach(callback => callback())
})
beforeEach(() => {
  frames = new Map()
  let frameId = 0
  vi.stubGlobal('requestAnimationFrame', callback => { frames.set(++frameId, callback); return frameId })
  vi.stubGlobal('cancelAnimationFrame', id => frames.delete(id))
})
afterEach(() => vi.unstubAllGlobals())

describe('original playback seeks', () => {
  // AC: @responsive-original-stream-seeking ac-rapid-relative, ac-responsive-controls
  it('coalesces one display frame and accepts latest intent before native completion', () => {
    const { result, video } = setup()
    act(() => {
      result.current.seekVideo(10)
      result.current.seekVideo(10)
      result.current.seekVideo(-10)
      result.current.seekVideo(10)
    })
    expect(result.current.currentTime).toBe(50)
    expect(video.requests).toEqual([])
    flushFrame()
    expect(video.requests).toEqual([50])
    expect(video.play).not.toHaveBeenCalled()
    act(() => result.current.seekVideo(10))
    flushFrame() // No wait for the first native seek to finish.
    expect(video.requests).toEqual([50, 60])
    act(() => video.complete(59)) // fastSeek can land on a nearby keyframe.
    act(() => result.current.seekVideo(10))
    flushFrame()
    expect(video.requests).toEqual([50, 60, 69])
  })

  // AC: @responsive-original-stream-seeking ac-original-path, ac-rapid-relative
  it('preserves paused playback and clamps rapid intent at duration', () => {
    const { result, video } = setup()
    video.paused = true
    act(() => {
      result.current.seekVideo(100)
      result.current.seekVideo(100)
      result.current.seekVideo(-10)
    })
    expect(result.current.currentTime).toBe(110)
    flushFrame()
    expect(video.requests).toEqual([110])
    act(() => video.complete())
    expect(video.play).not.toHaveBeenCalled()
  })

  // AC: @responsive-original-stream-seeking ac-original-path
  it('does not use a stale streaming transition or pending SVP seek for original playback', () => {
    const { result, video, stream } = setup({
      streamTransitioningRef: { current: true },
      svpPendingSeek: 80,
      getCurrentAbsoluteTime: () => 80,
    })
    act(() => result.current.seekVideo(10))
    flushFrame()
    expect(video.requests).toEqual([40])
    act(() => video.complete())
    act(() => result.current.handleTimeUpdate())
    expect(result.current.currentTimeRef.current).toBe(40)
    expect(stream.restartSVPFromPosition).not.toHaveBeenCalled()
    expect(stream.restartTranscodeFromPosition).not.toHaveBeenCalled()
  })

  // AC: @responsive-original-stream-seeking ac-owner-boundary
  it('discards scheduled and unsettled intent on source replacement, removal and unmount', () => {
    const { result, video, mediaRef, unmount } = setup()
    act(() => { result.current.seekVideo(10); result.current.seekVideo(10) })
    mediaRef.current = new DelayedVideo()
    flushFrame()
    expect(video.requests).toEqual([])
    act(() => result.current.seekVideo(10))
    flushFrame()
    expect(mediaRef.current.requests).toEqual([40])
    act(() => result.current.seekVideo(10))
    mediaRef.current.src = 'http://synthetic.invalid/replacement.mp4'
    flushFrame()
    expect(mediaRef.current.requests).toEqual([40])
    act(() => { result.current.seekVideo(10); result.current.seekVideo(10) })
    unmount()
    flushFrame()
    expect(mediaRef.current.requests).toEqual([40])
    video.complete() // Removed host cannot dispatch into the successor.
    expect(video.requests).toEqual([])
  })

  // AC: @responsive-original-stream-seeking ac-rejected-seek
  it('clears unsettled intent on a native error and permits another seek', () => {
    const { result, video } = setup()
    act(() => result.current.seekVideo(10))
    flushFrame()
    act(() => video.dispatchEvent(new Event('error')))
    act(() => result.current.seekVideo(-10))
    flushFrame()
    expect(video.requests).toEqual([40, 20])
  })

  // AC: @responsive-original-stream-seeking ac-rapid-relative, ac-owner-boundary
  it('ignores an older seeked while a newer native seek is still in progress', () => {
    const { result, video } = setup()
    act(() => result.current.seekVideo(10))
    flushFrame()
    act(() => result.current.seekVideo(10))
    flushFrame()
    video.currentTime = 39
    video.seeking = true
    act(() => video.dispatchEvent(new Event('seeked')))
    act(() => result.current.seekVideo(10))
    flushFrame()
    expect(video.requests).toEqual([40, 50, 60])
  })

  // AC: @responsive-original-stream-seeking ac-original-path
  it('uses currentTime when fastSeek is unavailable and preserves precise timeline input', () => {
    const { result, video } = setup()
    video.fastSeek = undefined
    let physicalTime = 30
    const precise = []
    Object.defineProperty(video, 'currentTime', {
      get: () => physicalTime,
      set: time => { precise.push(time); video.seeking = true },
    })
    act(() => result.current.seekVideo(10))
    flushFrame()
    expect(precise).toEqual([40])
    physicalTime = 40
    video.seeking = false
    act(() => video.dispatchEvent(new Event('seeked')))
    video.fastSeek = vi.fn()
    result.current.timelineRef.current = { getBoundingClientRect: () => ({ left: 0, width: 100 }) }
    act(() => result.current.handleSeekStart({ clientX: 75 }))
    act(() => result.current.handleSeekEnd())
    flushFrame()
    expect(precise).toEqual([40, 90])
    expect(video.fastSeek).not.toHaveBeenCalled()
    expect(video.play).not.toHaveBeenCalled()
  })

  // AC: @responsive-original-stream-seeking ac-rejected-seek, ac-owner-boundary
  it('reports native and scheduled setter errors, ignores removed-host errors and clears on reopen', () => {
    const { result, video, mediaRef } = setup()
    const removed = new DelayedVideo()
    act(() => result.current.handlePlaybackError(removed))
    expect(result.current.playbackError).toBeNull()
    video.currentSrc = video.src
    video.src = 'http://synthetic.invalid/replacement.mp4'
    act(() => result.current.handlePlaybackError(video))
    expect(result.current.playbackError).toBeNull()
    video.currentSrc = video.src
    video.fastSeek.mockImplementation(() => { throw new Error('synthetic seek rejection') })
    act(() => result.current.seekVideo(10))
    expect(() => flushFrame()).not.toThrow()
    expect(result.current.playbackError).toMatch(/Close and reopen/)
    act(() => result.current.handleLoadedMetadata())
    expect(result.current.playbackError).toBeNull()
    act(() => result.current.handlePlaybackError(video))
    expect(result.current.playbackError).toMatch(/Close and reopen/)
    mediaRef.current = removed
    act(() => result.current.resetPlaybackState())
    act(() => result.current.handlePlaybackError(video))
    expect(result.current.playbackError).toBeNull()
  })

  // AC: @responsive-original-stream-seeking ac-stream-routing
  it('leaves SVP and transcoded stream seek routes intact', () => {
    const svp = setup({ svpStreamUrl: 'synthetic.m3u8', svpStartOffset: 0, svpBufferedDuration: 5 })
    act(() => svp.result.current.seekVideo(10))
    expect(svp.stream.restartSVPFromPosition).toHaveBeenCalledWith(40)
    expect(svp.video.requests).toEqual([])
    svp.unmount()
    const transcode = setup({ transcodeStreamUrl: 'synthetic.m3u8', transcodeStartOffset: 0, transcodeBufferedDuration: 5 })
    act(() => transcode.result.current.seekVideo(10))
    expect(transcode.stream.restartTranscodeFromPosition).toHaveBeenCalledWith(40)
    expect(transcode.video.requests).toEqual([])
    transcode.unmount()
    const buffered = setup({ svpStreamUrl: 'synthetic.m3u8', svpStartOffset: 10, svpBufferedDuration: 90 })
    act(() => buffered.result.current.seekVideo(10))
    expect(buffered.video.currentTime).toBe(30)
    expect(buffered.stream.restartSVPFromPosition).not.toHaveBeenCalled()
    expect(buffered.stream.setSvpPendingSeek).toHaveBeenCalledWith(null)
    buffered.unmount()
    const encoded = setup({ transcodeStreamUrl: 'synthetic.m3u8', transcodeStartOffset: 10, transcodeBufferedDuration: 90 })
    act(() => encoded.result.current.seekVideo(10))
    expect(encoded.video.currentTime).toBe(30)
    expect(encoded.stream.restartTranscodeFromPosition).not.toHaveBeenCalled()
  })
})
