import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const mocks = vi.hoisted(() => {
  const noop = () => {}
  return {
    noop, initialSvp: false, embedded: true, configLoaded: true, listeners: null,
    bridge: {
      acquireSvpVideoHostEpoch: vi.fn(async () => 7),
      configureLocalVideoResolution: vi.fn(async () => true),
      verifyLocalVideoResolution: vi.fn(async () => true),
      updateSvpManagerPlayback: vi.fn(async () => {}),
      subscribeToSvpManager: vi.fn(async listeners => { mocks.listeners = listeners; return noop }),
    },
    toast: { error: vi.fn(), success: vi.fn() },
    handleEncodedQuality: vi.fn(async () => {}),
    sourceResolution: vi.fn(), dimensions: vi.fn(),
    ui: { showUI: true, isFullscreen: false, resetHideTimer: noop, handleMouseMove: noop,
      handleTouchInteractionStart: noop, consumeRevealTap: () => false, cancelRevealTap: noop,
      handleToggleFullscreen: noop },
    zoom: { zoom: { scale: 1 }, getZoomTransform: () => ({}), resetZoom: noop,
      touchMoved: { current: false }, touchHandled: { current: false } },
    playback: { duration: 120, durationRef: { current: 120 }, currentTimeRef: { current: 0 },
      isPlaying: true, volume: 1, videoDisplayMode: 'fit', playbackSpeed: 1,
      resetPlaybackState: noop, handleLoadedMetadata: noop, handleVideoPlay: noop,
      handleVideoPause: noop, handleTimeUpdate: noop, applyResumeState: noop },
    subtitles: { stopSubtitlesStream: noop, autoGenerate: noop, subtitles: [] },
  }
})
vi.mock('../../api', () => ({
  getMediaUrl: path => path, getAssetUrl: path => path, isUsingLocalServer: () => mocks.embedded,
  getFileDimensions: (...args) => mocks.dimensions(...args),
  getPlaybackPosition: async () => ({}), fetchCollections: async () => ({ collections: [] }),
  getSVPConfig: async () => ({ enabled: mocks.initialSvp }),
  updateSVPConfig: async config => config,
  addToCollection: mocks.noop, createCollection: mocks.noop, getShareNetworkInfo: mocks.noop, uploadImage: mocks.noop,
}))
vi.mock('../../tauriAPI', () => ({ getDesktopAPI: () => mocks.bridge }))
vi.mock('../../serverManager', () => ({ isMobileApp: () => false }))
vi.mock('../Toast', () => ({ toast: mocks.toast }))
vi.mock('../../hooks/useAddonStatus', () => ({ useAddonStatus: () => ({ installed: true }) }))
vi.mock('../../hooks/useImageWorkflow', () => ({ useImageWorkflow: () => ({ available: false }) }))
vi.mock('./hooks/useUIVisibility', () => ({ useUIVisibility: () => mocks.ui }))
vi.mock('./hooks/useZoomPan', () => ({ useZoomPan: () => mocks.zoom }))
vi.mock('./hooks/useVideoPlayback', () => ({ useVideoPlayback: ref => { mocks.mediaRef = ref; return mocks.playback } }))
vi.mock('./hooks/useWhisperSubtitles', () => ({ useWhisperSubtitles: () => mocks.subtitles }))
vi.mock('./hooks/useCastSession', () => ({ useCastSession: () => ({ isCasting: false }) }))
vi.mock('./hooks/useTimelinePreview', () => ({ useTimelinePreview: () => ({ previewFrames: [] }) }))
vi.mock('./hooks/useVideoGestures', () => ({ useVideoGestures: () => ({}) }))
vi.mock('./hooks/useAutoAdvance', () => ({ useAutoAdvance: () => ({ countdown: null }) }))
vi.mock('./hooks/useShareStream', () => ({ useShareStream: () => ({}) }))
vi.mock('./VRVideoViewport', () => ({ default: () => null }))
vi.mock('../ContextMenu', () => ({ default: () => null }))
vi.mock('../SVPSideMenu', () => ({ default: () => null }))
vi.mock('./hooks/useVideoStreaming', async () => {
  const { useState } = await import('react')
  return { useVideoStreaming: ref => {
    const [svpConfig, setSvpConfig] = useState({ enabled: mocks.initialSvp })
    return { nativeSvpPlayback: mocks.embedded, svpConfigLoaded: mocks.configLoaded, svpConfig, setSvpConfig,
      sourceResolution: { width: 1920, height: 1080 }, setSourceResolution: mocks.sourceResolution,
      streamTransitioningRef: { current: false }, opticalFlowConfig: { enabled: false },
      capturePlaybackIntent: () => ({ position: ref.current.currentTime, shouldPlay: !ref.current.paused }),
      handleQualityChange: mocks.handleEncodedQuality, setSvpError: mocks.noop, setSvpLoading: mocks.noop,
      checkCodecFallback: mocks.noop,
    }
  } }
})
import Lightbox from './Lightbox'

const images = [1, 2].map(id => ({ id, filename: `synthetic-${id}.mp4`, original_filename: `synthetic-${id}.mp4`,
  file_path: `/synthetic-${id}.mp4`, url: `/synthetic-${id}.mp4`, video_fps: 24 }))
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  mocks.playback.isPlaying = true
  mocks.playback.handleVideoPlay = vi.fn(() => { mocks.playback.isPlaying = true })
  mocks.playback.handleVideoPause = vi.fn(() => { mocks.playback.isPlaying = false })
  mocks.playback.volume = 1
  mocks.playback.isMuted = false
  mocks.playback.restoreAudioState = vi.fn((volume, muted) => {
    mocks.mediaRef.current.volume = volume
    mocks.mediaRef.current.muted = muted
  })
  mocks.dimensions.mockResolvedValue({ width: 1920, height: 1080, fps: 24 })
  mocks.playback.playbackError = null
  mocks.playback.handlePlaybackError = vi.fn()
  mocks.initialSvp = false
  mocks.configLoaded = true
  mocks.embedded = true
  mocks.bridge.configureLocalVideoResolution.mockResolvedValue(true)
  mocks.bridge.verifyLocalVideoResolution.mockResolvedValue(true)
  mocks.bridge.updateSvpManagerPlayback.mockResolvedValue(undefined)
  Object.defineProperties(HTMLMediaElement.prototype, {
    pause: { configurable: true, value: vi.fn(function () { this.__paused = true }) },
    play: { configurable: true, value: vi.fn(function () { this.__paused = false; return Promise.resolve() }) },
    load: { configurable: true, value: vi.fn() },
    paused: { configurable: true, get() { return this.__paused ?? false } },
    readyState: { configurable: true, get() { return 3 } },
    duration: { configurable: true, get() { return 120 } },
  })
  Object.defineProperties(HTMLVideoElement.prototype, {
    videoWidth: { configurable: true, get() { return 1280 } },
    videoHeight: { configurable: true, get() { return 720 } },
  })
})
afterEach(cleanup)
function mount(index = 0) {
  return render(<Lightbox images={images} currentIndex={index} onClose={mocks.noop} onNav={mocks.noop} />)
}
async function videoReady(container, index = 0) {
  await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toBe(images[index].url))
  return container.querySelector('video')
}
function choose(label) {
  fireEvent.click(screen.getAllByTitle(/^Quality:/)[0])
  fireEvent.click(screen.getByText(label, { selector: '.quality-label' }))
}

// AC: @local-decoded-video-resolution ac-ownership
test.each([false, true])('local replacement retains seek/audio and paused=%s without encoded playback', async paused => {
  const { container } = mount()
  const first = await videoReady(container)
  first.currentTime = 37; first.__paused = paused; first.volume = 0.35; first.muted = true
  mocks.playback.volume = 0.35; mocks.playback.isMuted = true
  choose('720p')
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first))
  const replacement = await videoReady(container)
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenLastCalledWith(expect.objectContaining({ bounds: { maxWidth: 1280, maxHeight: 720 } }))
  fireEvent.loadedMetadata(replacement)
  await waitFor(() => expect(mocks.bridge.verifyLocalVideoResolution).toHaveBeenCalled())
  expect(replacement.currentTime).toBe(37)
  fireEvent.seeked(replacement)
  expect(replacement.paused).toBe(paused)
  expect(replacement.volume).toBe(0.35)
  expect(replacement.muted).toBe(true)
  expect(mocks.handleEncodedQuality).not.toHaveBeenCalled()
  expect(mocks.sourceResolution).toHaveBeenCalledWith({ width: 1920, height: 1080 })
})

test('failed graph withdrawal resumes the current player and reports a visible error', async () => {
  const { container } = mount()
  const first = await videoReady(container)
  first.currentTime = 37
  first.__paused = false
  mocks.bridge.updateSvpManagerPlayback.mockRejectedValueOnce(new Error('synthetic bridge failure'))
  choose('480p')
  await waitFor(() => expect(mocks.toast.error).toHaveBeenCalledWith(expect.stringContaining('synthetic bridge failure')))
  expect(container.querySelector('video')).toBe(first)
  expect(first.currentTime).toBe(37)
  expect(first.paused).toBe(false)
})

test('navigation invalidates delayed resolution readiness from the previous physical player', async () => {
  const { container, rerender } = mount()
  const first = await videoReady(container)
  let release
  mocks.bridge.configureLocalVideoResolution.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
  choose('480p')
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first))
  await waitFor(() => expect(release).toBeTypeOf('function'))
  const stale = container.querySelector('video')
  expect(stale.getAttribute('src')).toBeNull()
  rerender(<Lightbox images={images} currentIndex={1} onClose={mocks.noop} onNav={mocks.noop} />)
  const current = await videoReady(container, 1)
  await act(async () => release(true))
  expect(container.querySelector('video')).toBe(current)
  expect(current.getAttribute('src')).toBe(images[1].url)
  expect(stale.getAttribute('src')).toBeNull()
})

test('rapid selections keep only the latest generation and original playback position', async () => {
  const { container } = mount()
  const first = await videoReady(container)
  first.currentTime = 37; first.__paused = false
  let release
  mocks.bridge.updateSvpManagerPlayback.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
  choose('720p')
  choose('480p')
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first))
  const current = await videoReady(container)
  await act(async () => release())
  expect(container.querySelector('video')).toBe(current)
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenLastCalledWith(expect.objectContaining({ bounds: { maxWidth: 854, maxHeight: 480 } }))
  fireEvent.loadedMetadata(current)
  await waitFor(() => expect(mocks.bridge.verifyLocalVideoResolution).toHaveBeenCalled())
  expect(current.currentTime).toBe(37)
  fireEvent.seeked(current)
  expect(current.paused).toBe(false)
})

test('a failed resize waits for Original acknowledgement before attaching the source', async () => {
  const { container } = mount()
  const first = await videoReady(container)
  let release
  mocks.bridge.configureLocalVideoResolution
    .mockRejectedValueOnce(new Error('synthetic resize failure'))
    .mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
  choose('480p')
  await waitFor(() => expect(release).toBeTypeOf('function'))
  const current = container.querySelector('video')
  expect(current).not.toBe(first)
  expect(current.getAttribute('src')).toBeNull()
  expect(mocks.toast.error).toHaveBeenCalledWith(expect.stringContaining('synthetic resize failure'))
  await act(async () => release(true))
  await videoReady(container)
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenLastCalledWith(expect.objectContaining({ bounds: null }))
})

// AC: @local-decoded-video-resolution ac-local-raw
// AC: @local-decoded-video-resolution ac-geometry
test('SVP enable, graph handoff and disable retain the selected raw resolution', async () => {
  localStorage.setItem('video_quality_preference', '720p')
  const { container } = mount()
  const first = await videoReady(container)
  first.currentTime = 37; first.__paused = false
  fireEvent.click(screen.getByTitle('Enable SVP interpolation'))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first))
  const waitingGraph = await videoReady(container)
  fireEvent.loadedMetadata(waitingGraph)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([update]) => update.enabled)).toBe(true))
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([update]) => update.enabled).at(-1)[0]
  expect(owner.resizeRevision).toEqual(expect.any(Number))
  expect(owner.width).toBe(1280)
  expect(owner.height).toBe(720)
  await act(async () => mocks.listeners.onFilterChanged({ ...owner, enabled: true }))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(waitingGraph))
  const withGraph = await videoReady(container)
  fireEvent.loadedMetadata(withGraph)
  fireEvent.seeked(withGraph)
  expect(withGraph.currentTime).toBe(37)
  fireEvent.click(screen.getByTitle('Disable SVP interpolation'))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(withGraph))
  const withoutGraph = await videoReady(container)
  fireEvent.loadedMetadata(withoutGraph)
  fireEvent.seeked(withoutGraph)
  expect(withoutGraph.currentTime).toBe(37)
  expect(withoutGraph.getAttribute('src')).toBe(images[0].url)
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenLastCalledWith(expect.objectContaining({ bounds: { maxWidth: 1280, maxHeight: 720 } }))
  expect(localStorage.getItem('video_quality_preference')).toBe('720p')
  expect(mocks.handleEncodedQuality).not.toHaveBeenCalled()
})

// AC: @local-decoded-video-resolution ac-runtime-proof
test.each(['stock WebKit', 'endpoint-only WebKit'])('SVP Off with %s visibly restores Original when scaler proof is missing', async runtime => {
  localStorage.setItem('video_quality_preference', '720p')
  mocks.bridge.verifyLocalVideoResolution.mockRejectedValue(new Error(`Native video geometry is not ready; ${runtime}`))
  const { container } = mount()
  const first = await videoReady(container)
  fireEvent.loadedMetadata(first)
  await waitFor(() => expect(mocks.toast.error).toHaveBeenCalledWith(expect.stringContaining('Restoring Original')), { timeout: 3000 })
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first))
  const current = await videoReady(container)
  expect(current.getAttribute('src')).toBe(images[0].url)
  expect(mocks.bridge.verifyLocalVideoResolution).toHaveBeenCalledTimes(5)
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenLastCalledWith(expect.objectContaining({ bounds: null }))
  expect(localStorage.getItem('video_quality_preference')).toBe('original')
  expect(mocks.handleEncodedQuality).not.toHaveBeenCalled()
})

test('a successful old verifier acknowledgement is ignored while a later choice is withdrawing', async () => {
  localStorage.setItem('video_quality_preference', '720p')
  let acknowledge, withdraw
  mocks.bridge.verifyLocalVideoResolution.mockImplementationOnce(() => new Promise(resolve => { acknowledge = resolve }))
  const { container } = mount()
  const first = await videoReady(container)
  fireEvent.loadedData(first)
  await waitFor(() => expect(acknowledge).toBeTypeOf('function'))
  mocks.bridge.updateSvpManagerPlayback.mockImplementationOnce(() => new Promise(resolve => { withdraw = resolve }))
  choose('480p')
  await act(async () => acknowledge(true))
  expect(container.querySelector('video')).toBe(first)
  expect(container.querySelector('.lightbox-video-loading-grid')).not.toBeNull()
  await act(async () => withdraw())
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first))
  await videoReady(container)
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenLastCalledWith(expect.objectContaining({ bounds: { maxWidth: 854, maxHeight: 480 } }))
})

// AC: @local-decoded-video-resolution ac-remote
test('switching backend gives a remote video a new physical host without local resize preparation', async () => {
  localStorage.setItem('video_quality_preference', '720p')
  const { container, rerender } = mount()
  const first = await videoReady(container)
  const prepared = mocks.bridge.configureLocalVideoResolution.mock.calls.length
  mocks.embedded = false
  rerender(<Lightbox images={images} currentIndex={0} onClose={mocks.noop} onNav={mocks.noop} />)
  const remote = container.querySelector('video')
  expect(remote).not.toBe(first)
  expect(remote.id).not.toBe(first.id)
  expect(remote.getAttribute('src')).toBeNull()
  expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenCalledTimes(prepared)
})

// AC: @native-svp-startup-gate ac-original
test('missing resize and Manager runtime offers explicit original playback at the same position', async () => {
  mocks.initialSvp = true
  localStorage.setItem('video_quality_preference', '720p')
  mocks.bridge.verifyLocalVideoResolution.mockRejectedValue(new Error('Native video geometry is not ready; stock runtime'))
  mocks.bridge.updateSvpManagerPlayback.mockImplementation(async update => {
    if (update.enabled) throw new Error('Native SVP video host is not registered; stock runtime')
  })
  const { container } = mount()
  const first = await videoReady(container)
  first.currentTime = 37; first.__paused = true
  fireEvent.loadedMetadata(first)
  await waitFor(() => expect(container.querySelector('video')).not.toBe(first), { timeout: 3000 })
  const waiting = await videoReady(container)
  fireEvent.loadedMetadata(waiting)
  fireEvent.canPlay(waiting)
  await waitFor(() => expect(screen.getByText('Play without SVP')).toBeDefined(), { timeout: 3000 })
  expect(waiting.play).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('Play without SVP'))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(waiting))
  const original = await videoReady(container)
  original.__paused = true
  fireEvent.loadedMetadata(original)
  fireEvent.seeked(original)
  expect(original.currentTime).toBe(37)
  expect(original.paused).toBe(false)
  expect(mocks.handleEncodedQuality).not.toHaveBeenCalled()
})

// AC: @responsive-original-stream-seeking ac-rejected-seek, ac-owner-boundary
test('original error is visible, stops the loading grid and offers the existing close/reopen recovery', async () => {
  mocks.playback.playbackError = 'Video playback failed. Close and reopen this video to retry.'
  const close = vi.fn()
  const view = render(<Lightbox images={images} currentIndex={0} onClose={close} onNav={mocks.noop} />)
  const video = await videoReady(view.container)
  fireEvent.error(video)
  expect(mocks.playback.handlePlaybackError).toHaveBeenCalledWith(video)
  expect(screen.getByText(/Close and reopen this video/)).toBeTruthy()
  expect(view.container.querySelector('.lightbox-video-loading-grid')).toBeNull()
  const closeVideo = screen.getByRole('button', { name: 'Close video' })
  fireEvent.click(closeVideo)
  expect(close).toHaveBeenCalledOnce()
  mocks.playback.playbackError = null
  view.rerender(<Lightbox images={images} currentIndex={1} onClose={close} onNav={mocks.noop} />)
  await videoReady(view.container, 1)
  expect(screen.queryByRole('button', { name: 'Close video' })).toBeNull()
  fireEvent.click(closeVideo) // A detached recovery action cannot close the next item.
  expect(close).toHaveBeenCalledOnce()
})

// AC: @native-svp-startup-gate ac-no-early-play
test('native SVP preloads and registers without autoplay or canplay starting audio', async () => {
  mocks.initialSvp = true
  const { container } = mount()
  const video = await videoReady(container)
  video.__paused = true
  expect(video.autoplay).toBe(false)
  fireEvent.loadedMetadata(video)
  fireEvent.loadedData(video)
  fireEvent.canPlay(video)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  expect(video.play).not.toHaveBeenCalled()
  expect(video.paused).toBe(true)
  expect(container.querySelector('.lightbox-video-loading-grid')).not.toBeNull()
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  // Physical preroll is paused, but Manager must see the requested autoplay intent.
  expect(owner.paused).toBe(false)
  fireEvent.play(video) // Guard unexpected native/backend play as well.
  fireEvent.pause(video)
  expect(video.pause).toHaveBeenCalled()
  expect(mocks.playback.handleVideoPlay).not.toHaveBeenCalled()
  expect(mocks.playback.handleVideoPause).not.toHaveBeenCalled()
  await act(async () => mocks.listeners.onPaused({ ...owner, paused: true }))
  expect(video.play).not.toHaveBeenCalled()
})

// AC: @native-svp-startup-gate ac-owned-handoff
test.each([false, true])('owned filtered startup restores position/audio and intended paused=%s', async paused => {
  mocks.initialSvp = true
  mocks.playback.isPlaying = !paused
  mocks.playback.volume = 0.35; mocks.playback.isMuted = true
  const { container } = mount()
  const original = await videoReady(container)
  original.__paused = true
  original.currentTime = 37; original.volume = 0.175; original.muted = true
  fireEvent.loadedMetadata(original)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  // A paused bootstrap report/echo must not replace the user's playing intent.
  await act(async () => mocks.listeners.onPaused({ ...owner, paused: true }))
  await act(async () => mocks.listeners.onFilterChanged({ ...owner, enabled: true }))
  await act(async () => mocks.listeners.onPaused({ ...owner, paused: true }))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(original))
  const filtered = await videoReady(container)
  filtered.__paused = true
  expect(filtered.autoplay).toBe(false)
  fireEvent.loadedMetadata(filtered)
  expect(filtered.currentTime).toBe(37)
  fireEvent.canPlay(filtered)
  expect(filtered.play).not.toHaveBeenCalled()
  fireEvent.seeked(filtered)
  expect(filtered.paused).toBe(paused)
  expect(filtered.volume).toBe(0.35)
  expect(filtered.muted).toBe(true)
  expect(container.querySelector('.lightbox-video-loading-grid')).toBeNull()
})

// AC: @native-svp-startup-gate ac-missing-fps
test('paused startup probes unknown FPS and reports missing FPS with an explicit fallback', async () => {
  mocks.initialSvp = true
  mocks.dimensions.mockResolvedValue({ width: 1920, height: 1080 })
  const unknown = { ...images[0], video_fps: null, is_local_direct_file: true }
  const { container } = render(<Lightbox images={[unknown]} currentIndex={0} onClose={mocks.noop} onNav={mocks.noop} />)
  await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toBe(unknown.url))
  const video = container.querySelector('video')
  video.__paused = true
  video.requestVideoFrameCallback = vi.fn()
  fireEvent.loadedMetadata(video)
  await waitFor(() => expect(mocks.dimensions).toHaveBeenCalledWith(unknown.file_path))
  await waitFor(() => expect(screen.getByRole('status').textContent).toMatch(/frame rate/i))
  expect(screen.getByText('Play without SVP')).toBeDefined()
  expect(video.requestVideoFrameCallback).not.toHaveBeenCalled()
  expect(video.play).not.toHaveBeenCalled()
})

// AC: @native-svp-startup-gate ac-original
test('ordinary startup retains autoplay and canplay behavior', async () => {
  const { container } = mount()
  const video = await videoReady(container)
  expect(video.autoplay).toBe(true)
  fireEvent.canPlay(video)
  expect(video.play).toHaveBeenCalledOnce()
})

// AC: @native-svp-startup-gate ac-missing-fps
test('unknown FPS is probed successfully without advancing the unfiltered bootstrap', async () => {
  mocks.initialSvp = true
  const unknown = { ...images[0], video_fps: null }
  const { container } = render(<Lightbox images={[unknown]} currentIndex={0} onClose={mocks.noop} onNav={mocks.noop} />)
  const video = await videoReady(container)
  video.__paused = true
  fireEvent.loadedMetadata(video)
  fireEvent.canPlay(video)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled && u.fps === 24)).toBe(true))
  expect(mocks.dimensions).toHaveBeenCalledWith(unknown.file_path)
  expect(video.play).not.toHaveBeenCalled()
})

// AC: @native-svp-startup-gate ac-original
test.each([false, true])('cancel pending SVP restores intended paused=%s without a new canplay', async paused => {
  mocks.initialSvp = true
  mocks.playback.isPlaying = !paused
  const { container } = mount()
  const pending = await videoReady(container)
  pending.__paused = true; pending.currentTime = 37
  fireEvent.loadedMetadata(pending)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  const oldOwner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  fireEvent.click(screen.getByTitle('Disable SVP interpolation'))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(pending))
  const original = await videoReady(container)
  original.__paused = true
  fireEvent.loadedMetadata(original)
  fireEvent.seeked(original)
  expect(original.currentTime).toBe(37)
  expect(original.paused).toBe(paused)
  await act(async () => mocks.listeners.onFilterChanged({ ...oldOwner, enabled: true }))
  expect(container.querySelector('video')).toBe(original)
  expect(container.querySelector('.lightbox-video-loading-grid')).toBeNull()
})

// AC: @native-svp-startup-gate ac-owned-handoff
test('late filter acknowledgment cannot start a navigated or superseded physical player', async () => {
  mocks.initialSvp = true
  const { container, rerender } = mount()
  const old = await videoReady(container)
  old.__paused = true
  fireEvent.loadedMetadata(old)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  rerender(<Lightbox images={images} currentIndex={1} onClose={mocks.noop} onNav={mocks.noop} />)
  const current = await videoReady(container, 1)
  current.__paused = true
  await act(async () => mocks.listeners.onFilterChanged({ ...owner, enabled: true }))
  fireEvent.canPlay(current)
  expect(container.querySelector('video')).toBe(current)
  expect(current.play).not.toHaveBeenCalled()
})

// AC: @native-svp-startup-gate ac-no-early-play
test('direct-file source waits for persisted SVP selection before attaching', async () => {
  mocks.initialSvp = true
  mocks.configLoaded = false
  const direct = { ...images[0], is_local_direct_file: true }
  const { container, rerender } = render(<Lightbox images={[direct]} currentIndex={0} onClose={mocks.noop} onNav={mocks.noop} />)
  await waitFor(() => expect(mocks.bridge.configureLocalVideoResolution).toHaveBeenCalled())
  expect(container.querySelector('video').getAttribute('src')).toBeNull()
  mocks.configLoaded = true
  rerender(<Lightbox images={[direct]} currentIndex={0} onClose={mocks.noop} onNav={mocks.noop} />)
  const video = await videoReady(container)
  expect(video.autoplay).toBe(false)
  fireEvent.canPlay(video)
  expect(video.play).not.toHaveBeenCalled()
})

// AC: @native-svp-startup-gate ac-no-early-play
test('filtered preroll failure remains paused until explicit fallback', async () => {
  mocks.initialSvp = true
  const { container } = mount()
  const original = await videoReady(container)
  original.__paused = true
  fireEvent.loadedMetadata(original)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  await act(async () => mocks.listeners.onFilterChanged({ ...owner, enabled: true }))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(original))
  const filtered = await videoReady(container)
  filtered.__paused = true
  fireEvent.error(filtered)
  fireEvent.canPlay(filtered)
  expect(filtered.play).not.toHaveBeenCalled()
  expect(screen.getByText('Play without SVP')).toBeDefined()
  expect(screen.getByRole('status').textContent).toMatch(/could not load/i)
})

// AC: @native-svp-startup-gate ac-owned-handoff
test('initial zero-position filtered preroll starts once after acknowledgment, not before', async () => {
  mocks.initialSvp = true
  const { container } = mount()
  const original = await videoReady(container)
  original.__paused = true
  fireEvent.loadedMetadata(original)
  fireEvent.canPlay(original)
  expect(original.play).not.toHaveBeenCalled()
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  await act(async () => mocks.listeners.onFilterChanged({ ...owner, enabled: true }))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(original))
  const filtered = await videoReady(container)
  filtered.__paused = true
  fireEvent.loadedMetadata(filtered)
  expect(filtered.play).not.toHaveBeenCalled()
  fireEvent.loadedData(filtered)
  expect(filtered.play).toHaveBeenCalledOnce()
  fireEvent.canPlay(filtered)
  expect(filtered.play).toHaveBeenCalledOnce()
  expect(container.querySelector('.lightbox-video-loading-grid')).toBeNull()
})

// AC: @native-svp-startup-gate ac-owned-handoff
test.each([true, false])('active graph refresh preserves Manager intent with resume command=%s', async resumeCommand => {
  mocks.initialSvp = true
  const { container } = mount()
  const bootstrap = await videoReady(container)
  bootstrap.__paused = true
  fireEvent.loadedMetadata(bootstrap)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.some(([u]) => u.enabled)).toBe(true))
  const firstOwner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  await act(async () => mocks.listeners.onFilterChanged({ ...firstOwner, enabled: true }))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(bootstrap))
  const active = await videoReady(container)
  active.__paused = true
  fireEvent.loadedMetadata(active)
  fireEvent.loadedData(active)
  expect(active.paused).toBe(false)
  active.currentTime = 37
  fireEvent.play(active)
  await waitFor(() => expect(mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0].hostId).toBe(active.id))
  const owner = mocks.bridge.updateSvpManagerPlayback.mock.calls.filter(([u]) => u.enabled).at(-1)[0]
  active.play.mockClear()
  await act(async () => mocks.listeners.onFilterChanged({ ...owner, enabled: true }))
  expect(active.paused).toBe(true)
  fireEvent.pause(active) // The physical handoff pause does not change UI intent.
  expect(mocks.playback.handleVideoPause).not.toHaveBeenCalled()
  // Established Manager commands retain their existing authority after startup.
  await act(async () => mocks.listeners.onPaused({ ...owner, paused: true }))
  if (resumeCommand) await act(async () => mocks.listeners.onPaused({ ...owner, paused: false }))
  await waitFor(() => expect(container.querySelector('video')).not.toBe(active))
  const replacement = await videoReady(container)
  replacement.__paused = true
  fireEvent.loadedMetadata(replacement)
  fireEvent.seeked(replacement)
  expect(replacement.currentTime).toBe(37)
  expect(replacement.paused).toBe(!resumeCommand)
  fireEvent.canPlay(replacement)
  expect(replacement.paused).toBe(!resumeCommand)
})
