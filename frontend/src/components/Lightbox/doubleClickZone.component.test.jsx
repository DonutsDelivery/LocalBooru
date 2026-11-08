import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const mocks = vi.hoisted(() => {
  const noop = () => {}
  return {
    noop, initialSvp: false, embedded: true, configLoaded: true, realPlayback: false, listeners: null,
    bridge: {
      acquireSvpVideoHostEpoch: vi.fn(async () => 7),
      configureLocalVideoResolution: vi.fn(async () => true),
      verifyLocalVideoResolution: vi.fn(async () => true),
      updateSvpManagerPlayback: vi.fn(async () => {}),
      subscribeToSvpManager: vi.fn(async listeners => { mocks.listeners = listeners; return noop }),
    },
    toast: { error: vi.fn(), success: vi.fn() },
    handleEncodedQuality: vi.fn(async () => {}),
    sourceResolution: vi.fn(), dimensions: vi.fn(), gain: vi.fn(),
    ui: { showUI: true, isFullscreen: false, resetHideTimer: noop, handleMouseMove: noop,
      handleTouchInteractionStart: noop, consumeRevealTap: () => false, cancelRevealTap: noop,
      handleToggleFullscreen: vi.fn() },
    zoom: { zoom: { scale: 1 }, getZoomTransform: () => ({}), resetZoom: noop,
      isZoomDefault: () => mocks.zoom.zoom.scale === 1, calculateFillScale: () => 1.5, setZoom: vi.fn(),
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
  getAudioGain: (...args) => mocks.gain(...args), savePlaybackPosition: async () => {},
  getPlaybackPosition: async () => ({}), fetchCollections: async () => ({ collections: [] }),
  getSVPConfig: async () => ({ enabled: mocks.initialSvp }),
  updateSVPConfig: async config => config,
  toggleFavorite: vi.fn(), deleteImage: vi.fn(),
  addToCollection: mocks.noop, createCollection: mocks.noop, getShareNetworkInfo: mocks.noop, uploadImage: mocks.noop,
}))
vi.mock('../../tauriAPI', () => ({ getDesktopAPI: () => mocks.bridge }))
vi.mock('../../serverManager', () => ({ isMobileApp: () => false }))
vi.mock('../Toast', () => ({ toast: mocks.toast }))
vi.mock('../../hooks/useAddonStatus', () => ({ useAddonStatus: () => ({ installed: true }) }))
vi.mock('../../hooks/useImageWorkflow', () => ({ useImageWorkflow: () => ({ available: false }) }))
vi.mock('./hooks/useUIVisibility', () => ({ useUIVisibility: () => mocks.ui }))
vi.mock('./hooks/useZoomPan', () => ({ useZoomPan: () => mocks.zoom }))
vi.mock('./hooks/useVideoPlayback', async () => {
  const actual = await vi.importActual('./hooks/useVideoPlayback')
  return { useVideoPlayback: (ref, ...args) => {
    mocks.mediaRef = ref
    return mocks.realPlayback ? actual.useVideoPlayback(ref, ...args) : mocks.playback
  } }
})
vi.mock('./hooks/useWhisperSubtitles', () => ({ useWhisperSubtitles: () => mocks.subtitles }))
vi.mock('./hooks/useCastSession', () => ({ useCastSession: () => ({ isCasting: false }) }))
vi.mock('./hooks/useTimelinePreview', () => ({ useTimelinePreview: () => ({ previewFrames: [] }) }))
vi.mock('./hooks/useAutoAdvance', () => ({ useAutoAdvance: () => ({ countdown: null }) }))
vi.mock('./hooks/useShareStream', () => ({ useShareStream: () => ({}) }))
vi.mock('./VRVideoViewport', () => ({ default: () => null }))
vi.mock('../ContextMenu', () => ({ default: () => null }))
vi.mock('../SVPSideMenu', () => ({ default: () => null }))
vi.mock('./hooks/useVideoStreaming', async () => {
  const { useState } = await import('react')
  const { useAudioNormalization } = await vi.importActual('./hooks/useAudioNormalization')
  return { useVideoStreaming: ref => {
    const [svpConfig, setSvpConfig] = useState({ enabled: mocks.initialSvp })
    const audio = useAudioNormalization(ref)
    mocks.audio = audio
    return { nativeSvpPlayback: mocks.embedded, svpConfigLoaded: mocks.configLoaded, svpConfig, setSvpConfig,
      sourceResolution: { width: 1920, height: 1080 }, setSourceResolution: mocks.sourceResolution,
      streamTransitioningRef: { current: false }, opticalFlowConfig: { enabled: false },
      capturePlaybackIntent: () => ({ position: ref.current.currentTime, shouldPlay: !ref.current.paused }),
      handleQualityChange: mocks.handleEncodedQuality, setSvpError: mocks.noop, setSvpLoading: mocks.noop,
      checkCodecFallback: mocks.noop, getCurrentAbsoluteTime: () => ref.current?.currentTime ?? 0,
      setAudioOutputVolume: audio.setOutputVolume, setAudioOutputMuted: audio.setOutputMuted,
    }
  } }
})
import Lightbox from './Lightbox'

const video = { id: 1, filename: 'a.mp4', original_filename: 'a.mp4', file_path: '/a.mp4', url: '/a.mp4', video_fps: 24 }
const photo = { id: 2, filename: 'a.jpg', original_filename: 'a.jpg', file_path: '/a.jpg', url: '/a.jpg' }
beforeEach(() => {
  vi.clearAllMocks()
  mocks.zoom.zoom = { scale: 1 }
  mocks.dimensions.mockResolvedValue({ width: 1920, height: 1080, fps: 24 })
  mocks.gain.mockResolvedValue({ gain_db: 0 })
})
afterEach(cleanup)

function mount(image, curationMode) {
  const view = render(<Lightbox images={[image]} currentIndex={0} onClose={mocks.noop} onNav={mocks.noop} curationMode={curationMode} />)
  const root = view.container.querySelector('.lightbox')
  root.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 600, right: 1000, bottom: 600 })
  const media = view.container.querySelector('video')
  if (media) media.getBoundingClientRect = root.getBoundingClientRect
  return { root, media }
}
const dbl = (el, x) => fireEvent.doubleClick(el, { clientX: x })

test('video double-click toggles fullscreen only in the middle third', () => {
  const { media } = mount(video)
  ;[50, 320, 680, 950].forEach(x => dbl(media, x))
  expect(mocks.ui.handleToggleFullscreen).not.toHaveBeenCalled()
  dbl(media, 500)
  expect(mocks.ui.handleToggleFullscreen).toHaveBeenCalledTimes(1)
})

test('image double-click zooms only in the middle 20%, never fullscreens', () => {
  const { root } = mount(photo)
  ;[50, 390, 610, 950].forEach(x => dbl(root, x))
  expect(mocks.zoom.setZoom).not.toHaveBeenCalled()
  dbl(root, 500)
  expect(mocks.zoom.setZoom).toHaveBeenCalledWith({ scale: 1.5, x: 0, y: 0 })
  expect(mocks.ui.handleToggleFullscreen).not.toHaveBeenCalled()
})

test('a zoomed desktop image resets on a double-click anywhere', () => {
  mocks.zoom.zoom = { scale: 2 }
  const { root } = mount(photo)
  dbl(root, 50)
  expect(mocks.zoom.setZoom).toHaveBeenCalledWith({ scale: 1, x: 0, y: 0 })
})

test('double-click does nothing in curation mode', () => {
  const curation = { busy: false, lastAction: null, onUndo: vi.fn(), onDiscard: vi.fn() }
  const { root } = mount(photo, curation)
  dbl(root, 500)
  expect(mocks.zoom.setZoom).not.toHaveBeenCalled()
  const v = mount(video, curation)
  dbl(v.media, 500)
  expect(mocks.ui.handleToggleFullscreen).not.toHaveBeenCalled()
})
