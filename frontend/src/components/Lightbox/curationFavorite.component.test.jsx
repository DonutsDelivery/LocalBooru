import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

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
vi.mock('./hooks/useVideoGestures', () => ({ useVideoGestures: () => ({}) }))
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

const image = { id: 1, directory_id: 1, library_id: 'synthetic', filename: 'synthetic-1.jpg',
  original_filename: 'synthetic-1.jpg', file_path: '/synthetic-1.jpg', url: '/synthetic-1.jpg', is_favorite: false }
const favorited = { ...image, is_favorite: true }
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  mocks.dimensions.mockResolvedValue({ width: 1920, height: 1080, fps: 24 })
  mocks.gain.mockResolvedValue({ gain_db: 0 })
})
afterEach(cleanup)

function mount(item, overrides = {}) {
  const curationMode = { busy: false, lastAction: null, onKeep: vi.fn(), onNext: vi.fn(),
    onDiscard: vi.fn(), onUndo: vi.fn(), ...overrides }
  const view = render(<Lightbox images={[item]} currentIndex={0} total={1} onClose={mocks.noop}
    onNav={mocks.noop} curationMode={curationMode} />)
  return { curationMode, ...view }
}

test('an undecided item shows a plain Favorite button, no badge and no Next', () => {
  const { curationMode } = mount(image)
  const button = screen.getByRole('button', { name: /Favorite$/ })
  expect(button.getAttribute('aria-pressed')).toBe('false')
  expect(screen.queryByText(/Favorited/)).toBeNull()
  expect(screen.queryByRole('button', { name: /Next/ })).toBeNull()
  fireEvent.click(button)
  expect(curationMode.onKeep).toHaveBeenCalledOnce()
  fireEvent.keyDown(window, { key: 'Enter' })
  expect(curationMode.onNext).not.toHaveBeenCalled()
})

test('a favorited item is unmistakable: filled button, badge, Next button and Enter', () => {
  const { curationMode } = mount(favorited)
  const button = screen.getByRole('button', { name: /Favorited$/ })
  expect(button.getAttribute('aria-pressed')).toBe('true')
  expect(button.className).toContain('is-favorited')
  expect(button.title).toMatch(/^Remove favorite/)
  expect(screen.getByRole('status').textContent).toContain('Favorited')
  fireEvent.click(button)
  expect(curationMode.onKeep).toHaveBeenCalledOnce()
  fireEvent.click(screen.getByRole('button', { name: /Next/ }))
  expect(curationMode.onNext).toHaveBeenCalledOnce()
  fireEvent.keyDown(window, { key: 'Enter' })
  expect(curationMode.onNext).toHaveBeenCalledTimes(2)
  fireEvent.keyDown(window, { key: 'f' })
  expect(curationMode.onKeep).toHaveBeenCalledTimes(2)
})

test('the favorite display follows the current item when it changes', () => {
  const { rerender, curationMode } = mount(image)
  expect(screen.queryByRole('status')).toBeNull()
  rerender(<Lightbox images={[favorited]} currentIndex={0} total={1} onClose={mocks.noop}
    onNav={mocks.noop} curationMode={curationMode} />)
  expect(screen.getByRole('status')).toBeTruthy()
  rerender(<Lightbox images={[image]} currentIndex={0} total={1} onClose={mocks.noop}
    onNav={mocks.noop} curationMode={curationMode} />)
  expect(screen.queryByRole('status')).toBeNull()
})
