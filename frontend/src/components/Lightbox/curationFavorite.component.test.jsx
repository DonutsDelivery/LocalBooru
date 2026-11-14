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
    toast: { error: vi.fn(), success: vi.fn() }, favorite: vi.fn(),
    handleEncodedQuality: vi.fn(async () => {}),
    sourceResolution: vi.fn(), dimensions: vi.fn(), gain: vi.fn(),
    ui: { showUI: true, isFullscreen: false, resetHideTimer: noop, handleMouseMove: noop,
      handleTouchInteractionStart: noop, consumeRevealTap: () => false, cancelRevealTap: noop,
      handleToggleFullscreen: noop },
    zoom: { zoom: { scale: 1 }, getZoomTransform: () => ({}), resetZoom: noop,
      touchMoved: { current: false }, touchHandled: { current: false } },
    playback: { duration: 120, durationRef: { current: 120 }, currentTimeRef: { current: 0 },
      toggleVideoPlay: vi.fn(), isPlaying: true, volume: 1, videoDisplayMode: 'fit', playbackSpeed: 1,
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
  setFavorite: (...args) => mocks.favorite(...args), toggleFavorite: vi.fn(), deleteImage: vi.fn(),
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
  mocks.favorite.mockReset()
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

function gallery(item, onImageUpdate, onNav) {
  return <Lightbox images={[item]} currentIndex={0} total={1} onClose={mocks.noop}
    onNav={onNav} onImageUpdate={onImageUpdate} />
}

test('gallery favorite survives a stale refresh and toggles off with one click without navigating', async () => {
  mocks.favorite.mockImplementation(async (_item, next) => ({ is_favorite: next }))
  const update = vi.fn(), navigate = vi.fn()
  const view = render(gallery(image, update, navigate))
  fireEvent.click(screen.getByRole('button', { name: 'Add to favorites' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Remove from favorites' }).disabled).toBe(false))
  expect(update).toHaveBeenCalledOnce()
  view.rerender(gallery({ ...image, is_favorite: false }, update, navigate))
  expect(screen.getByRole('button', { name: 'Remove from favorites' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Remove from favorites' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Add to favorites' }).disabled).toBe(false))
  expect(mocks.favorite.mock.calls.map(call => call[1])).toEqual([true, false])
  expect(navigate).not.toHaveBeenCalled()
})

test.each([true, false])('late favorite completion (%s) cannot change another library with the same image id', async success => {
  let resolve, reject
  mocks.favorite.mockReturnValueOnce(new Promise((yes, no) => { resolve = yes; reject = no }))
  const update = vi.fn(), navigate = vi.fn()
  const view = render(gallery(image, update, navigate))
  fireEvent.click(screen.getByRole('button', { name: 'Add to favorites' }))
  await waitFor(() => expect(mocks.favorite).toHaveBeenCalledOnce())
  const other = { ...image, library_id: 'other-synthetic', directory_id: 2, url: '/other-synthetic.jpg', is_favorite: false }
  view.rerender(gallery(other, update, navigate))
  await act(async () => success ? resolve({ is_favorite: true }) : reject(new Error('synthetic favorite failure')))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Add to favorites' }).disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Remove from favorites' })).toBeNull()
  if (success) {
    expect(update).toHaveBeenCalledWith(expect.objectContaining({ libraryId: 'synthetic', directoryId: 1, imageId: 1 }), { is_favorite: true })
  } else expect(update).not.toHaveBeenCalled()
})

test('curation toolbar, central favorite and keyboard all use the queue handler', () => {
  const { curationMode } = mount(favorited)
  fireEvent.click(screen.getByRole('button', { name: 'Remove from favorites' }))
  fireEvent.click(screen.getByRole('button', { name: /Favorited$/ }))
  fireEvent.keyDown(window, { key: 'f' })
  expect(curationMode.onKeep).toHaveBeenCalledTimes(3)
  expect(mocks.favorite).not.toHaveBeenCalled()
})

test('curation video Space toggles playback without favoriting', () => {
  const { curationMode } = mount({ ...image, filename: 'synthetic.mp4', original_filename: 'synthetic.mp4' })
  fireEvent.keyDown(window, { key: ' ' })
  expect(mocks.playback.toggleVideoPlay).toHaveBeenCalledOnce()
  expect(curationMode.onKeep).not.toHaveBeenCalled()
  expect(mocks.favorite).not.toHaveBeenCalled()
})
