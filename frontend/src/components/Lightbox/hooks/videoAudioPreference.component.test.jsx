import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { useAudioNormalization } from './useAudioNormalization'
import { useVideoPlayback } from './useVideoPlayback'
import { readVideoAudioPreference, writeVideoAudioPreference } from './videoAudioPreference'

const mocks = vi.hoisted(() => ({ gain: vi.fn() }))
vi.mock('../../../api', () => ({ getAudioGain: mocks.gain, savePlaybackPosition: vi.fn() }))
function setup() {
  const video = { volume: 1, muted: false, readyState: 3, currentTime: 0, duration: 120,
    play: vi.fn(() => Promise.resolve()), pause: vi.fn(), playbackRate: 1 }
  const mediaRef = { current: video }
  const ready = { current: true }
  const hook = renderHook(() => {
    const audio = useAudioNormalization(mediaRef)
    const playback = useVideoPlayback(mediaRef, {
      streamTransitioningRef: { current: false }, interactionReadyRef: ready,
      setAudioOutputVolume: audio.setOutputVolume, setAudioOutputMuted: audio.setOutputMuted,
    })
    return { audio, playback }
  })
  return { ...hook, video, mediaRef, ready }
}
beforeEach(() => { localStorage.clear(); mocks.gain.mockReset(); mocks.gain.mockResolvedValue({ gain_db: 0 }) })
afterEach(() => { cleanup(); vi.restoreAllMocks() })

// AC: @logical-video-volume ac-restore
test('reopening restores the same logical volume and mute to controls and output', () => {
  let hook = setup()
  act(() => hook.result.current.playback.handleVolumeChange({ target: { value: '0.35' } }))
  act(() => hook.result.current.playback.toggleMute())
  hook.unmount()
  hook = setup()
  expect(hook.result.current.playback.volume).toBe(0.35)
  expect(hook.result.current.playback.isMuted).toBe(true)
  expect(hook.video.volume).toBe(0)
  expect(hook.video.muted).toBe(true)
  act(() => hook.result.current.playback.toggleMute())
  expect(hook.video.volume).toBe(0.35)
})

// AC: @logical-video-volume ac-normalization
test('attenuation changes output once without changing the remembered knob', async () => {
  writeVideoAudioPreference({ volume: 0.35, muted: false })
  const hook = setup()
  mocks.gain.mockResolvedValue({ gain_db: 20 * Math.log10(0.5) })
  await act(() => hook.result.current.audio.applyNormalization('/synthetic'))
  expect(hook.result.current.playback.volume).toBe(0.35)
  expect(hook.video.volume).toBeCloseTo(0.175)
  expect(readVideoAudioPreference().volume).toBe(0.35)
  act(() => hook.result.current.audio.resetGain())
  expect(hook.video.volume).toBe(0.35)
})

// AC: @logical-video-volume ac-normalization
test('physical replacement restores logical gain without treating attenuated DOM gain as user gain', async () => {
  const hook = setup()
  act(() => hook.result.current.playback.handleVolumeChange({ target: { value: '0.35' } }))
  mocks.gain.mockResolvedValue({ gain_db: 20 * Math.log10(0.5) })
  await act(() => hook.result.current.audio.applyNormalization('/synthetic'))
  const replacement = { ...hook.video, volume: 1, muted: false }
  hook.mediaRef.current = replacement
  act(() => hook.result.current.playback.restoreAudioState(0.35, true))
  expect(replacement.volume).toBe(0)
  act(() => hook.result.current.playback.restoreAudioState(0.35, false))
  expect(replacement.volume).toBeCloseTo(0.175)
  expect(hook.result.current.playback.volume).toBe(0.35)
})

// AC: @logical-video-volume ac-handoff
test('restored playback updates normalization logical state and defers play while startup is pending', async () => {
  const hook = setup()
  hook.ready.current = false
  act(() => hook.result.current.playback.restorePlaybackState({ volume: 0.4, muted: true, paused: false }))
  expect(hook.result.current.playback.isPlaying).toBe(true)
  expect(hook.video.play).not.toHaveBeenCalled()
  act(() => hook.result.current.audio.resetGain())
  expect(hook.video.volume).toBe(0)
  hook.ready.current = true
  act(() => hook.result.current.playback.restorePlaybackState({ volume: 0.4, muted: false, paused: false }))
  expect(hook.video.play).toHaveBeenCalledTimes(1)
  mocks.gain.mockResolvedValue({ gain_db: 20 * Math.log10(0.5) })
  await act(() => hook.result.current.audio.applyNormalization('/synthetic'))
  expect(hook.video.volume).toBeCloseTo(0.2)
  expect(hook.result.current.playback.volume).toBe(0.4)
})

// AC: @logical-video-volume ac-validation
test.each(['{', '{"volume":null,"muted":false}', '{"volume":"0.35","muted":"false"}', '{"volume":8,"muted":false}'])('invalid saved preference %s safely defaults', value => {
  localStorage.setItem('video_audio_preference', value)
  const hook = setup()
  expect(hook.result.current.playback.volume).toBe(1)
  expect(hook.result.current.playback.isMuted).toBe(false)
  expect(hook.video.volume).toBe(1)
})

// AC: @logical-video-volume ac-validation
test('blocked storage leaves volume and mute controls usable', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
  const hook = setup()
  act(() => hook.result.current.playback.handleVolumeChange({ target: { value: '0.65' } }))
  expect(hook.result.current.playback.volume).toBe(0.65)
  expect(hook.video.volume).toBe(0.65)
  act(() => hook.result.current.playback.toggleMute())
  expect(hook.video.muted).toBe(true)
})

// AC: @logical-video-volume ac-normalization
test('resetting normalization rejects a delayed old gain result', async () => {
  let resolve
  mocks.gain.mockImplementation(() => new Promise(done => { resolve = done }))
  const hook = setup()
  act(() => hook.result.current.playback.handleVolumeChange({ target: { value: '0.35' } }))
  let pending
  act(() => { pending = hook.result.current.audio.applyNormalization('/synthetic') })
  await waitFor(() => expect(resolve).toBeTypeOf('function'))
  act(() => hook.result.current.audio.resetGain())
  await act(async () => { resolve({ gain_db: -12 }); await pending })
  expect(hook.video.volume).toBe(0.35)
})
