import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useLayoutEffect, useRef } from 'react'

const api = vi.hoisted(() => ({
  fetchRelatedMusic: vi.fn(),
  getMediaUrl: vi.fn(path => path || ''),
}))
const native = vi.hoisted(() => ({
  isTauri: vi.fn(() => false),
  subscribe: vi.fn(),
  subscribeToEvents: vi.fn(),
}))
vi.mock('../../api', () => api)
vi.mock('../../tauriAPI', () => ({
  isTauri: native.isTauri,
  nativeVideoAPI: { subscribe: native.subscribe },
  videoControlAPI: { subscribeToEvents: native.subscribeToEvents },
}))

import { MusicPlayerProvider, useMusicPlayer } from './MusicPlayer'

const song = (id, artist = 'Artist A') => ({
  id, title: `Song ${id}`, artist, library_id: 'library-a',
  stream_url: `/api/music/tracks/${id}/file`, artwork_url: null,
})

function PlayerHarness({ immediateNext = false }) {
  const { session, playing, startSong, advance, queueTrack } = useMusicPlayer()
  const advanced = useRef(false)
  // A layout effect can press Next after the selected song renders but before
  // the recommendation prefetch effect has marked the request as loading.
  useLayoutEffect(() => {
    if (immediateNext && session?.current?.id === 1 && !advanced.current) {
      advanced.current = true
      advance()
    }
  }, [immediateNext, session, advance])
  return <>
    <button onClick={() => startSong(song(1))}>Start song</button>
    <button onClick={advance}>Skip</button>
    <button onClick={() => queueTrack(song(80))}>Queue song 80</button>
    <button onClick={() => queueTrack(song(81))}>Queue song 81</button>
    <output data-testid="current">{session?.current?.id || ''}</output>
    <output data-testid="seed">{session?.seed?.id || ''}</output>
    <output data-testid="playing">{String(playing)}</output>
    <output data-testid="notice">{session?.notice || ''}</output>
    <output data-testid="queued">{session?.explicitQueue.map(track => track.id).join(',') || ''}</output>
    <output data-testid="related">{session?.recommendations.map(track => track.id).join(',') || ''}</output>
    <output data-testid="playback-error">{session?.playbackError || ''}</output>
  </>
}

beforeEach(() => {
  vi.clearAllMocks()
  sessionStorage.clear()
  native.isTauri.mockReturnValue(false)
  native.subscribe.mockResolvedValue(() => {})
  native.subscribeToEvents.mockResolvedValue(() => {})
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
})

afterEach(() => cleanup())

test('immediate Next waits for local recommendations and retains the selected mix seed', async () => {
  let resolveRelated
  api.fetchRelatedMusic.mockImplementation(() => new Promise(resolve => { resolveRelated = resolve }))
  render(<MusicPlayerProvider><PlayerHarness immediateNext /></MusicPlayerProvider>)
  fireEvent.click(screen.getByRole('button', { name: 'Start song' }))
  await waitFor(() => expect(api.fetchRelatedMusic).toHaveBeenCalledTimes(1))
  expect(screen.getByTestId('current').textContent).toBe('1')
  expect(screen.getByTestId('notice').textContent).toBe('')
  await act(async () => { resolveRelated({ tracks: [song(2, 'Artist B'), song(3, 'Artist C')] }) })
  await waitFor(() => expect(screen.getByTestId('current').textContent).toBe('2'))
  expect(screen.getByTestId('seed').textContent).toBe('1')
  fireEvent.click(screen.getByRole('button', { name: 'Skip' }))
  expect(screen.getByTestId('current').textContent).toBe('3')
  expect(screen.getByTestId('seed').textContent).toBe('1')
})

test('native video playback pauses music without replacing its queue', async () => {
  native.isTauri.mockReturnValue(true)
  let onSnapshot
  native.subscribe.mockImplementation(async handlers => {
    onSnapshot = handlers.onSnapshot
    return () => {}
  })
  api.fetchRelatedMusic.mockResolvedValue({ tracks: [song(2, 'Artist B')] })
  render(<MusicPlayerProvider><PlayerHarness /></MusicPlayerProvider>)
  fireEvent.click(screen.getByRole('button', { name: 'Start song' }))
  await waitFor(() => expect(screen.getByTestId('playing').textContent).toBe('true'))
  await waitFor(() => expect(onSnapshot).toBeTypeOf('function'))
  act(() => onSnapshot({ presentation: 'native_video', playback: { paused: false } }))
  expect(screen.getByTestId('playing').textContent).toBe('false')
  expect(screen.getByTestId('current').textContent).toBe('1')
  expect(screen.getByTestId('seed').textContent).toBe('1')
})

test('removing an explicitly queued song leaves the current song and recommendations intact', async () => {
  api.fetchRelatedMusic.mockResolvedValue({ tracks: [2, 3, 4, 5, 6, 7].map(id => song(id)) })
  render(<MusicPlayerProvider><PlayerHarness /></MusicPlayerProvider>)
  fireEvent.click(screen.getByRole('button', { name: 'Start song' }))
  await waitFor(() => expect(screen.getByTestId('related').textContent).toBe('2,3,4,5,6,7'))
  fireEvent.click(screen.getByRole('button', { name: 'Queue song 80' }))
  fireEvent.click(screen.getByRole('button', { name: 'Queue song 81' }))
  expect(screen.getByTestId('queued').textContent).toBe('80,81')
  fireEvent.click(screen.getByRole('button', { name: 'Remove Song 80 from queue' }))
  expect(screen.getByTestId('queued').textContent).toBe('81')
  expect(screen.getByTestId('current').textContent).toBe('1')
  expect(screen.getByTestId('seed').textContent).toBe('1')
  expect(screen.getByTestId('related').textContent).toBe('2,3,4,5,6,7')
  fireEvent.click(screen.getByRole('button', { name: 'Skip' }))
  expect(screen.getByTestId('current').textContent).toBe('81')
})

test('audio element errors pause playback and show a clear session notice', async () => {
  api.fetchRelatedMusic.mockResolvedValue({ tracks: [2, 3, 4, 5, 6, 7].map(id => song(id)) })
  const { container } = render(<MusicPlayerProvider><PlayerHarness /></MusicPlayerProvider>)
  fireEvent.click(screen.getByRole('button', { name: 'Start song' }))
  fireEvent.click(screen.getByRole('button', { name: 'Queue song 80' }))
  fireEvent.error(container.querySelector('audio'))
  expect(screen.getByTestId('playing').textContent).toBe('false')
  expect(screen.getByTestId('queued').textContent).toBe('80')
  expect(screen.getByTestId('playback-error').textContent).toMatch(/Could not play “Song 1”/)
  expect(screen.getByRole('alert').textContent).toMatch(/file may be missing or unsupported/)
})

test('a rejected audio play request reports an actionable playback notice', async () => {
  HTMLMediaElement.prototype.play.mockRejectedValueOnce(Object.assign(new Error('blocked'), { name: 'NotAllowedError' }))
  api.fetchRelatedMusic.mockResolvedValue({ tracks: [2, 3, 4, 5, 6, 7].map(id => song(id)) })
  render(<MusicPlayerProvider><PlayerHarness /></MusicPlayerProvider>)
  fireEvent.click(screen.getByRole('button', { name: 'Start song' }))
  await waitFor(() => expect(screen.getByTestId('playback-error').textContent).toMatch(/Press Play to try again/))
  expect(screen.getByTestId('playing').textContent).toBe('false')
  expect(screen.getByTestId('current').textContent).toBe('1')
})
