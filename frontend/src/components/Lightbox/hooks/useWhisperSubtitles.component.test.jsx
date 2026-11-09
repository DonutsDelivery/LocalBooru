import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useWhisperSubtitles } from './useWhisperSubtitles'
import { generateSubtitles, subscribeToSubtitleEvents } from '../../../api'

vi.mock('../../../api', () => ({
  getWhisperConfig: vi.fn(async () => ({ language: 'ja', task: 'translate' })),
  installWhisper: vi.fn(),
  generateSubtitles: vi.fn(),
  stopSubtitles: vi.fn(async () => ({ success: true })),
  subscribeToSubtitleEvents: vi.fn(),
}))

globalThis.IS_REACT_ACT_ENVIRONMENT = true

class FakeCue {
  constructor(start, end, text) {
    Object.assign(this, { start, end, text })
  }
}

function fakeVideo() {
  const tracks = []
  return {
    currentTime: 0,
    textTracks: tracks,
    addTextTrack(kind, label) {
      const track = { kind, label, mode: 'disabled', cues: [], addCue(cue) { this.cues.push(cue) } }
      tracks.push(track)
      return track
    },
  }
}

describe('useWhisperSubtitles sidecar events', () => {
  let container
  let root
  let hook
  let video
  let emit

  function Harness() {
    hook = useWhisperSubtitles({ current: video }, { id: 1, filename: 'clip.mp4', file_path: '/media/clip.mp4' })
    return null
  }

  beforeEach(async () => {
    globalThis.VTTCue = FakeCue
    video = fakeVideo()
    generateSubtitles.mockResolvedValue({ stream_id: 'stream-1', cached: false })
    subscribeToSubtitleEvents.mockImplementation((streamId, onEvent) => {
      emit = onEvent
      return () => {}
    })
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    await act(async () => root.render(<Harness />))
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    delete globalThis.VTTCue
  })

  it('shows model loading, adds cues and finishes on the flat sidecar events', async () => {
    await act(async () => hook.startSubtitles())
    expect(hook.generating).toBe(true)

    await act(async () => emit({ type: 'loading_model', stream_id: 'stream-1' }))
    expect(hook.loadingModel).toBe(true)

    await act(async () => emit({ type: 'started', stream_id: 'stream-1', duration: 8 }))
    expect(hook.loadingModel).toBe(false)

    await act(async () => emit({ type: 'cue', stream_id: 'stream-1', start: 1, end: 2, text: 'hello' }))
    await act(async () => emit({ type: 'progress', stream_id: 'stream-1', progress: 50 }))
    expect(video.textTracks[0].cues.map(cue => cue.text)).toEqual(['hello'])
    expect(hook.progress).toBe(50)

    await act(async () => emit({ type: 'completed', stream_id: 'stream-1' }))
    expect(hook.completed).toBe(true)
    expect(hook.generating).toBe(false)
  })

  it('surfaces a sidecar error', async () => {
    await act(async () => hook.startSubtitles())
    await act(async () => emit({ type: 'error', stream_id: 'stream-1', error: 'decode failed' }))
    expect(hook.error).toBe('decode failed')
    expect(hook.generating).toBe(false)
  })
})
