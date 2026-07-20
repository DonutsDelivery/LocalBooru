import assert from 'node:assert/strict'
import test from 'node:test'

import { MSESessionController } from './Lightbox/utils/MSESessionController.js'

class FakeSourceBuffer extends EventTarget {
  constructor() {
    super()
    this.mode = null
    this.updating = false
    this.timestampOffset = 0
    this.appended = []
    this.buffered = {
      length: 0,
      start: () => 0,
      end: () => 0
    }
  }

  appendBuffer(bytes) {
    this.updating = true
    this.appended.push({ bytes: [...bytes], timestampOffset: this.timestampOffset })
    queueMicrotask(() => {
      this.updating = false
      this.dispatchEvent(new Event('updateend'))
    })
  }

  remove() {
    this.updating = true
    queueMicrotask(() => {
      this.updating = false
      this.dispatchEvent(new Event('updateend'))
    })
  }
}

class FakeMediaSource extends EventTarget {
  static isTypeSupported(mimeType) {
    return mimeType.includes('video/mp4')
  }

  constructor() {
    super()
    this.readyState = 'closed'
    this.sourceBuffer = null
    queueMicrotask(() => {
      this.readyState = 'open'
      this.dispatchEvent(new Event('sourceopen'))
    })
  }

  addSourceBuffer() {
    this.sourceBuffer = new FakeSourceBuffer()
    return this.sourceBuffer
  }

  endOfStream() {
    this.readyState = 'ended'
  }
}

const metadataEvent = generation => ({
  protocol_version: 1,
  event_sequence: generation * 10,
  session_id: 'session-1',
  generation,
  type: 'metadata',
  metadata: {
    source_duration: 120,
    width: 1920,
    height: 1080,
    source_fps: 24,
    output_fps: 48,
    mime_type: 'video/mp4; codecs="avc1.640034, mp4a.40.2"',
    initial_source_position: 0,
    max_av_drift_ms: 50,
    seekable: true
  }
})

const initEvent = generation => ({
  protocol_version: 1,
  event_sequence: generation * 10 + 1,
  session_id: 'session-1',
  generation,
  type: 'init_ready',
  init_segment: { generation, byte_length: 2, filename: `${generation}-init.mp4` }
})

const mediaEvent = (generation, sequence = 0) => ({
  protocol_version: 1,
  event_sequence: generation * 10 + 2 + sequence,
  session_id: 'session-1',
  generation,
  type: 'segment_ready',
  segment: {
    generation,
    sequence,
    source_start: 0,
    duration: 1,
    byte_length: 3,
    filename: `${generation}-${sequence}.m4s`,
    independent: true,
    av_drift_ms: 0
  }
})

const makeClient = () => {
  const calls = []
  const batches = new Map([
    [1, [metadataEvent(1), initEvent(1), mediaEvent(1)]],
    [2, [metadataEvent(2), initEvent(2), mediaEvent(2)]]
  ])
  let generation = 1
  return {
    calls,
    open: async () => ({ session_id: 'session-1', generation: 1, state: 'running' }),
    events: async () => {
      const events = batches.get(generation) || []
      batches.delete(generation)
      return { events, next_cursor: events.at(-1)?.event_sequence ?? -1 }
    },
    segment: async (_sessionId, currentGeneration, filename) => {
      calls.push(['segment', currentGeneration, filename])
      return filename.includes('init') ? Uint8Array.from([1, 2]) : Uint8Array.from([3, 4, 5])
    },
    ackInit: async (_sessionId, currentGeneration) => calls.push(['ackInit', currentGeneration]),
    ackMedia: async (_sessionId, currentGeneration, sequence) => calls.push(['ackMedia', currentGeneration, sequence]),
    pause: async () => {},
    resume: async () => {},
    seek: async (_sessionId, previousGeneration, nextGeneration, position) => {
      calls.push(['seek', previousGeneration, nextGeneration, position])
      generation = nextGeneration
    },
    stop: async (_sessionId, currentGeneration) => calls.push(['stop', currentGeneration])
  }
}

class FakeVideo extends EventTarget {
  constructor() {
    super()
    this.src = ''
    this.currentTime = 0
    this.paused = true
  }

  removeAttribute(name) {
    if (name === 'src') this.src = ''
  }

  load() {}
}

const makeVideo = () => new FakeVideo()

const nextTurn = () => new Promise(resolve => setTimeout(resolve, 5))

// AC: @svp-single-player ac-one-player
// AC: @svp-bounded-stream ac-transactional-seek
test('attaches one MediaSource and appends only current-generation media', async () => {
  const client = makeClient()
  const video = makeVideo()
  const controller = new MSESessionController(video, client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    pollDelay: 1
  })

  await controller.open('/media/video.mp4', 10, 1)
  await nextTurn()
  assert.equal(video.src, 'blob:test')
  assert.deepEqual(client.calls.slice(0, 4), [
    ['segment', 1, '1-init.mp4'],
    ['ackInit', 1],
    ['segment', 1, '1-0.m4s'],
    ['ackMedia', 1, 0]
  ])
  assert.deepEqual(controller.sourceBuffer.appended.map(item => item.timestampOffset), [10, 10])

  await controller.seek(40)
  await nextTurn()
  assert.deepEqual(client.calls.find(call => call[0] === 'seek'), ['seek', 1, 2, 40])
  assert.deepEqual(controller.sourceBuffer.appended.slice(-2).map(item => item.timestampOffset), [40, 40])
  assert.ok(client.calls.filter(call => call[0] === 'segment').every(call => call[1] >= 1))

  await controller.close()
  assert.equal(video.src, '')
})

test('advances the event cursor past stale generations', async () => {
  const client = makeClient()
  const cursors = []
  let pollCount = 0
  client.events = async (_sessionId, after) => {
    cursors.push(after)
    pollCount += 1
    if (pollCount === 1) {
      return { events: [{ ...metadataEvent(0), event_sequence: 5 }], next_cursor: 5 }
    }
    if (pollCount === 2) {
      return { events: [metadataEvent(1), initEvent(1)], next_cursor: 11 }
    }
    return { events: [], next_cursor: after }
  }

  const controller = new MSESessionController(makeVideo(), client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    pollDelay: 1
  })

  await controller.open('/media/video.mp4', 0, 1)
  assert.deepEqual(cursors.slice(0, 2), [-1, 5])
  await controller.close()
})

test('close rejects pending startup and remains idempotent', async () => {
  const client = makeClient()
  client.events = async () => ({ events: [], next_cursor: -1 })
  let revoked = 0
  const controller = new MSESessionController(makeVideo(), client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => { revoked += 1 },
    pollDelay: 1
  })

  const opening = assert.rejects(
    controller.open('/media/video.mp4', 0, 1),
    /SVP session closed/
  )
  await nextTurn()
  await controller.close()
  await opening
  await controller.close()

  assert.equal(revoked, 1)
  assert.equal(client.calls.filter(call => call[0] === 'stop').length, 1)
})

test('serializes pause and resume commands in request order', async () => {
  const client = makeClient()
  const commands = []
  let releasePause
  client.pause = async () => {
    commands.push('pause')
    await new Promise(resolve => { releasePause = resolve })
  }
  client.resume = async () => {
    commands.push('resume')
  }
  const controller = new MSESessionController(makeVideo(), client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    pollDelay: 1
  })

  await controller.open('/media/video.mp4', 0, 1)
  const pausing = controller.pause()
  const resuming = controller.resume()
  await nextTurn()
  assert.deepEqual(commands, ['pause'])
  releasePause()
  await Promise.all([pausing, resuming])
  assert.deepEqual(commands, ['pause', 'resume'])
  await controller.close()
})

test('close stops the accepted generation after a failed seek request', async () => {
  const client = makeClient()
  const stopCalls = []
  client.seek = async () => {
    throw new Error('request failed before acceptance')
  }
  client.stop = async (_sessionId, generation) => {
    stopCalls.push(generation)
    if (generation !== 1) throw new Error('generation conflict')
  }
  const controller = new MSESessionController(makeVideo(), client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    pollDelay: 1
  })

  await controller.open('/media/video.mp4', 0, 1)
  await assert.rejects(controller.seek(20), /request failed before acceptance/)
  await controller.close()

  assert.deepEqual(stopCalls, [1])
})

test('close retries the tentative generation when a seek response is lost', async () => {
  const client = makeClient()
  const stopCalls = []
  client.seek = async () => {
    throw new Error('response lost after acceptance')
  }
  client.stop = async (_sessionId, generation) => {
    stopCalls.push(generation)
    if (generation === 1) throw new Error('generation conflict')
  }
  const controller = new MSESessionController(makeVideo(), client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    pollDelay: 1
  })

  await controller.open('/media/video.mp4', 0, 1)
  await assert.rejects(controller.seek(20), /response lost after acceptance/)
  await controller.close()

  assert.deepEqual(stopCalls, [1, 2])
})

test('rejects media when the frontend append byte limit is exceeded', async () => {
  const client = makeClient()
  const errors = []
  const controller = new MSESessionController(makeVideo(), client, {
    mediaSourceFactory: () => new FakeMediaSource(),
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    maxQueueBytes: 1,
    onError: error => errors.push(error)
  })

  await assert.rejects(controller.open('/media/video.mp4', 0, 1), /append queue exceeded/)
  assert.equal(errors.length, 1)
  await controller.close()
})
