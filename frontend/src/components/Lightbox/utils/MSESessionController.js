const waitForEvent = (target, name, signal) => new Promise((resolve, reject) => {
  const cleanup = () => {
    target.removeEventListener(name, onEvent)
    target.removeEventListener('error', onError)
    signal?.removeEventListener('abort', onAbort)
  }
  const onEvent = () => {
    cleanup()
    resolve()
  }
  const onError = () => {
    cleanup()
    reject(new Error(`${name} failed`))
  }
  const onAbort = () => {
    cleanup()
    reject(new DOMException(`${name} aborted`, 'AbortError'))
  }
  if (signal?.aborted) {
    onAbort()
    return
  }
  target.addEventListener(name, onEvent, { once: true })
  target.addEventListener('error', onError, { once: true })
  signal?.addEventListener('abort', onAbort, { once: true })
})

export class MSESessionController {
  constructor(video, client, options = {}) {
    this.video = video
    this.client = client
    this.mediaSourceFactory = options.mediaSourceFactory || (() => new MediaSource())
    this.createObjectURL = options.createObjectURL || (value => URL.createObjectURL(value))
    this.revokeObjectURL = options.revokeObjectURL || (value => URL.revokeObjectURL(value))
    this.pollDelay = options.pollDelay ?? 40
    this.maxQueueBytes = options.maxQueueBytes ?? 32 * 1024 * 1024
    this.keepBehind = options.keepBehind ?? 30
    this.onError = options.onError || (() => {})
    this.onBuffer = options.onBuffer || (() => {})
    this.onMetadata = options.onMetadata || (() => {})
    this.generation = 0
    this.serverGeneration = 0
    this.cursor = -1
    this.sessionId = null
    this.sourceBuffer = null
    this.mediaSource = null
    this.objectURL = null
    this.closed = true
    this.queuedBytes = 0
    this.abortController = null
    this.timestampOffset = 0
    this.pollPromise = null
    this.pendingReady = null
    this.playbackCommand = Promise.resolve()
    this.playbackCommandVersion = 0
  }

  async open(filePath, startPosition = 0, generation = 1) {
    await this.close()
    this.closed = false
    this.generation = generation
    this.cursor = -1
    this.timestampOffset = startPosition
    this.abortController = new AbortController()
    this.mediaSource = this.mediaSourceFactory()
    this.objectURL = this.createObjectURL(this.mediaSource)
    this.video.src = this.objectURL
    const sourceOpen = waitForEvent(this.mediaSource, 'sourceopen', this.abortController.signal)
    const opened = await this.client.open(filePath, startPosition, generation)
    if (this.closed || generation !== this.generation) {
      await this.client.stop(opened.session_id, generation).catch(() => {})
      throw new Error('SVP session was superseded')
    }
    this.sessionId = opened.session_id
    this.serverGeneration = generation
    await sourceOpen
    const ready = this.#deferred()
    this.pendingReady = ready
    this.pollPromise = this.#poll(generation, ready)
    try {
      await ready.promise
    } finally {
      if (this.pendingReady === ready) this.pendingReady = null
    }
    return opened
  }

  async pause() {
    return this.#queuePlaybackCommand('pause')
  }

  async resume() {
    return this.#queuePlaybackCommand('resume')
  }

  async seek(position) {
    if (!this.sessionId || this.closed) return
    this.pendingReady?.reject(new Error('SVP seek was superseded'))
    this.pendingReady = null
    const previousGeneration = this.generation
    const generation = previousGeneration + 1
    this.generation = generation
    this.timestampOffset = position
    this.abortController?.abort()
    this.abortController = new AbortController()
    await this.#clearBuffer()
    if (this.closed || generation !== this.generation) {
      throw new Error('SVP seek was superseded')
    }
    await this.client.seek(this.sessionId, previousGeneration, generation, position)
    this.serverGeneration = generation
    if (this.closed || generation !== this.generation) {
      await this.client.stop(this.sessionId, generation).catch(() => {})
      throw new Error('SVP seek was superseded')
    }
    const ready = this.#deferred()
    this.pendingReady = ready
    this.pollPromise = this.#poll(generation, ready)
    try {
      await ready.promise
    } finally {
      if (this.pendingReady === ready) this.pendingReady = null
    }
  }

  async close() {
    const sessionId = this.sessionId
    const serverGeneration = this.serverGeneration
    const generation = this.generation
    const objectURL = this.objectURL
    this.closed = true
    this.playbackCommandVersion += 1
    this.generation += 1
    const pendingReady = this.pendingReady
    this.pendingReady = null
    pendingReady?.reject(new Error('SVP session closed'))
    this.abortController?.abort()
    this.abortController = null
    this.sessionId = null
    this.sourceBuffer = null
    this.objectURL = null
    this.mediaSource = null
    this.queuedBytes = 0

    if (objectURL) this.revokeObjectURL(objectURL)
    if (objectURL && this.video.src === objectURL) {
      this.video.removeAttribute('src')
      this.video.load?.()
    }

    if (sessionId) {
      const stopGenerations = serverGeneration === generation
        ? [serverGeneration]
        : [serverGeneration, generation]
      for (const stopGeneration of stopGenerations) {
        try {
          await this.client.stop(sessionId, stopGeneration)
          break
        } catch {
          // A seek response can fail after the server changed generation.
        }
      }
    }
  }

  async #poll(generation, ready) {
    let emptyDelay = this.pollDelay
    try {
      while (!this.closed && generation === this.generation) {
        const batch = await this.client.events(this.sessionId, this.cursor)
        if (this.closed || generation !== this.generation) return
        if (batch.events.length) emptyDelay = this.pollDelay
        for (const event of batch.events) {
          this.cursor = Math.max(this.cursor, event.event_sequence)
          if (event.generation !== generation) continue
          if (event.type === 'metadata') {
            this.onMetadata(event.metadata)
            await this.#ensureSourceBuffer(event.metadata.mime_type)
          } else if (event.type === 'init_ready') {
            await this.#appendDescriptor(event.init_segment, true)
            ready.resolve()
          } else if (event.type === 'segment_ready') {
            await this.#appendDescriptor(event.segment, false)
            await this.#pruneBuffer()
          } else if (event.type === 'buffer_state') {
            this.onBuffer(event.buffered_duration, event.buffered_bytes)
          } else if (event.type === 'terminal_error') {
            throw new Error(event.message || 'SVP processing failed')
          } else if (event.type === 'ended') {
            await this.#waitForIdle()
            if (this.mediaSource?.readyState === 'open') this.mediaSource.endOfStream()
            return
          }
        }
        if (!batch.events.length) {
          await new Promise(resolve => setTimeout(resolve, emptyDelay))
          emptyDelay = Math.min(emptyDelay * 2, 500)
        }
      }
    } catch (error) {
      ready.reject(error)
      if (!this.closed && generation === this.generation) this.onError(error)
    }
  }

  async #ensureSourceBuffer(mimeType) {
    if (this.sourceBuffer) return
    const constructor = this.mediaSource.constructor
    if (constructor.isTypeSupported && !constructor.isTypeSupported(mimeType)) {
      throw new Error(`MSE codec is not supported: ${mimeType}`)
    }
    this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType)
    this.sourceBuffer.mode = 'segments'
  }

  async #appendDescriptor(descriptor, isInit) {
    if (!this.sourceBuffer) throw new Error('SVP metadata was not received before media')
    const generation = this.generation
    const signal = this.abortController.signal
    const data = await this.client.segment(
      this.sessionId,
      generation,
      descriptor.filename,
      signal
    )
    if (this.closed || generation !== this.generation) return
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data)
    if (bytes.byteLength > this.maxQueueBytes || this.queuedBytes + bytes.byteLength > this.maxQueueBytes) {
      throw new Error('SVP append queue exceeded its byte limit')
    }
    this.queuedBytes += bytes.byteLength
    try {
      await this.#waitForIdle(signal)
      this.sourceBuffer.timestampOffset = this.timestampOffset
      this.sourceBuffer.appendBuffer(bytes)
      await waitForEvent(this.sourceBuffer, 'updateend', signal)
    } finally {
      this.queuedBytes -= bytes.byteLength
    }
    if (isInit) {
      await this.client.ackInit(this.sessionId, generation)
    } else {
      await this.client.ackMedia(this.sessionId, generation, descriptor.sequence)
    }
  }

  async #clearBuffer() {
    if (!this.sourceBuffer) return
    await this.#waitForIdle()
    if (this.sourceBuffer.buffered.length) {
      const end = this.sourceBuffer.buffered.end(this.sourceBuffer.buffered.length - 1)
      this.sourceBuffer.remove(0, end)
      await waitForEvent(this.sourceBuffer, 'updateend', this.abortController?.signal)
    }
  }

  async #pruneBuffer() {
    if (!this.sourceBuffer || !this.sourceBuffer.buffered.length) return
    const removeEnd = this.video.currentTime - this.keepBehind
    if (removeEnd <= 0 || this.sourceBuffer.buffered.start(0) >= removeEnd) return
    await this.#waitForIdle()
    this.sourceBuffer.remove(0, removeEnd)
    await waitForEvent(this.sourceBuffer, 'updateend', this.abortController?.signal)
  }

  async #waitForIdle(signal = this.abortController?.signal) {
    if (this.sourceBuffer?.updating) await waitForEvent(this.sourceBuffer, 'updateend', signal)
  }

  #queuePlaybackCommand(command) {
    if (!this.sessionId || this.closed) return Promise.resolve()
    const sessionId = this.sessionId
    const generation = this.generation
    const version = this.playbackCommandVersion
    const queued = this.playbackCommand.catch(() => {}).then(async () => {
      if (this.closed
          || version !== this.playbackCommandVersion
          || sessionId !== this.sessionId
          || generation !== this.generation) return
      await this.client[command](sessionId, generation)
    })
    this.playbackCommand = queued
    return queued
  }

  #deferred() {
    let resolve
    let reject
    const promise = new Promise((onResolve, onReject) => {
      resolve = onResolve
      reject = onReject
    })
    return { promise, resolve, reject }
  }
}
