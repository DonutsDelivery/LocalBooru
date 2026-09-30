const CHANNEL = 'donut-create-basic-v1'

function abortError() {
  return new DOMException('The studio session changed. Reconnect to continue.', 'AbortError')
}

/** A request is scoped to one iframe, origin, and studio capability. */
export function createStudioBridge({ iframe, sessionId, url, signal }) {
  const origin = new URL(url, window.location.href).origin
  const pending = new Map()
  let disposed = false
  let sequence = 0
  const prefix = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`

  function finish(requestId, error, snapshot) {
    const request = pending.get(requestId)
    if (!request) return
    pending.delete(requestId)
    clearTimeout(request.timer)
    if (error) request.reject(error)
    else request.resolve(snapshot)
  }

  function receive(event) {
    if (disposed || event.source !== iframe.contentWindow || event.origin !== origin) return
    const message = event.data
    if (!message || message.channel !== CHANNEL || message.sessionId !== sessionId || !pending.has(message.requestId)) return
    if (message.ok === false) {
      finish(message.requestId, new Error(typeof message.error === 'string' ? message.error : 'The studio could not complete this action.'))
    } else if (message.ok === true && message.snapshot && typeof message.snapshot === 'object') {
      finish(message.requestId, null, message.snapshot)
    } else {
      finish(message.requestId, new Error('The studio returned an incomplete response. Retry the connection.'))
    }
  }

  function dispose() {
    if (disposed) return
    disposed = true
    window.removeEventListener('message', receive)
    signal?.removeEventListener('abort', dispose)
    for (const requestId of pending.keys()) finish(requestId, abortError())
  }

  function request(action, payload, { timeout = 15000 } = {}) {
    if (disposed || signal?.aborted) return Promise.reject(abortError())
    if (!iframe.contentWindow) return Promise.reject(new Error('The studio is still opening. Try again in a moment.'))
    const requestId = `${prefix}:${++sequence}`
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => finish(requestId, new Error(action === 'snapshot'
        ? 'The studio is taking longer to connect. Retry, or open the Advanced editor.'
        : 'The studio did not respond in time. Check the queue before trying again.')), timeout)
      pending.set(requestId, { resolve, reject, timer })
      try {
        iframe.contentWindow.postMessage({ channel: CHANNEL, sessionId, requestId, action, payload }, origin)
      } catch (error) {
        finish(requestId, error)
      }
    })
  }

  window.addEventListener('message', receive)
  signal?.addEventListener('abort', dispose, { once: true })
  if (signal?.aborted) dispose()
  return { request, dispose }
}
