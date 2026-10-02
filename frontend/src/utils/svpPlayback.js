// One identity per loaded client, independent of the phone/desktop clock.
const clientSessionId = globalThis.crypto?.randomUUID?.()
  ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`

export function svpClientTransition(transitionId) {
  return { client_session_id: clientSessionId, client_transition_id: transitionId }
}

export function svpPlaybackError(error, fallback = 'Failed to start SVP playback') {
  const detail = error?.response?.data?.detail
  return typeof detail === 'string' && detail.trim()
    ? detail
    : error?.message || fallback
}
