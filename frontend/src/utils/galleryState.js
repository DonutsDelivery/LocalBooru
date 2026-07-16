const MIN_RETRY_DELAY_MS = 250
const MAX_RETRY_DELAY_MS = 30_000

export function mergeFirstPage(existing, incoming) {
  const incomingIds = new Set(incoming.map(image => image.id))
  return [...incoming, ...existing.filter(image => !incomingIds.has(image.id))]
}

export function nextLoadRetryDelay(currentDelay, succeeded) {
  if (succeeded) return MIN_RETRY_DELAY_MS
  return Math.min(MAX_RETRY_DELAY_MS, Math.max(MIN_RETRY_DELAY_MS, currentDelay) * 2)
}

export function isUnexpectedEmptyPage({ append, pageLength, total, loaded }) {
  return append && pageLength === 0 && total > loaded
}
