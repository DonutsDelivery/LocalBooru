export function shouldSuppressOptionalNotFound(config, status) {
  return config?.suppressErrorToast === true && status === 404
}

export function isRemoteConnectionFailure(status, data) {
  const detail = typeof data === 'string' ? data : data?.detail
  return status === 502 && typeof detail === 'string' && /^Proxy error(?: \(primary \+ fallback\))?:/.test(detail)
}

// Every read endpoint can fail when one remote server goes offline.
export function createRemoteConnectionToastGate() {
  let lastNotified
  return (config, status, data, now = Date.now()) => {
    if (config?.method?.toLowerCase() !== 'get' || !isRemoteConnectionFailure(status, data)) return false
    if (lastNotified !== undefined && now - lastNotified < 600_000) return true
    lastNotified = now
    return false
  }
}

// An offline library can fail every gallery refresh. Keep the first error
// actionable without stacking another toast for every page or retry.
export function createUnavailableLibraryToastGate() {
  const lastNotified = new Map()
  return (config, status, detail, now = Date.now()) => {
    if (config?.method?.toLowerCase() !== 'get' || status !== 404
      || typeof detail !== 'string' || !/^Library '[^']+' not found or not mounted$/.test(detail)) {
      return false
    }
    const previous = lastNotified.get(detail)
    if (previous !== undefined && now - previous < 600_000) return true
    lastNotified.set(detail, now)
    return false
  }
}
