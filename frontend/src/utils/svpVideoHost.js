let nextHostRevision = 0

export function createSVPVideoHostId() {
  const token = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`
  return `localbooru-svp-host-${token}`
}

export function nextSVPVideoHostRevision() {
  return ++nextHostRevision
}

export function svpVideoHostUpdate(owner, enabled, update = {}) {
  return { ...update, ...owner, enabled }
}

export function svpVideoHostOwnsEvent(owner, event) {
  if (!owner || owner.hostEpoch === null) return false
  // The native Linux relay uses host leases. Other desktop bridges retain
  // their existing events and return epoch 0.
  if (owner.hostEpoch === 0) return true
  return event?.hostId === owner.hostId
    && event?.hostEpoch === owner.hostEpoch
    && event?.hostRevision === owner.hostRevision
}

export async function publishSVPVideoHost(send, update, isCurrent, wait = delay => new Promise(resolve => setTimeout(resolve, delay))) {
  // Metadata can race the WebProcess registration during a physical remount.
  for (let attempt = 0; attempt < 5 && isCurrent(); attempt++) {
    try {
      await send(update)
      return true
    } catch (error) {
      if (!isCurrent()) return false
      if (!String(error?.message ?? error).includes('SVP video host is not registered') || attempt === 4) throw error
      await wait(250)
    }
  }
  return false
}
