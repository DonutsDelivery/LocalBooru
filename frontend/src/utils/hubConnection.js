// SPDX-License-Identifier: GPL-3.0-only

export function safeHubHttpsUrl(value) {
  try {
    const url = new URL(value)
    if (url.protocol !== 'https:' || url.username || url.password) return null
    return url.href
  } catch {
    return null
  }
}

export function validateHubReview(hub) {
  const descriptor = hub?.descriptor
  if (!descriptor || typeof descriptor.instance?.name !== 'string' || !descriptor.instance.name.trim()
    || typeof descriptor.instance.id !== 'string' || !safeHubHttpsUrl(descriptor.instance.origin)
    || !['public', 'members'].includes(descriptor.access?.catalog)
    || !['open', 'closed'].includes(descriptor.access?.registration)
    || typeof descriptor.access?.submission !== 'string' || typeof descriptor.access?.moderation !== 'string'
    || !Array.isArray(descriptor.authentication?.methods)
    || descriptor.authentication.methods.some(method => typeof method !== 'string')
    || !descriptor.capabilities || !['relayed', 'standard'].includes(hub.transport_mode?.toLowerCase())) {
    throw new Error('The community returned incomplete identity or access details. Ask its operator to update the community descriptor.')
  }
  return hub
}

// Enrollment links are bootstrap hints, never permission to save or sign in.
export function parseHubEnrollmentLink(value) {
  if (typeof value !== 'string' || value.length > 4096) throw new Error('Enter a valid Open in DMC community link.')
  let link
  try { link = new URL(value.trim()) } catch { throw new Error('Enter a valid Open in DMC community link.') }
  if (link.protocol !== 'donutmediacenter:' || link.hostname !== 'hub' || link.pathname !== '/add' || link.username || link.password || link.port || link.hash) {
    throw new Error('Expected a donutmediacenter://hub/add community link.')
  }
  const allowed = new Set(['url', 'collection', 'peer_id', 'relay_url'])
  for (const key of link.searchParams.keys()) {
    if (!allowed.has(key) || link.searchParams.getAll(key).length !== 1) throw new Error(`Unsupported or repeated link field: ${key}`)
  }
  const baseUrl = safeHubHttpsUrl(link.searchParams.get('url'))
  if (!baseUrl) throw new Error('The community link must contain a credential-free HTTPS URL.')
  const origin = new URL(baseUrl)
  if (origin.pathname !== '/' || origin.search || origin.hash) throw new Error('The community address must be an HTTPS origin without a path, query or fragment.')
  const peerId = link.searchParams.get('peer_id') || ''
  const relayValue = link.searchParams.get('relay_url') || ''
  const relayUrl = relayValue ? safeHubHttpsUrl(relayValue) : ''
  if (relayValue && !relayUrl) throw new Error('The relay must use a credential-free HTTPS URL.')
  if (peerId && !/^[a-zA-Z0-9]{32,128}$/.test(peerId)) throw new Error('The community peer identity is invalid.')
  const collection = link.searchParams.get('collection') || ''
  if (collection.length > 128 || /[\u0000-\u001f\u007f]/.test(collection)) throw new Error('The collection identifier is invalid.')
  return { base_url: baseUrl, peer_id: peerId, relay_url: relayUrl, collection }
}

export async function openHubBrowser(url) {
  const safeUrl = safeHubHttpsUrl(url)
  if (!safeUrl) throw new Error('The hub returned an unsafe browser address.')
  if (window.__TAURI_INTERNALS__) {
    const { open } = await import('@tauri-apps/plugin-shell')
    await open(safeUrl)
  } else {
    const opened = window.open(safeUrl, '_blank', 'noopener,noreferrer')
    if (opened) opened.opener = null
  }
}
