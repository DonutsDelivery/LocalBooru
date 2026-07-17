/**
 * Server Manager - handles multi-server support for mobile app
 * Uses localStorage on all platforms (works in Tauri WebView)
 */

const SERVERS_KEY = 'localbooru_servers'
const ACTIVE_SERVER_KEY = 'localbooru_active_server'

// Local embedded server constant (always available on Tauri mobile)
export const LOCAL_SERVER = {
  id: '__local__',
  name: 'This Device',
  url: null, // uses relative /api URLs, same as desktop
  isLocal: true,
}

// Check if running as a Tauri mobile app
export function isMobileApp() {
  return window.__TAURI_INTERNALS__ !== undefined &&
         /Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
}

// Check if running in any Tauri context (desktop or mobile)
export function isTauriApp() {
  return typeof window !== 'undefined' && window.__TAURI_INTERNALS__ !== undefined
}

export function isLinuxDesktopApp() {
  return isTauriApp() && !isMobileApp() && /Linux/i.test(navigator.userAgent)
}

export function isWindowsOrMacDesktopApp() {
  return isTauriApp() && !isMobileApp() && /Windows|Macintosh|Mac OS X/i.test(navigator.userAgent)
}

// Default server structure
function createServer(data) {
  return {
    id: data.id || crypto.randomUUID(),
    name: data.name || 'LocalBooru Server',
    url: data.url,
    fallbackUrl: data.fallbackUrl || null,  // Optional secondary URL (e.g. Tailscale) used when primary fails
    username: data.username || null,
    password: data.password || null,
    token: data.token || null,  // JWT token from QR pairing
    certFingerprint: data.certFingerprint || null,  // TLS certificate fingerprint for pinning
    lastConnected: data.lastConnected || null,
  }
}

// Storage helpers — always use localStorage (works in Tauri WebView, persistent across sessions)
async function getStorageItem(key) {
  return localStorage.getItem(key)
}

async function setStorageItem(key, value) {
  localStorage.setItem(key, value)
}

// Get all saved servers
export async function getServers() {
  try {
    const data = await getStorageItem(SERVERS_KEY)
    return data ? JSON.parse(data) : []
  } catch (e) {
    console.error('Failed to get servers:', e)
    return []
  }
}

// Save servers list
export async function saveServers(servers) {
  await setStorageItem(SERVERS_KEY, JSON.stringify(servers))
}

// Add a new server
export async function addServer(serverData) {
  const servers = await getServers()
  const server = createServer(serverData)
  servers.push(server)
  await saveServers(servers)

  // If this is the first server, make it active
  if (servers.length === 1) {
    await setActiveServerId(server.id)
  }

  return server
}

// Add a server or update existing one if URL matches
export async function addOrUpdateServer(serverData) {
  const servers = await getServers()
  const normalizeUrl = (url) => url?.replace(/\/+$/, '')
  const existing = servers.find(s => normalizeUrl(s.url) === normalizeUrl(serverData.url))

  if (existing) {
    // Update the existing server entry with new token/name
    const updated = { ...existing, ...serverData, id: existing.id }
    const index = servers.indexOf(existing)
    servers[index] = updated
    await saveServers(servers)
    return updated
  }

  // No match — add as new
  const server = createServer(serverData)
  servers.push(server)
  await saveServers(servers)

  if (servers.length === 1) {
    await setActiveServerId(server.id)
  }

  return server
}

// Update an existing server
export async function updateServer(id, updates) {
  const servers = await getServers()
  const index = servers.findIndex(s => s.id === id)
  if (index !== -1) {
    servers[index] = { ...servers[index], ...updates }
    await saveServers(servers)
    return servers[index]
  }
  return null
}

// Remove a server
export async function removeServer(id) {
  const servers = await getServers()
  const filtered = servers.filter(s => s.id !== id)
  await saveServers(filtered)

  // If we removed the active server, switch to another
  const activeId = await getActiveServerId()
  if (activeId === id && filtered.length > 0) {
    await setActiveServerId(filtered[0].id)
  } else if (filtered.length === 0) {
    await setActiveServerId(null)
  }
}

// Get active server ID
export async function getActiveServerId() {
  return await getStorageItem(ACTIVE_SERVER_KEY)
}

// Set active server ID
export async function setActiveServerId(id) {
  if (id) {
    await setStorageItem(ACTIVE_SERVER_KEY, id)
  } else {
    localStorage.removeItem(ACTIVE_SERVER_KEY)
  }
}

// Get the currently active server
export async function getActiveServer() {
  const id = await getActiveServerId()
  if (!id) return null

  // Return the local server sentinel if selected
  if (id === LOCAL_SERVER.id) return LOCAL_SERVER

  const servers = await getServers()
  return servers.find(s => s.id === id) || null
}

// Test connection to a single URL. Returns:
//   { success: true } on HTTP success
//   { success: false, error, networkFailure: true } on network/timeout/refused
//   { success: false, error, networkFailure: false } on HTTP error response (server reachable)
export async function testServerConnection(url, username = null, password = null) {
  try {
    const headers = {}
    if (username && password) {
      headers['Authorization'] = 'Basic ' + btoa(`${username}:${password}`)
    }

    // Patch legacy saves that omitted the scheme.
    const target = /^https?:\/\//i.test(url) ? url : `http://${url}`
    const response = await fetch(`${target}/api`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000),
    })

    if (response.status === 401) {
      return { success: false, error: 'Authentication required', networkFailure: false }
    }

    if (!response.ok) {
      return { success: false, error: `Server returned ${response.status}`, networkFailure: false }
    }

    return { success: true }
  } catch (e) {
    if (e.name === 'AbortError' || e.name === 'TimeoutError') {
      return { success: false, error: 'Connection timeout', networkFailure: true }
    }
    return { success: false, error: e.message || 'Connection failed', networkFailure: true }
  }
}

// Probe a server's primary URL, then fall back to its secondary URL on network failure.
// Returns the working URL on success, plus the underlying probe result.
export async function probeServer(server) {
  if (!server || !server.url) return { success: false, error: 'No URL configured' }
  const primary = await testServerConnection(server.url, server.username, server.password)
  if (primary.success) {
    return { success: true, url: server.url, usedFallback: false }
  }
  // Only fall back on network-level failure; HTTP error means server is reachable.
  if (server.fallbackUrl && primary.networkFailure) {
    const fb = await testServerConnection(server.fallbackUrl, server.username, server.password)
    if (fb.success) {
      return { success: true, url: server.fallbackUrl, usedFallback: true }
    }
    // Both failed — return the fallback's error (most recent attempt)
    return { success: false, error: fb.error, networkFailure: fb.networkFailure }
  }
  return primary
}

// Get the API base URL
export async function getApiBaseUrl() {
  if (!isMobileApp()) {
    // Desktop: always use embedded server
    const isDevServer = window.location.port === '5173' || window.location.port === '5174'
    return isDevServer ? 'http://127.0.0.1:8790/api' : '/api'
  }

  // Mobile: check if using local or remote server
  const server = await getActiveServer()
  if (!server || server.id === LOCAL_SERVER.id) {
    // Local embedded server — use relative URL like desktop
    return '/api'
  }

  return `${server.url}/api`
}

// Ping all servers in parallel and return status map.
// Stashes the URL that responded onto the server entry as `_lastReachableUrl`
// so callers (auto-connect, updateServerConfig) can pick the right primary
// without paying for a second probe.
export async function pingAllServers(servers) {
  const results = await Promise.all(
    servers.map(async (server) => {
      const result = await probeServer(server)
      if (result.success && result.url) {
        try { await updateServer(server.id, { _lastReachableUrl: result.url }) } catch (e) { /* ignore */ }
      }
      return { id: server.id, online: result.success }
    })
  )
  return Object.fromEntries(results.map(r => [r.id, r.online ? 'online' : 'offline']))
}

// Get auth headers for the active server
export async function getAuthHeaders() {
  if (!isMobileApp()) {
    return {}
  }

  const server = await getActiveServer()
  if (!server || server.isLocal) return {}

  // Prefer JWT token (from QR pairing) over Basic auth
  if (server.token) {
    return { 'Authorization': 'Bearer ' + server.token }
  }

  if (server.username && server.password) {
    return { 'Authorization': 'Basic ' + btoa(`${server.username}:${server.password}`) }
  }

  return {}
}
