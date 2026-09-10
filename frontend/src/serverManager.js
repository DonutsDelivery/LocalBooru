/**
 * Server Manager - handles multi-server support for mobile app
 * Stores public server metadata locally and credentials in native protected storage when available.
 */

const SERVERS_KEY = 'localbooru_servers'
const ACTIVE_SERVER_KEY = 'localbooru_active_server'
const LOCAL_SERVER_PORT = import.meta.env?.VITE_LOCALBOORU_PORT || '8790'
let serverMutationQueue = Promise.resolve()

function serializeServerMutation(operation) {
  const next = serverMutationQueue.then(operation, operation)
  serverMutationQueue = next.catch(() => {})
  return next
}

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

export function serverFromQrHandshake(qrData, workingUrl, handshake) {
  if (!handshake?.success || !handshake.token) {
    throw new Error(handshake?.error || 'The server did not issue a device credential')
  }
  if (typeof handshake.serverId !== 'string' || !handshake.serverId.trim()) {
    throw new Error('The server did not provide its stable identity with the device credential')
  }
  return {
    id: handshake.serverId,
    name: handshake.serverName || qrData.name || 'LocalBooru Server',
    url: workingUrl,
    token: handshake.token,
    username: null,
    password: null,
    certFingerprint: qrData.cert_fingerprint || null,
    lastConnected: new Date().toISOString(),
  }
}

// Public metadata helpers. Credentials are stripped before native-app writes.
async function getStorageItem(key) {
  return localStorage.getItem(key)
}

async function setStorageItem(key, value) {
  localStorage.setItem(key, value)
}

let cachedProtectedCredentials = undefined
let protectedCredentialsLoad = null

async function loadProtectedCredentials() {
  if (!isTauriApp()) return null
  if (cachedProtectedCredentials !== undefined) return cachedProtectedCredentials
  if (protectedCredentialsLoad) return protectedCredentialsLoad
  protectedCredentialsLoad = (async () => {
    try {
      if (window.AndroidCredentialStore?.load) {
        cachedProtectedCredentials = JSON.parse(window.AndroidCredentialStore.load())
        return cachedProtectedCredentials
      }
      const { invoke } = await import('@tauri-apps/api/core')
      cachedProtectedCredentials = await invoke('load_paired_server_credentials')
      return cachedProtectedCredentials
    } catch (error) {
      console.warn('[Servers] Protected credential store is unavailable:', error)
      cachedProtectedCredentials = null
      return null
    } finally {
      protectedCredentialsLoad = null
    }
  })()
  return protectedCredentialsLoad
}

async function storeProtectedCredentials(credentials) {
  if (!isTauriApp()) return false
  if (window.AndroidCredentialStore?.store) {
    if (!window.AndroidCredentialStore.store(JSON.stringify(credentials))) {
      throw new Error('Android protected credential store rejected the update')
    }
    cachedProtectedCredentials = credentials
    return true
  }
  const { invoke } = await import('@tauri-apps/api/core')
  await invoke('store_paired_server_credentials', { credentials })
  cachedProtectedCredentials = credentials
  return true
}

// Get all saved servers
export async function getServers() {
  try {
    const data = await getStorageItem(SERVERS_KEY)
    const servers = data ? JSON.parse(data) : []
    const credentials = await loadProtectedCredentials()
    if (!credentials) return servers
    const merged = servers.map(server => ({
      ...server,
      token: credentials[server.id]?.token || server.token || null,
      password: credentials[server.id]?.password || server.password || null,
    }))
    if (servers.some(server => server.token || server.password)) {
      await saveServers(merged)
    }
    return merged
  } catch (e) {
    console.error('Failed to get servers:', e)
    return []
  }
}

// Save servers list
export async function saveServers(servers) {
  if (isTauriApp()) {
    const credentials = Object.fromEntries(servers
      .filter(server => server.token || server.password)
      .map(server => [server.id, {
        ...(server.token ? { token: server.token } : {}),
        ...(server.password ? { password: server.password } : {}),
      }]))
    await storeProtectedCredentials(credentials)
    const metadata = servers.map(server => {
      const publicServer = { ...server }
      delete publicServer.token
      delete publicServer.password
      return publicServer
    })
    await setStorageItem(SERVERS_KEY, JSON.stringify(metadata))
    return
  }
  await setStorageItem(SERVERS_KEY, JSON.stringify(servers))
}

// Add a new server
export async function addServer(serverData) {
  return serializeServerMutation(async () => {
    const servers = await getServers()
    const server = createServer(serverData)
    servers.push(server)
    await saveServers(servers)

    // If this is the first server, make it active
    if (servers.length === 1) {
      await setActiveServerId(server.id)
    }

    return server
  })
}

// Add a server or update existing one if URL matches
export async function addOrUpdateServer(serverData) {
  return serializeServerMutation(async () => {
    if (!serverData.id || serverData.id === LOCAL_SERVER.id) {
      throw new Error('Paired server returned an invalid stable identity')
    }
    const servers = await getServers()
    const normalizeUrl = (url) => url?.replace(/\/+$/, '')
    const identityMatch = servers.find(server => server.id === serverData.id)
    const urlMatch = servers.find(server => normalizeUrl(server.url) === normalizeUrl(serverData.url))
    if (urlMatch && urlMatch.id !== serverData.id) {
      throw new Error('This address now identifies a different LocalBooru server')
    }
    if (identityMatch) {
      const updated = { ...identityMatch, ...serverData, id: identityMatch.id }
      servers[servers.indexOf(identityMatch)] = updated
      await saveServers(servers)
      return updated
    }
    const server = createServer(serverData)
    servers.push(server)
    await saveServers(servers)
    if (servers.length === 1) await setActiveServerId(server.id)
    return server
  })
}

// Update an existing server
export async function updateServer(id, updates) {
  return serializeServerMutation(async () => {
    const servers = await getServers()
    const index = servers.findIndex(s => s.id === id)
    if (index !== -1) {
      servers[index] = { ...servers[index], ...updates }
      await saveServers(servers)
      return servers[index]
    }
    return null
  })
}

// Remove a server
export async function removeServer(id) {
  return serializeServerMutation(async () => {
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
  })
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
export async function testServerConnection(url, username = null, password = null, token = null) {
  try {
    const headers = {}
    if (token) {
      headers['Authorization'] = 'Bearer ' + token
    } else if (username && password) {
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
  const primary = await testServerConnection(server.url, server.username, server.password, server.token)
  if (primary.success) {
    return { success: true, url: server.url, usedFallback: false }
  }
  // Only fall back on network-level failure; HTTP error means server is reachable.
  if (server.fallbackUrl && primary.networkFailure) {
    const fb = await testServerConnection(server.fallbackUrl, server.username, server.password, server.token)
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
  if (!isTauriApp()) {
    // Desktop: always use embedded server
    const isDevServer = window.location.port === '5173' || window.location.port === '5174'
    return isDevServer ? `http://127.0.0.1:${LOCAL_SERVER_PORT}/api` : '/api'
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
        server._lastReachableUrl = result.url
      }
      return { id: server.id, online: result.success }
    })
  )
  return Object.fromEntries(results.map(r => [r.id, r.online ? 'online' : 'offline']))
}

// Get auth headers for the active server
export async function getAuthHeaders() {
  if (!isTauriApp()) {
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
