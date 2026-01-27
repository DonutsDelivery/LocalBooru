/**
 * Server Manager - handles multi-server support for mobile app
 * Uses Capacitor Preferences on mobile, localStorage on web
 */

import { Preferences } from '@capacitor/preferences'

const SERVERS_KEY = 'localbooru_servers'
const ACTIVE_SERVER_KEY = 'localbooru_active_server'

// Check if running in Capacitor native app (not web)
export function isMobileApp() {
  return window.Capacitor?.isNativePlatform?.() === true
}

// Default server structure
function createServer(data) {
  return {
    id: data.id || crypto.randomUUID(),
    name: data.name || 'LocalBooru Server',
    url: data.url,
    username: data.username || null,
    password: data.password || null,
    lastConnected: data.lastConnected || null,
  }
}

// Storage helpers that work on both web and mobile
async function getStorageItem(key) {
  if (isMobileApp()) {
    const { value } = await Preferences.get({ key })
    return value
  }
  return localStorage.getItem(key)
}

async function setStorageItem(key, value) {
  if (isMobileApp()) {
    await Preferences.set({ key, value })
  } else {
    localStorage.setItem(key, value)
  }
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
    if (isMobileApp()) {
      await Preferences.remove({ key: ACTIVE_SERVER_KEY })
    } else {
      localStorage.removeItem(ACTIVE_SERVER_KEY)
    }
  }
}

// Get the currently active server
export async function getActiveServer() {
  const id = await getActiveServerId()
  if (!id) return null

  const servers = await getServers()
  return servers.find(s => s.id === id) || null
}

// Test connection to a server
export async function testServerConnection(url, username = null, password = null) {
  try {
    const headers = {}
    if (username && password) {
      headers['Authorization'] = 'Basic ' + btoa(`${username}:${password}`)
    }

    const response = await fetch(`${url}/api`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000),
    })

    if (response.status === 401) {
      return { success: false, error: 'Authentication required' }
    }

    if (!response.ok) {
      return { success: false, error: `Server returned ${response.status}` }
    }

    return { success: true }
  } catch (e) {
    if (e.name === 'AbortError' || e.name === 'TimeoutError') {
      return { success: false, error: 'Connection timeout' }
    }
    return { success: false, error: e.message || 'Connection failed' }
  }
}

// Get the API base URL
export async function getApiBaseUrl() {
  // If not a mobile app, use relative URL (served from backend)
  if (!isMobileApp()) {
    const isDevServer = window.location.port === '5173' || window.location.port === '5174'
    return isDevServer ? 'http://127.0.0.1:8790/api' : '/api'
  }

  // Mobile app - use configured server
  const server = await getActiveServer()
  if (!server) {
    return null
  }

  return `${server.url}/api`
}

// Get auth headers for the active server
export async function getAuthHeaders() {
  if (!isMobileApp()) {
    return {}
  }

  const server = await getActiveServer()
  if (!server || !server.username || !server.password) {
    return {}
  }

  return {
    'Authorization': 'Basic ' + btoa(`${server.username}:${server.password}`)
  }
}
