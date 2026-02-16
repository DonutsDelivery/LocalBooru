import { useState, useEffect } from 'react'
import { getAddons } from '../api'

// Shared cache so multiple components don't spam the API
let cachedAddons = null
let cacheTime = 0
let pendingFetch = null
const CACHE_TTL = 10000 // 10s

async function fetchAddonsCached() {
  const now = Date.now()
  if (cachedAddons && (now - cacheTime) < CACHE_TTL) {
    return cachedAddons
  }
  // Deduplicate concurrent requests
  if (!pendingFetch) {
    pendingFetch = getAddons()
      .then(data => {
        cachedAddons = data.addons || []
        cacheTime = Date.now()
        pendingFetch = null
        return cachedAddons
      })
      .catch(e => {
        console.error('Failed to load addon status:', e)
        pendingFetch = null
        return cachedAddons || []
      })
  }
  return pendingFetch
}

/**
 * Hook to check a single addon's status.
 * Returns { loading, installed, running, addon }
 */
export function useAddonStatus(addonId) {
  const [addon, setAddon] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetchAddonsCached().then(addons => {
      if (cancelled) return
      setAddon(addons.find(a => a.id === addonId) || null)
      setLoading(false)
    })
    return () => { cancelled = true }
  }, [addonId])

  return {
    loading,
    addon,
    installed: addon?.installed === true,
    running: addon?.status === 'running',
  }
}

/**
 * Hook to get all addon statuses at once.
 * Returns { loading, addons, isInstalled(id), isRunning(id) }
 */
export function useAllAddonStatuses() {
  const [addons, setAddons] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetchAddonsCached().then(list => {
      if (cancelled) return
      setAddons(list)
      setLoading(false)
    })
    return () => { cancelled = true }
  }, [])

  return {
    loading,
    addons,
    isInstalled: (id) => addons.some(a => a.id === id && a.installed),
    isRunning: (id) => addons.some(a => a.id === id && a.status === 'running'),
  }
}

/** Invalidate the cache (call after install/uninstall) */
export function invalidateAddonCache() {
  cachedAddons = null
  cacheTime = 0
}
