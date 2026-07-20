/**
 * React hook for desktop API access
 *
 * Provides access to the supported Tauri desktop environment.
 */
import { useState, useEffect } from 'react'
import { getDesktopAPI, isDesktopApp, isTauriApp } from '../tauriAPI'

/**
 * Hook to access the Tauri desktop API.
 * Returns null when running in a browser without desktop features
 */
export function useDesktopAPI() {
  const [api, setApi] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const desktopAPI = getDesktopAPI()
    setApi(desktopAPI)
    setReady(true)
  }, [])

  return { api, ready, isDesktopApp: isDesktopApp(), isTauri: isTauriApp() }
}

/**
 * Hook to check if running in a desktop environment
 */
export function useIsDesktop() {
  return isDesktopApp()
}

/**
 * Hook to check if running in Tauri
 */
export function useIsTauri() {
  return isTauriApp()
}

export default useDesktopAPI
