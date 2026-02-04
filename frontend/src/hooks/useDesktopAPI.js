/**
 * React hook for desktop API access
 *
 * Provides a unified API that works in both Electron and Tauri environments.
 */
import { useState, useEffect } from 'react'
import { getDesktopAPI, isDesktopApp, isTauriApp, isElectronApp } from '../tauriAPI'

/**
 * Hook to access the desktop API (Electron or Tauri)
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

  return { api, ready, isDesktopApp: isDesktopApp(), isTauri: isTauriApp(), isElectron: isElectronApp() }
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

/**
 * Hook to check if running in Electron
 */
export function useIsElectron() {
  return isElectronApp()
}

export default useDesktopAPI
