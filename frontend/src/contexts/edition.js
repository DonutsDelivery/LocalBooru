import { createContext, useContext } from 'react'
export const FULL = Object.freeze({ edition: 'full', community_hubs: true, downloaded_executable_addons: true, lan_library: true, self_updates: true })
export const UNKNOWN = Object.freeze({ edition: 'unknown', community_hubs: false, downloaded_executable_addons: false, lan_library: true, self_updates: false })
export const EditionContext = createContext(FULL)

export function useEditionCapabilities() { return useContext(EditionContext) }

export function addonAvailableInEdition(addon, capabilities) {
  return Boolean(capabilities.downloaded_executable_addons || addon?.installation === 'builtin' || addon?.runtime === 'builtin')
}
