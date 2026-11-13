import { useEditionCapabilities } from '../contexts/edition'
import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { completeHubAuthorization } from '../api'
import { toast } from './Toast'

// Links request enrollment or finish a pending PKCE flow; neither grants access.
export default function HubLinkHandler() {
  const navigate = useNavigate()
  const edition = useEditionCapabilities()
  const handled = useRef(new Set())

  useEffect(() => {
    if (!window.__TAURI_INTERNALS__) return
    let active = true
    let unlisten
    const handle = async raw => {
      if (!active || typeof raw !== 'string' || raw.length > 8192 || handled.current.has(raw)) return
      let url
      try { url = new URL(raw) } catch { return }
      if (url.protocol !== 'donutmediacenter:' || url.username || url.password || url.port || url.hash) return
      if (!((url.hostname === 'hub' && url.pathname === '/add') || (url.hostname === 'auth' && url.pathname === '/callback'))) return
      handled.current.add(raw)
      if (handled.current.size > 32) handled.current.delete(handled.current.values().next().value)
      if (!edition.community_hubs) { toast.error('Community connections are not included in DonutMediaCenter Lite.'); return }
      if (url.hostname === 'hub') {
        window.__dmcPendingHubLink = raw
        navigate('/settings/fediverse')
        window.dispatchEvent(new CustomEvent('dmc-hub-link', { detail: raw }))
      } else {
        try {
          const connection = await completeHubAuthorization(raw)
          if (!active) return
          window.dispatchEvent(new CustomEvent('dmc-hub-authorized', { detail: connection }))
          toast.success('Community account connected.')
        } catch {
          if (active) toast.error('Community sign-in could not be completed. Start sign-in again in Connected communities.')
        }
      }
    }
    Promise.all([import('@tauri-apps/api/core'), import('@tauri-apps/api/event')])
      .then(async ([{ invoke }, { listen }]) => {
        const stop = await listen('deep-link://new-url', event => {
          for (const raw of event.payload || []) void handle(raw)
        })
        if (!active) { stop(); return }
        unlisten = stop
        const current = await invoke('plugin:deep-link|get_current')
        if (active) for (const raw of current || []) void handle(raw)
      })
      .catch(() => { if (active) toast.error('Community links are unavailable in this app. Paste the link in Settings → Fediverse & Nodes.') })
    return () => { active = false; unlisten?.() }
  }, [navigate, edition.community_hubs])

  return null
}
