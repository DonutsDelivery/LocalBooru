import { useEffect, useState } from 'react'
import { EditionContext, UNKNOWN, useEditionCapabilities } from './edition'

// Native capabilities are injected immutably before application scripts run.
// Never fetch this through the selected remote library's API client.
export function EditionProvider({ children }) {
  const [capabilities, setCapabilities] = useState(() => window.__DMC_EDITION_CAPABILITIES__ || UNKNOWN)
  useEffect(() => {
    if (window.__DMC_EDITION_CAPABILITIES__) return
    const controller = new AbortController()
    fetch('/api/edition', { signal: controller.signal, credentials: 'same-origin' })
      .then(response => { if (!response.ok) throw new Error('Edition unavailable'); return response.json() })
      .then(value => { if (!controller.signal.aborted) setCapabilities(Object.freeze(value)) })
      .catch(() => {}) // Optional capabilities stay disabled when discovery fails.
    return () => controller.abort()
  }, [])
  return <EditionContext.Provider value={capabilities}>{children}</EditionContext.Provider>
}

export function CapabilityGate({ capability, children }) {
  const capabilities = useEditionCapabilities()
  return capabilities[capability] ? children : <p role="status">This feature is not included in this edition of DonutMediaCenter.</p>
}
