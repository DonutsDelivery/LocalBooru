const READINESS_LABELS = {
  unsupported_platform: 'Unsupported platform',
  not_installed: 'Not installed',
  accelerator_unavailable: 'Accelerator unavailable',
  incompatible_driver: 'Incompatible driver',
  downloading: 'Downloading',
  installing: 'Installing',
  probing: 'Verifying accelerator',
  repair_required: 'Repair required',
  update_available: 'Update available',
  ready: 'Ready',
  runtime_failure: 'Runtime failure',
}

export function normalizeAddonStatus(status) {
  if (typeof status === 'string') return { key: status, reason: null }
  if (status && typeof status === 'object' && typeof status.error === 'string') {
    return { key: 'error', reason: status.error }
  }
  return { key: 'error', reason: 'The add-on returned an unknown status' }
}

export function formatLadaReadiness(readiness) {
  if (!readiness) return null
  const key = readiness.status
  const status = READINESS_LABELS[key] || 'Unknown status'
  const configuredBackend = readiness.configured_backend || 'auto'
  const activeBackend = key === 'ready' ? readiness.active_backend || null : null
  const transitional = ['downloading', 'installing', 'probing'].includes(key)
  return {
    key,
    status,
    reason: readiness.reason || null,
    configuredBackend,
    activeBackend,
    transitional,
    badge: activeBackend ? 'running' : transitional ? 'starting' : key === 'not_installed' ? 'not-installed' : 'error',
  }
}

export function formatBackend(backend) {
  if (backend === 'cuda') return 'CUDA'
  if (backend === 'xpu') return 'Intel XPU'
  return 'Automatic'
}
