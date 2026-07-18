const PROVIDER_LABELS = {
  CUDAExecutionProvider: 'CUDA',
  CPUExecutionProvider: 'CPU',
}

export function formatProvider(provider) {
  return PROVIDER_LABELS[provider] || provider || 'None'
}

export function formatProviderList(providers) {
  if (!providers) return 'Not available'
  return providers.length ? providers.map(formatProvider).join(', ') : 'None'
}

export function formatExecutionState(executionState) {
  const labels = {
    not_run: 'Not verified — run a prediction',
    cuda: 'CUDA',
    cpu: 'CPU',
    mixed: 'Mixed (CUDA and CPU)',
    unknown: 'Unknown',
  }

  return labels[executionState] || 'Unknown'
}

function formatMilliseconds(value) {
  return `${Math.round(value * 100) / 100} ms`
}

export function formatProviderMetric(values, unit = '') {
  const entries = Object.entries(values || {})
  if (!entries.length) return 'Not available'

  return entries
    .map(([provider, value]) => `${formatProvider(provider)}: ${unit === 'ms' ? formatMilliseconds(value) : value}`)
    .join(', ')
}

export function formatTimings(timings) {
  if (!timings) return 'Not available'

  const labels = {
    preprocess: 'Preprocess',
    inference: 'Inference',
    postprocess: 'Postprocess',
    total: 'Total',
  }
  const entries = Object.entries(labels)
    .filter(([key]) => Number.isFinite(timings[key]))
    .map(([key, label]) => `${label}: ${formatMilliseconds(timings[key])}`)

  return entries.length ? entries.join(', ') : 'Not available'
}
