import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatExecutionState,
  formatCudaDiagnostic,
  formatPackageVersions,
  formatPreload,
  formatProvider,
  formatProviderList,
  formatProviderMetric,
  formatTimings,
  runtimeDiagnosticTimeoutMs,
} from './autoTaggerRuntime.js'

// AC: @auto-tagger-execution-verification ac-6
test('formats provider availability and registration lists', () => {
  assert.equal(formatProviderList(['CUDAExecutionProvider', 'CPUExecutionProvider']), 'CUDA, CPU')
  assert.equal(formatProviderList([]), 'None')
  assert.equal(formatProviderList(undefined), 'Not available')
})

// AC: @auto-tagger-execution-verification ac-6
test('distinguishes unverified, mixed, and observed execution states', () => {
  assert.equal(formatExecutionState('not_run'), 'Not verified — run a prediction')
  assert.equal(formatExecutionState('mixed'), 'Mixed (CUDA and CPU)')
  assert.equal(formatExecutionState('cuda'), 'CUDA')
  assert.equal(formatExecutionState('cpu'), 'CPU')
  assert.equal(formatExecutionState('unknown'), 'Unknown')
})

// AC: @auto-tagger-execution-verification ac-6
test('keeps legacy active_provider separate from observed execution', () => {
  assert.equal(formatExecutionState(undefined), 'Unknown')
  assert.equal(formatProvider('CUDAExecutionProvider'), 'CUDA')
  assert.equal(formatProvider('CPUExecutionProvider'), 'CPU')
  assert.equal(formatProvider('CustomExecutionProvider'), 'CustomExecutionProvider')
})

// AC: @auto-tagger-execution-verification ac-6
test('formats provider node counts, durations, and prediction timings', () => {
  assert.equal(
    formatProviderMetric({ CUDAExecutionProvider: 178, CPUExecutionProvider: 12 }),
    'CUDA: 178, CPU: 12',
  )
  assert.equal(
    formatProviderMetric({ CUDAExecutionProvider: 36.456 }, 'ms'),
    'CUDA: 36.46 ms',
  )
  assert.equal(
    formatTimings({ preprocess: 4.2, inference: 36.456, postprocess: 1, total: 41.656 }),
    'Preprocess: 4.2 ms, Inference: 36.46 ms, Postprocess: 1 ms, Total: 41.66 ms',
  )
})

// AC: @auto-tagger-runtime-acceleration-deployment ac-2
test('formats deployment packages and native preload evidence', () => {
  assert.equal(
    formatPackageVersions({
      'onnxruntime-gpu': '1.23.2',
      'nvidia-cudnn-cu12': '9.14.0.64',
    }),
    'onnxruntime-gpu 1.23.2, nvidia-cudnn-cu12 9.14.0.64',
  )
  assert.equal(formatPackageVersions({}), 'Not available')
  assert.equal(formatPreload({ attempted: true, succeeded: true, error: null }), 'Succeeded')
  assert.equal(
    formatPreload({ attempted: true, succeeded: false, error: 'cudnn64_9.dll missing' }),
    'Failed: cudnn64_9.dll missing',
  )
  assert.equal(formatPreload({ attempted: false }), 'Not attempted')
  assert.equal(formatPreload(undefined), 'Not available')
  assert.equal(formatPreload({ attempted: true, succeeded: null }), 'Unknown')
})

// AC: @auto-tagger-runtime-acceleration-deployment ac-strict-diagnostic
test('keeps the client diagnostic deadline beyond the backend probe deadline', () => {
  assert.equal(runtimeDiagnosticTimeoutMs(300), 310_000)
  assert.ok(runtimeDiagnosticTimeoutMs(300) > 300_000)
})

// AC: @auto-tagger-runtime-acceleration-deployment ac-strict-diagnostic
test('formats strict CUDA report with native output and preserved failure evidence', () => {
  const formatted = formatCudaDiagnostic({
    status: 'failed',
    exit_code: 1,
    probe: {
      model: { name: 'eva02-large-v3', sha256: '9e768793' },
      runtime: {
        provider_options: { CUDAExecutionProvider: { device_id: '0' } },
        packages: { 'nvidia-cusparse-cu12': '12.5' },
      },
      execution: { error: 'CUDA launch failed', provider_node_counts: { CPUExecutionProvider: 1920 } },
      strict_stage: { execution: { error: 'node assignment failed' } },
    },
    stderr: 'native ORT log',
  })

  assert.match(formatted, /eva02-large-v3/)
  assert.match(formatted, /CUDA launch failed/)
  assert.match(formatted, /native ORT log/)
  assert.match(formatted, /nvidia-cusparse-cu12/)
})
