import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatExecutionState,
  formatPackageVersions,
  formatPreload,
  formatProvider,
  formatProviderList,
  formatProviderMetric,
  formatTimings,
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
