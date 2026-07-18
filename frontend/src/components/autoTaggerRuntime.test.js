import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatExecutionState,
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
