import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import CreatePromptEnhancer from './CreatePromptEnhancer'
import SimpleCreateControls from './SimpleCreateControls'
import { getEnhancerStatus, startEnhancer, cancelEnhancer } from '../services/donutCreate'

vi.mock('../services/donutCreate', () => ({
  getEnhancerStatus: vi.fn(), startEnhancer: vi.fn(), cancelEnhancer: vi.fn(),
  createErrorMessage: error => error.message, formatCreateBytes: bytes => `${bytes} bytes`,
}))
const idle = { models: [{ id: 'minicpm', name: 'MiniCPM5-2B', size: 10, ready: true }], supported: true, busy: false }
beforeEach(() => { vi.clearAllMocks(); getEnhancerStatus.mockResolvedValue(idle); cancelEnhancer.mockResolvedValue(idle) })
afterEach(cleanup)

// AC: @create-managed-prompt-enhancer ac-enhance-preview
// AC: @create-managed-prompt-enhancer ac-apply-routing
test('sends both inputs and previews before applying both fields without generating', async () => {
  const onApply = vi.fn().mockResolvedValue(undefined), onPauseInstant = vi.fn()
  startEnhancer.mockResolvedValue({ ...idle, job: { id: 'job', state: 'complete', kind: 'enhance', result: { main_prompt: 'An owl looks left.', face_prompt: 'The owl looks left.' } } })
  render(<CreatePromptEnhancer sessionId="synthetic" imageBrief="An owl." faceBrief="The owl looks left." onApply={onApply} onPauseInstant={onPauseInstant} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Enhance prompts' }).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Enhance prompts' }))
  await screen.findByText('Review enhanced prompts')
  expect(startEnhancer).toHaveBeenCalledWith('synthetic', { kind: 'enhance', model: 'minicpm', image_brief: 'An owl.', face_brief: 'The owl looks left.' })
  expect(onPauseInstant).toHaveBeenCalled()
  expect(onApply).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Apply both prompts' }))
  await waitFor(() => expect(onApply).toHaveBeenCalledWith({ main_prompt: 'An owl looks left.', face_prompt: 'The owl looks left.' }, { imageBrief: 'An owl.', faceBrief: 'The owl looks left.' }))
})

// AC: @create-managed-prompt-enhancer ac-apply-routing
test('editing either input invalidates a completed preview', async () => {
  startEnhancer.mockResolvedValue({ ...idle, job: { id: 'job', state: 'complete', kind: 'enhance', result: { main_prompt: 'An owl.', face_prompt: '' } } })
  const onApply = vi.fn(), view = render(<CreatePromptEnhancer sessionId="synthetic" imageBrief="An owl." faceBrief="" onApply={onApply} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  await screen.findByRole('button', { name: 'Enhance prompts' })
  fireEvent.click(screen.getByRole('button', { name: 'Enhance prompts' }))
  await screen.findByText('Review enhanced prompts')
  view.rerender(<CreatePromptEnhancer sessionId="synthetic" imageBrief="An owl." faceBrief="Eyes closed." onApply={onApply} />)
  expect(screen.getByRole('button', { name: 'Apply both prompts' }).disabled).toBe(true)
  expect(onApply).not.toHaveBeenCalled()
})

// AC: @create-managed-prompt-enhancer ac-apply-routing
test('the real Create controls send the hidden face brief and apply one guarded pair patch', async () => {
  const onPatch = vi.fn().mockResolvedValue(undefined), onGenerate = vi.fn(), onRunInstantChange = vi.fn()
  startEnhancer.mockResolvedValue({ ...idle, job: { id: 'routing', state: 'complete', kind: 'enhance', result: { main_prompt: 'An owl gazes left.', face_prompt: 'The owl gazes left.' } } })
  render(<SimpleCreateControls sessionId="synthetic" activeTab="create" connected snapshot={{ fields: {
    prompt: { kind: 'text', value: 'An owl.' }, facePrompt: { kind: 'text', value: 'The owl gazes left.' },
    faceDetail: { kind: 'boolean', value: false },
  } }} onPatch={onPatch} onGenerate={onGenerate} onRunInstantChange={onRunInstantChange} />)
  expect(screen.queryByLabelText('Describe only the face')).toBeNull()
  fireEvent.click(screen.getByText('Prompt enhancer'))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Enhance prompts' }).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Enhance prompts' }))
  await screen.findByText('Review enhanced prompts')
  fireEvent.click(screen.getByRole('button', { name: 'Apply both prompts' }))
  await waitFor(() => expect(onPatch).toHaveBeenCalledWith({ prompt: 'An owl gazes left.', facePrompt: 'The owl gazes left.' }, { prompt: 'An owl.', facePrompt: 'The owl gazes left.' }))
  expect(onRunInstantChange).toHaveBeenCalledWith(false)
  expect(onGenerate).not.toHaveBeenCalled()
})

// AC: @create-managed-prompt-enhancer ac-owned-private
test('unmount cancels only the operation started by this enhancer panel', async () => {
  startEnhancer.mockResolvedValue({ ...idle, busy: true, job: { id: 'owned', state: 'running', kind: 'enhance' } })
  const view = render(<CreatePromptEnhancer sessionId="synthetic" imageBrief="An owl." faceBrief="" onApply={vi.fn()} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Enhance prompts' }).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Enhance prompts' }))
  await screen.findByText('Enhancing both prompts…')
  view.unmount()
  expect(cancelEnhancer).toHaveBeenCalledWith('synthetic', 'owned')
})

// AC: @create-managed-prompt-enhancer ac-managed-models
test.each([false, true])('model settings stay accessible when an older workflow lacks the face prompt control (ready=%s)', async ready => {
  getEnhancerStatus.mockResolvedValue({ ...idle, models: [{ ...idle.models[0], ready }] })
  startEnhancer.mockResolvedValue({ ...idle, job: { id: 'setup', kind: 'install', state: 'complete' } })
  const onPatch = vi.fn(), onGenerate = vi.fn(), onLoadUpdatedV5 = vi.fn()
  render(<SimpleCreateControls sessionId="synthetic" activeTab="create" connected snapshot={{ fields: {
    prompt: { kind: 'text', value: 'A synthetic owl.' },
    facePrompt: { kind: 'text', value: null, available: false },
    faceDetail: { kind: 'boolean', value: true },
  } }} onPatch={onPatch} onGenerate={onGenerate} onLoadUpdatedV5={onLoadUpdatedV5} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  const setupName = ready ? 'Repair / verify model' : 'Download model'
  await waitFor(() => expect(screen.getByRole('button', { name: setupName }).disabled).toBe(false))
  expect(screen.getByLabelText('Enhancer model')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Enhance prompts' }).disabled).toBe(true)
  expect(screen.getByText(/This workflow does not expose both prompt controls/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: setupName }))
  await waitFor(() => expect(startEnhancer).toHaveBeenCalledWith('synthetic', { kind: 'install', model: 'minicpm' }))
  expect(onLoadUpdatedV5).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Load latest v5 preset (replaces recipe)' }))
  expect(onLoadUpdatedV5).toHaveBeenCalledTimes(1)
  expect(onPatch).not.toHaveBeenCalled()
  expect(onGenerate).not.toHaveBeenCalled()
})
