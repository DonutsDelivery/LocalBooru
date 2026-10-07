import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import CreatePromptEnhancer from './CreatePromptEnhancer'
import SimpleCreateControls from './SimpleCreateControls'
import { getEnhancerStatus, startEnhancer, cancelEnhancer } from '../services/donutCreate'
vi.mock('../services/donutCreate', () => ({ getEnhancerStatus: vi.fn(), startEnhancer: vi.fn(), cancelEnhancer: vi.fn(), createErrorMessage: error => error.message, formatCreateBytes: bytes => `${bytes} bytes` }))
const idle = { models: [{ id: 'minicpm', name: 'MiniCPM5-2B', ready: true }, { id: 'qwen', name: 'Qwen3.5-4B', ready: true }], supported: true, busy: false }
beforeEach(() => { vi.clearAllMocks(); getEnhancerStatus.mockResolvedValue(idle); cancelEnhancer.mockResolvedValue(idle) })
afterEach(cleanup)
// AC: @create-managed-prompt-enhancer ac-infer-empty-face
test('explains how an empty face brief derives a visible face while preserving overrides', async () => {
  render(<CreatePromptEnhancer sessionId="synthetic" config={{enabled:true,model:'minicpm'}} onConfigure={vi.fn()} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  expect(await screen.findByText(/Leave the face brief blank to derive one identifiable visible face from Describe your image/)).toBeTruthy()
  expect(screen.getByText(/A supplied face brief takes priority; scenes with hidden or ambiguous faces keep the face prompt empty/)).toBeTruthy()
  expect(startEnhancer).not.toHaveBeenCalled()
})
// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('enables once without enhancing or applying either source input', async () => {
  const onConfigure = vi.fn().mockResolvedValue(undefined)
  render(<CreatePromptEnhancer sessionId="synthetic" config={{enabled:false,model:'minicpm'}} onConfigure={onConfigure} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  const toggle = await screen.findByRole('checkbox',{name:'Enhance every run'})
  await waitFor(()=>expect(toggle.disabled).toBe(false)); fireEvent.click(toggle)
  await waitFor(()=>expect(onConfigure).toHaveBeenCalledWith({enabled:true,model:'minicpm'}))
  expect(startEnhancer).not.toHaveBeenCalled()
  expect(screen.queryByRole('button',{name:'Enhance prompts'})).toBeNull()
  expect(screen.queryByRole('button',{name:'Apply both prompts'})).toBeNull()
})
// AC: @create-managed-prompt-enhancer ac-enhance-preview
test('shows actual last-run prompts and seed without changing source controls', async () => {
  const onConfigure=vi.fn()
  render(<CreatePromptEnhancer sessionId="synthetic" config={{enabled:true,model:'minicpm',lastRun:{seed:12345,model:'minicpm',main_prompt:'An owl in soft light.',face_prompt:'The owl looks left.'}}} onConfigure={onConfigure} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  expect((await screen.findByLabelText('Enhanced image prompt')).value).toBe('An owl in soft light.')
  expect(screen.getByLabelText('Enhanced face prompt').value).toBe('The owl looks left.')
  expect(screen.getByLabelText('Enhanced image prompt').readOnly).toBe(true)
  expect(screen.getByText(/12345/)).toBeTruthy()
  expect(onConfigure).not.toHaveBeenCalled(); expect(startEnhancer).not.toHaveBeenCalled()
})
// AC: @create-managed-prompt-enhancer ac-apply-routing
test('Create commits original written briefs and lets the queue enhance the run', async()=>{
  const onPatch=vi.fn().mockResolvedValue(undefined),onGenerate=vi.fn().mockResolvedValue(undefined)
  render(<SimpleCreateControls sessionId="synthetic" activeTab="create" connected snapshot={{enhancer:{enabled:true,model:'minicpm'},fields:{prompt:{kind:'text',value:'An owl.'},facePrompt:{kind:'text',value:'The owl looks left.'},faceDetail:{kind:'boolean',value:false}}}} onPatch={onPatch} onGenerate={onGenerate} onEnhancerConfig={vi.fn()} />)
  expect(screen.queryByLabelText('Describe only the face')).toBeNull()
  fireEvent.change(screen.getByLabelText('Describe your image'),{target:{value:'Invent a room containing one owl.'}})
  fireEvent.click(screen.getByRole('button',{name:'Generate image',exact:true}))
  await waitFor(()=>expect(onGenerate).toHaveBeenCalledTimes(1))
  expect(onPatch).toHaveBeenCalledWith({prompt:'Invent a room containing one owl.'})
  expect(onPatch.mock.calls).toHaveLength(1)
  expect(startEnhancer).not.toHaveBeenCalled()
})
// AC: @create-managed-prompt-enhancer ac-owned-private
test('unmount cancels only installation started by this panel',async()=>{
  startEnhancer.mockResolvedValue({...idle,busy:true,job:{id:'owned-install',state:'running',kind:'install'}})
  const view=render(<CreatePromptEnhancer sessionId="synthetic" config={{enabled:false,model:'minicpm'}} onConfigure={vi.fn()} />)
  fireEvent.click(screen.getByText('Prompt enhancer')); const repair=await screen.findByRole('button',{name:'Repair / verify model'})
  await waitFor(()=>expect(repair.disabled).toBe(false)); fireEvent.click(repair)
  await waitFor(()=>expect(startEnhancer).toHaveBeenCalledWith('synthetic',{kind:'install',model:'minicpm'}))
  view.unmount(); expect(cancelEnhancer).toHaveBeenCalledWith('synthetic','owned-install')
})
// AC: @create-managed-prompt-enhancer ac-managed-models
test.each([false,true])('setup accessible but automatic enhancement disabled with missing face binding (ready=%s)',async ready=>{
  getEnhancerStatus.mockResolvedValue({...idle,models:[{...idle.models[0],ready}]})
  render(<SimpleCreateControls sessionId="synthetic" activeTab="create" connected snapshot={{fields:{prompt:{kind:'text',value:'A synthetic owl.'},facePrompt:{kind:'text',value:null,available:false},faceDetail:{kind:'boolean',value:true}}}} onPatch={vi.fn()} onGenerate={vi.fn()} onEnhancerConfig={vi.fn()} />)
  fireEvent.click(screen.getByText('Prompt enhancer')); const name=ready?'Repair / verify model':'Download model'
  await waitFor(()=>expect(screen.getByRole('button',{name}).disabled).toBe(false))
  expect(screen.getByRole('checkbox',{name:'Enhance every run'}).disabled).toBe(true)
  expect(screen.getByText(/This custom workflow does not expose both prompt controls/)).toBeTruthy()
})
// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('model selection changes future runs without enhancing a draft',async()=>{
  const onConfigure=vi.fn().mockResolvedValue(undefined)
  render(<CreatePromptEnhancer sessionId="synthetic" config={{enabled:true,model:'minicpm'}} onConfigure={onConfigure} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  await waitFor(()=>expect(screen.getByLabelText('Enhancer model').options.length).toBe(2))
  fireEvent.change(screen.getByLabelText('Enhancer model'),{target:{value:'qwen'}})
  await waitFor(()=>expect(onConfigure).toHaveBeenCalledWith({enabled:true,model:'qwen'}))
  expect(startEnhancer).not.toHaveBeenCalled()
})

// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('can disable automatic enhancement if the configured model or workflow becomes unavailable',async()=>{
  getEnhancerStatus.mockResolvedValue({...idle,supported:false,models:[{...idle.models[0],ready:false}]})
  const onConfigure=vi.fn().mockResolvedValue(undefined)
  render(<CreatePromptEnhancer sessionId="synthetic" config={{enabled:true,model:'minicpm'}} disabled unavailableReason="Synthetic missing binding" onConfigure={onConfigure} />)
  fireEvent.click(screen.getByText('Prompt enhancer'))
  const toggle=await screen.findByRole('checkbox',{name:'Enhance every run'})
  expect(toggle.disabled).toBe(false); fireEvent.click(toggle)
  await waitFor(()=>expect(onConfigure).toHaveBeenCalledWith({enabled:false,model:'minicpm'}))
})
