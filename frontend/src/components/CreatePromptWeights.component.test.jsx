import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import SimpleCreateControls from './SimpleCreateControls'
afterEach(cleanup)

// AC: @create-krea-prompt-weights ac-control
test('prompt weights toggle restores strength and changing it submits the actual control', async () => {
  const onPatch = vi.fn().mockResolvedValue(undefined)
  const props = {activeTab:'create',connected:true,onPatch,onGenerate:vi.fn(),snapshot:{fields:{
    prompt:{kind:'text',value:'A synthetic cat.'},promptWeightStrength:{kind:'number',value:0,min:0,max:4,step:.05,default:1}}}}
  const view = render(<SimpleCreateControls {...props} />)
  expect(screen.getByRole('switch',{name:'Prompt weights'}).checked).toBe(false)
  fireEvent.click(screen.getByRole('switch',{name:'Prompt weights'}))
  await waitFor(()=>expect(onPatch).toHaveBeenCalledWith({promptWeightStrength:1}))
  view.rerender(<SimpleCreateControls {...props} snapshot={{fields:{...props.snapshot.fields,promptWeightStrength:{...props.snapshot.fields.promptWeightStrength,value:1}}}} />)
  fireEvent.click(screen.getByRole('button',{name:'Prompt weights',exact:true}))
  const input=screen.getByRole('spinbutton',{name:'Weight strength'})
  fireEvent.change(input,{target:{value:'1.25'}});fireEvent.blur(input)
  await waitFor(()=>expect(onPatch).toHaveBeenCalledWith({promptWeightStrength:1.25}))
  expect(screen.getByText(/Use \(smile:1.5\)/)).toBeTruthy()
})

test('shift arrow weights the marked phrase by 0.05 from 1.00', () => {
  const onPatch = vi.fn().mockResolvedValue(undefined)
  render(<SimpleCreateControls activeTab="create" connected onPatch={onPatch} onGenerate={vi.fn()} snapshot={{fields:{
    prompt:{kind:'text',value:'A synthetic cat in fog.'},
    stylePrompt:{kind:'text',value:'soft light'},
    promptWeightStrength:{kind:'number',value:1,min:0,max:4,step:.05,default:1}}}} />)
  const box = screen.getByLabelText('Describe your image')
  expect(fireEvent.keyDown(box, { key: 'ArrowLeft', shiftKey: true })).toBe(true)
  const start = box.value.indexOf('cat')
  box.setSelectionRange(start, start + 3)
  fireEvent.keyDown(box, { key: 'ArrowUp', shiftKey: true })
  expect(box.value).toBe('A synthetic (cat:1.05) in fog.')
  expect(box.selectionStart).toBe(box.value.indexOf('(cat:1.05)'))
  expect(box.selectionEnd).toBe(box.selectionStart + '(cat:1.05)'.length)
  fireEvent.keyDown(box, { key: 'ArrowUp', shiftKey: true })
  expect(box.value).toBe('A synthetic (cat:1.1) in fog.')
  fireEvent.keyDown(box, { key: 'ArrowDown', shiftKey: true })
  fireEvent.keyDown(box, { key: 'ArrowDown', shiftKey: true })
  expect(box.value).toBe('A synthetic cat in fog.')
  const style = screen.getByLabelText('Style prompt')
  const at = style.value.indexOf('light')
  style.setSelectionRange(at, at)
  fireEvent.keyDown(style, { key: 'ArrowDown', shiftKey: true })
  expect(style.value).toBe('soft (light:0.95)')
})

test('LoRA dropdown matches any text in the filename, not only its first letters', async () => {
  const onPatch = vi.fn().mockResolvedValue(undefined)
  const name = 'krea2/Krea2_NSFW_Aesthetics_V1.safetensors'
  render(<SimpleCreateControls activeTab="create" connected onPatch={onPatch} onGenerate={vi.fn()} snapshot={{fields:{
    loras:{kind:'loras', value:[{id:'synthetic-lora', enabled:true, lora_name:'style.safetensors', model_weight:1, clip_weight:1}],
      options:['style.safetensors', name]}}}} />)
  const input = screen.getByLabelText('LoRA')
  fireEvent.focus(input)
  fireEvent.change(input, { target: { value: 'hetics' } })
  expect(screen.getByRole('option', { name: 'Krea2 NSFW Aesthetics V1' })).toBeTruthy()
  expect(screen.queryByRole('option', { name: 'style' })).toBeNull()
  const option = screen.getByRole('option', { name: 'Krea2 NSFW Aesthetics V1' })
  expect(input.getAttribute('aria-controls')).toBe(screen.getByRole('listbox').id)
  expect(input.getAttribute('aria-activedescendant')).toBe(option.id)
  fireEvent.mouseDown(option)
  fireEvent.click(option)
  await waitFor(() => expect(onPatch).toHaveBeenCalledWith({ loras: [expect.objectContaining({ lora_name: name })] }))
})

test('LoRA keyboard selection exposes its active option and Escape preserves the choice', async () => {
  const onPatch = vi.fn().mockResolvedValue(undefined)
  render(<SimpleCreateControls activeTab="create" connected onPatch={onPatch} onGenerate={vi.fn()} snapshot={{fields:{
    loras:{kind:'loras', value:[{id:'keyboard-lora', enabled:true, lora_name:'first.safetensors', model_weight:1, clip_weight:1}],
      options:['first.safetensors', 'second.safetensors']}}}} />)
  const input = screen.getByRole('combobox', { name: 'LoRA' })
  fireEvent.focus(input)
  fireEvent.keyDown(input, { key: 'ArrowDown' })
  expect(input.getAttribute('aria-activedescendant')).toBe(screen.getByRole('option', { name: 'second' }).id)
  fireEvent.keyDown(input, { key: 'Escape' })
  expect(input.getAttribute('aria-expanded')).toBe('false')
  expect(onPatch).not.toHaveBeenCalled()
  fireEvent.keyDown(input, { key: 'ArrowDown' })
  fireEvent.keyDown(input, { key: 'ArrowDown' })
  fireEvent.keyDown(input, { key: 'Enter' })
  await waitFor(() => expect(onPatch).toHaveBeenCalledTimes(1))
  expect(onPatch).toHaveBeenCalledWith({loras:[expect.objectContaining({lora_name:'second.safetensors'})]})
})

test('weight editing uses the current multiline selection after an earlier marked phrase', () => {
  render(<SimpleCreateControls activeTab="create" connected onPatch={vi.fn()} onGenerate={vi.fn()} snapshot={{fields:{
    prompt:{kind:'text',value:'A cat in fog.\nSoft light.'}}}} />)
  const box = screen.getByLabelText('Describe your image')
  box.setSelectionRange(2, 5)
  fireEvent.select(box)
  box.setSelectionRange(2, box.value.length)
  fireEvent.keyDown(box, { key: 'ArrowUp', shiftKey: true })
  expect(box.value).toBe('A (cat in fog.\nSoft light.:1.05)')
})

test('weight shortcuts update only the workspace that owns the textarea', async () => {
  const firstPatch = vi.fn().mockResolvedValue(undefined)
  const secondPatch = vi.fn().mockResolvedValue(undefined)
  const props = text => ({ activeTab:'create', connected:true, onGenerate:vi.fn(),
    snapshot:{fields:{prompt:{kind:'text',value:text}}} })
  render(<><SimpleCreateControls {...props('First owl.')} onPatch={firstPatch} />
    <SimpleCreateControls {...props('Second owl.')} onPatch={secondPatch} /></>)
  const [first, second] = screen.getAllByLabelText('Describe your image')
  second.setSelectionRange(0, 6)
  fireEvent.keyDown(second, { key:'ArrowUp', shiftKey:true })
  fireEvent.blur(second)
  expect(first.value).toBe('First owl.')
  expect(second.value).toBe('(Second:1.05) owl.')
  await waitFor(() => expect(secondPatch).toHaveBeenCalledWith({prompt:'(Second:1.05) owl.'}))
  expect(firstPatch).not.toHaveBeenCalled()
})

// AC: @create-krea-prompt-weights ac-control
test('weights are absent when a custom workflow has no supported control', () => {
  render(<SimpleCreateControls activeTab="create" connected snapshot={{fields:{}}} onPatch={vi.fn()} onGenerate={vi.fn()} />)
  expect(screen.queryByRole('switch',{name:'Prompt weights'})).toBeNull()
})
