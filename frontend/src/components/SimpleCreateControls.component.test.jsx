import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import SimpleCreateControls from './SimpleCreateControls'

const number = value => ({ kind: 'number', value, min: 0, max: 2, step: 0.01 })
const boolean = value => ({ kind: 'boolean', value })
const props = fields => ({
  snapshot: { fields }, activeTab: 'create', connected: true, onPatch: vi.fn().mockResolvedValue(undefined),
  onTabChange: vi.fn(), onGenerate: vi.fn(), onRunInstantChange: vi.fn(),
})
const panel = button => document.getElementById(button.getAttribute('aria-controls'))

beforeEach(() => { window.innerWidth = 1280; localStorage.clear() })
afterEach(() => { cleanup(); window.innerWidth = 1024 })

test.each([
  ['decensor', 'Decensor', 'decensorWeight'],
  ['upscale1', 'First upscale', 'upscale1Denoise'],
  ['upscale2', 'Second upscale', 'upscale2Denoise'],
  ['faceDetail', 'Face detail', 'faceDenoise'],
  ['postUpscale', 'Final upscale', 'postUpscaleScale'],
])('%s can expand while off and collapse while on without changing activation or values', async (key, title, setting) => {
  const initial = props({ [key]: boolean(false), [setting]: number(0.35) })
  const view = render(<SimpleCreateControls {...initial} />)
  const disclosure = screen.getByRole('button', { name: title, exact: true })
  expect(disclosure.getAttribute('aria-expanded')).toBe('false')
  fireEvent.click(disclosure)
  expect(panel(disclosure).hidden).toBe(false)
  expect(screen.getByRole('switch', { name: title }).checked).toBe(false)
  expect(initial.onPatch).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('switch', { name: title }))
  await waitFor(() => expect(initial.onPatch).toHaveBeenCalledWith({ [key]: true }))
  view.rerender(<SimpleCreateControls {...initial} snapshot={{ fields: { ...initial.snapshot.fields, [key]: boolean(true) } }} />)
  expect(panel(disclosure).hidden).toBe(false)
  initial.onPatch.mockClear()
  fireEvent.click(disclosure)
  expect(panel(disclosure).hidden).toBe(true)
  expect(screen.getByRole('switch', { name: title }).checked).toBe(true)
  expect(initial.onPatch).not.toHaveBeenCalled()
  fireEvent.click(disclosure)
  expect(panel(disclosure).querySelector('input[type="number"]').value).toBe('0.35')
})

test.each([['nagStrength', 'NAG'], ['sdaStrength', 'SDA'], ['toneStrength', 'ToneLab']])('%s retains collapse state and remembered strength across activation changes', async (key, title) => {
  const initial = props({ [key]: number(0.45) })
  const view = render(<SimpleCreateControls {...initial} />)
  const disclosure = screen.getByRole('button', { name: title, exact: true })
  expect(panel(disclosure).hidden).toBe(false)
  fireEvent.click(disclosure)
  fireEvent.click(screen.getByRole('switch', { name: title }))
  await waitFor(() => expect(initial.onPatch).toHaveBeenLastCalledWith({ [key]: 0 }))
  view.rerender(<SimpleCreateControls {...initial} snapshot={{ fields: { [key]: number(0) } }} />)
  expect(panel(disclosure).hidden).toBe(true)
  fireEvent.click(disclosure)
  expect(screen.getByRole('switch', { name: title }).checked).toBe(false)
  expect(panel(disclosure).hidden).toBe(false)
  fireEvent.click(screen.getByRole('switch', { name: title }))
  await waitFor(() => expect(initial.onPatch).toHaveBeenLastCalledWith({ [key]: 0.45 }))
  expect(panel(disclosure).hidden).toBe(false)
})

test('compatibility can collapse with its preset off and keeps TAB strength hidden', () => {
  const initial = props({ compatibilityPreset: { kind: 'select', value: 'Off', options: ['Off', 'Rebalance'] }, tapStrength: number(1) })
  render(<SimpleCreateControls {...initial} />)
  const disclosure = screen.getByRole('button', { name: 'Compatibility' })
  expect(panel(disclosure).hidden).toBe(false)
  expect(screen.queryByLabelText('TAB strength')).toBeNull()
  fireEvent.click(disclosure)
  expect(panel(disclosure).hidden).toBe(true)
  fireEvent.click(disclosure)
  expect(panel(disclosure).hidden).toBe(false)
  expect(initial.onPatch).not.toHaveBeenCalled()
})

test('disclosures remain usable while backend controls are disabled', () => {
  render(<SimpleCreateControls {...props({ decensor: boolean(false), decensorWeight: number(1) })} connected={false} />)
  const disclosure = screen.getByRole('button', { name: 'Decensor', exact: true })
  expect(disclosure.disabled).toBe(false)
  fireEvent.click(disclosure)
  expect(panel(disclosure).hidden).toBe(false)
  expect(screen.getByRole('switch', { name: 'Decensor' }).disabled).toBe(true)
})

test('desktop navigation lives in each sidebar; mobile keeps a single shared navigation', () => {
  const exitNavigation = <div className="create-exit-slot"><button type="button">Exit</button></div>
  const modeNavigation = <div className="create-mode-actions"><button type="button">Advanced</button></div>
  render(<SimpleCreateControls {...props({})} exitNavigation={exitNavigation} modeNavigation={modeNavigation}
    navigation={<nav aria-label="Studio navigation">{exitNavigation}{modeNavigation}</nav>} />)
  expect(screen.getByRole('button', { name: 'Exit' }).closest('.create-controls')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Advanced' }).closest('.create-effects')).toBeTruthy()
  window.innerWidth = 360
  fireEvent(window, new Event('resize'))
  expect(screen.getAllByRole('button', { name: 'Exit' })).toHaveLength(1)
  expect(screen.getAllByRole('button', { name: 'Advanced' })).toHaveLength(1)
  expect(screen.getByRole('button', { name: 'Exit' }).closest('nav')).toBe(screen.getByRole('navigation', { name: 'Studio navigation' }))
})

const select = (value, options = [value]) => ({ kind: 'select', value, options })
function placementFields() {
  return {
    modelMode: select('Merge two models', ['Single model', 'Merge two models']),
    model: select('primary.safetensors'), secondaryModel: select('secondary.safetensors'), modelBlend: number(0.25),
    loras: { kind: 'loras', value: [{ id: 'synthetic-lora', enabled: true, lora_name: 'style.safetensors', model_weight: 0.987, clip_weight: 1.2 }],
      options: ['style.safetensors', 'subject.safetensors'], min: -1000, max: 1000, step: 0.01 },
    upscale1: boolean(true), upscale2: boolean(true), postUpscale: boolean(true),
    upscale1Engine: select('SeedVR2', ['SeedVR2', 'Donut']), upscale2Engine: select('Donut', ['SeedVR2', 'Donut']),
    upscale1Model: select('seed-model.safetensors'), upscale1Vae: select('seed-vae.safetensors'),
    upscale1SeedVrSteps: { ...number(12), max: 100 }, upscale1SeedVrDenoise: number(0.35),
    upscale1Scale: number(1.25), upscale1Denoise: number(0.36), upscale2Scale: number(1.5), upscale2Denoise: number(0.45),
    upscaleModel: select('donut-model.pth'),
    postUpscaleModel: select('final-model.pth'), postUpscaleVae: select('final-vae.safetensors'),
    postUpscaleScale: number(1.75), postUpscaleSteps: { ...number(8), max: 100 }, postUpscaleDenoise: number(0.36), postUpscaleColorCorrection: select('Match'),
    editLora: select('editing.safetensors', ['None', 'editing.safetensors']), editLoraStrength: number(0.75),
    decensor: boolean(true), decensorWeight: number(0.3),
  }
}

// AC: @create-sidebar-layout ac-placement
test('all upscaler fields are left, recipe and LoRAs are right, and only History is lower', () => {
  const initial = props(placementFields())
  render(<SimpleCreateControls {...initial} historyContent={<button type="button">Synthetic history output</button>} />)
  const left = screen.getByRole('complementary', { name: 'Image controls' })
  const right = screen.getByRole('complementary', { name: 'Models and image effects' })
  for (const title of ['First upscale', 'Second upscale', 'Final upscale']) {
    expect(screen.getByRole('button', { name: title, exact: true }).closest('aside')).toBe(left)
  }
  for (const label of ['Engine', 'Upscale model', 'Upscale VAE', 'Scale', 'Denoise', 'Upscale steps', 'Upscale denoise', 'Donut upscale model', 'Color correction']) {
    for (const control of screen.getAllByLabelText(label, { exact: true })) expect(control.closest('aside')).toBe(left)
  }
  for (const label of ['Model recipe', 'Primary model', 'Second model', 'Uniform model blend', 'LoRA', 'Model strength', 'Text strength']) {
    expect(screen.getByLabelText(label, { exact: true }).closest('aside')).toBe(right)
  }
  const history = screen.getByRole('region', { name: 'Image history' })
  expect(within(history).queryByRole('button', { name: 'Synthetic history output' })).toBeNull()
  fireEvent.click(within(history).getByRole('tab', { name: 'History' }))
  expect(within(history).getByRole('button', { name: 'Synthetic history output' })).toBeDefined()
  expect(within(history).queryByLabelText('Primary model')).toBeNull()
  expect(within(history).queryByRole('button', { name: /Add LoRA/ })).toBeNull()
  expect(screen.queryByRole('tab', { name: 'Models & LoRAs' })).toBeNull()
  fireEvent.click(within(history).getByRole('button', { name: 'Collapse history' }))
  expect(within(history).queryByRole('button', { name: 'Synthetic history output' })).toBeNull()
})

// AC: @create-sidebar-layout ac-state
test('upscaler disclosure and values stay mounted across Create Edit and Finalize', () => {
  const initial = props(placementFields())
  const view = render(<SimpleCreateControls {...initial} />)
  const disclosure = screen.getByRole('button', { name: 'First upscale', exact: true })
  fireEvent.click(disclosure)
  const settings = panel(disclosure)
  const scale = settings.querySelector('input[type="number"]')
  for (const activeTab of ['edit', 'finalize', 'create']) {
    view.rerender(<SimpleCreateControls {...initial} activeTab={activeTab} />)
    expect(screen.getByRole('button', { name: 'First upscale', exact: true })).toBe(disclosure)
    expect(settings.hidden).toBe(true)
    expect(settings.querySelector('input[type="number"]')).toBe(scale)
    expect(screen.getByRole('switch', { name: 'First upscale' }).checked).toBe(true)
  }
  expect(initial.onPatch).not.toHaveBeenCalled()
  view.rerender(<SimpleCreateControls {...initial} activeTab="edit" />)
  expect(screen.getByLabelText('Editing LoRA').closest('.create-effects')).toBeTruthy()
  expect(screen.getByLabelText('Editing strength').value).toBe('0.75')
})

// AC: @create-sidebar-layout ac-controls
test('left upscaler engine switches preserve existing conditional model and VAE mapping', async () => {
  const fields = placementFields(), initial = props(fields)
  const view = render(<SimpleCreateControls {...initial} />)
  const settings = panel(screen.getByRole('button', { name: 'First upscale', exact: true }))
  fireEvent.change(within(settings).getByLabelText('Engine'), { target: { value: 'Donut' } })
  await waitFor(() => expect(initial.onPatch).toHaveBeenLastCalledWith({ upscale1Engine: 'Donut' }))
  view.rerender(<SimpleCreateControls {...initial} snapshot={{ fields: { ...fields, upscale1Engine: select('Donut', ['SeedVR2', 'Donut']) } }} />)
  expect(within(settings).getByLabelText('Donut upscale model').value).toBe('donut-model.pth')
  expect(within(settings).queryByLabelText('Upscale VAE')).toBeNull()
  fireEvent.change(within(settings).getByLabelText('Engine'), { target: { value: 'SeedVR2' } })
  await waitFor(() => expect(initial.onPatch).toHaveBeenLastCalledWith({ upscale1Engine: 'SeedVR2' }))
  view.rerender(<SimpleCreateControls {...initial} />)
  expect(within(settings).getByLabelText('Upscale model').value).toBe('seed-model.safetensors')
  expect(within(settings).getByLabelText('Upscale VAE').value).toBe('seed-vae.safetensors')
})

// AC: @create-sidebar-layout ac-controls
test('right LoRA number keeps fine precision and slider commits once after release', async () => {
  const initial = props(placementFields())
  render(<SimpleCreateControls {...initial} />)
  const right = screen.getByRole('complementary', { name: 'Models and image effects' })
  const model = within(right).getByLabelText('Model strength')
  expect(model.value).toBe('0.987')
  fireEvent.change(model, { target: { value: '0.64321' } })
  expect(initial.onPatch).not.toHaveBeenCalled()
  fireEvent.blur(model)
  await waitFor(() => expect(initial.onPatch).toHaveBeenLastCalledWith({ loras: [expect.objectContaining({ model_weight: 0.64321, clip_weight: 1.2 })] }))
  initial.onPatch.mockClear()
  const textRange = within(right).getByRole('slider', { name: 'Text strength slider for LoRA 1' })
  fireEvent.pointerDown(textRange)
  fireEvent.input(textRange, { target: { value: '0.65' } })
  expect(within(right).getByLabelText('Text strength').value).toBe('0.65')
  expect(initial.onPatch).not.toHaveBeenCalled()
  expect(screen.getByRole('button', { name: /Generate image/ }).disabled).toBe(true)
  fireEvent.pointerUp(textRange)
  fireEvent.blur(textRange)
  await waitFor(() => expect(initial.onPatch).toHaveBeenCalledTimes(1))
  expect(initial.onPatch).toHaveBeenCalledWith({ loras: [expect.objectContaining({ clip_weight: 0.65 })] })
})

// AC: @create-sidebar-layout ac-responsive, ac-state
test('mobile Models and Effects select mounted right-sidebar sections and preserve model drafts', async () => {
  window.innerWidth = 390
  const initial = props(placementFields())
  const historyContent = <button type="button">Synthetic mobile history</button>
  const view = render(<SimpleCreateControls {...initial} mobilePane="models" historyContent={historyContent} />)
  const recipe = screen.getByLabelText('Primary model').closest('.create-model-workspace')
  const effects = screen.getByRole('button', { name: 'Decensor', exact: true, hidden: true }).closest('.create-effects-workspace')
  expect(recipe.closest('.create-effects')).toBeTruthy()
  expect(recipe.hidden).toBe(false)
  expect(effects.hidden).toBe(true)
  const modelStrength = screen.getByLabelText('Model strength')
  fireEvent.change(modelStrength, { target: { value: '0.64321' } })
  view.rerender(<SimpleCreateControls {...initial} mobilePane="effects" historyContent={historyContent} />)
  expect(recipe.hidden).toBe(true)
  expect(effects.hidden).toBe(false)
  view.rerender(<SimpleCreateControls {...initial} mobilePane="models" historyContent={historyContent} />)
  expect(screen.getByLabelText('Model strength')).toBe(modelStrength)
  expect(modelStrength.value).toBe('0.64321')
  fireEvent.blur(modelStrength)
  await waitFor(() => expect(initial.onPatch).toHaveBeenCalledWith({ loras: [expect.objectContaining({ model_weight: 0.64321 })] }))
  expect(screen.getByRole('separator', { name: 'Resize models and effects pane' }).tabIndex).toBe(-1)
  const history = screen.getByRole('button', { name: 'Synthetic mobile history' }).closest('[role="tabpanel"]')
  view.rerender(<SimpleCreateControls {...initial} mobilePane="history" historyContent={historyContent} />)
  expect(screen.getByRole('button', { name: 'Synthetic mobile history' }).closest('[role="tabpanel"]')).toBe(history)
  expect(history.hidden).toBe(false)
})

// AC: @create-sidebar-layout ac-responsive
test('sidebar and History keyboard resizing retain central space and relocated controls', () => {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({ width: 1000, height: 800 })
  const initial = props(placementFields())
  const { container } = render(<SimpleCreateControls {...initial} />)
  const left = screen.getByRole('separator', { name: 'Resize controls pane' })
  const form = container.querySelector('form')
  fireEvent.keyDown(left, { key: 'End' })
  const width = key => Number(form.style.getPropertyValue(`--create-${key}-size`).replace('px', ''))
  expect(1000 - width('left') - width('right')).toBe(320)
  expect(screen.getByLabelText('Primary model').closest('.create-effects')).toBeTruthy()
  fireEvent.keyDown(screen.getByRole('separator', { name: 'Resize history pane' }), { key: 'ArrowUp' })
  expect(screen.getByRole('button', { name: 'Collapse history' }).getAttribute('aria-expanded')).toBe('true')
  expect(Number(form.style.getPropertyValue('--create-lower-size').replace('px', ''))).toBeGreaterThan(160)
  vi.restoreAllMocks()
})
