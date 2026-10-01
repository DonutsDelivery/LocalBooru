import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
