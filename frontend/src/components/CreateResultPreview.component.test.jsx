import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import CreateResultPreview from './CreateResultPreview'

let viewport
let resize

beforeEach(() => {
  viewport = { width: 800, height: 600 }
  vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(() => viewport.width)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(() => viewport.height)
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback) { resize = callback }
    observe() {}
    disconnect() {}
  })
})

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

function loadImage(width, height) {
  const image = screen.getByRole('img')
  Object.defineProperties(image, {
    naturalWidth: { configurable: true, value: width },
    naturalHeight: { configurable: true, value: height },
  })
  fireEvent.load(image)
  return image
}

function fitSelected() {
  return screen.getByRole('button', { name: 'Fit', exact: true }).getAttribute('aria-pressed')
}

// AC: @donut-create-plugin ac-preview-fit
test('fits first pass, upscale and portrait stage changes and resets previous pixel zoom', () => {
  const { rerender } = render(<CreateResultPreview src="first-pass" imageKey="first" alt="Synthetic output" />)
  const image = loadImage(1024, 768)
  expect(screen.getByLabelText('Preview scale').textContent).toBe('78%')
  fireEvent.click(screen.getByRole('button', { name: '100%', exact: true }))
  expect(image.style.width).toBe('1024px')
  rerender(<CreateResultPreview src="upscale" imageKey="upscale" alt="Synthetic output" />)
  loadImage(4096, 3072)
  expect(fitSelected()).toBe('true')
  expect(image.className).toBe('create-preview-image')
  expect(image.style.width).toBe('')
  expect(screen.getByLabelText('Preview scale').textContent).toBe('20%')
  rerender(<CreateResultPreview src="portrait" imageKey="portrait" alt="Synthetic output" />)
  loadImage(512, 1024)
  expect(fitSelected()).toBe('true')
  expect(screen.getByLabelText('Preview scale').textContent).toBe('59%')
})

// AC: @donut-create-plugin ac-preview-fit
test('keeps intentional zoom on same-stage live frames but fits when dimensions change', () => {
  const { rerender } = render(<CreateResultPreview src="frame-1" imageKey="live-stage" alt="Live preview" />)
  const image = loadImage(1024, 768)
  fireEvent.click(screen.getByRole('button', { name: '100%', exact: true }))
  rerender(<CreateResultPreview src="frame-2" imageKey="live-stage" alt="Live preview" />)
  loadImage(1024, 768)
  expect(fitSelected()).toBe('false')
  expect(image.style.width).toBe('1024px')
  rerender(<CreateResultPreview src="frame-3" imageKey="live-stage" alt="Live preview" />)
  loadImage(4096, 3072)
  expect(fitSelected()).toBe('true')
  expect(image.style.width).toBe('')
})

// AC: @donut-create-plugin ac-preview-fit
test('follows viewport resizing in Fit while retaining explicit 100% zoom and clamping pan', () => {
  render(<CreateResultPreview src="synthetic" imageKey="image" alt="Synthetic output" />)
  const image = loadImage(1024, 768)
  viewport = { width: 400, height: 300 }
  act(() => resize())
  expect(screen.getByLabelText('Preview scale').textContent).toBe('39%')
  fireEvent.click(screen.getByRole('button', { name: '100%', exact: true }))
  expect(image.style.width).toBe('1024px')
  const stage = screen.getByLabelText('Image preview. Scroll to zoom, drag to pan.')
  fireEvent.keyDown(stage, { key: 'ArrowLeft' })
  expect(image.style.left).toBe('calc(50% + 40px)')
  viewport = { width: 1200, height: 900 }
  act(() => resize())
  expect(image.style.left).toBe('calc(50% + 0px)')
  expect(image.style.width).toBe('1024px')
  fireEvent.keyDown(stage, { key: '0' })
  expect(fitSelected()).toBe('true')
  expect(screen.getByLabelText('Preview scale').textContent).toBe('117%')
})

// AC: @donut-create-plugin ac-preview-fit
test('supports wheel and keyboard zoom with a return to Fit', () => {
  render(<CreateResultPreview src="synthetic" imageKey="image" alt="Synthetic output" />)
  const image = loadImage(1600, 1200)
  const stage = screen.getByLabelText('Image preview. Scroll to zoom, drag to pan.')
  fireEvent.wheel(stage, { deltaY: -100, clientX: 400, clientY: 300 })
  expect(fitSelected()).toBe('false')
  expect(parseFloat(image.style.width)).toBeGreaterThan(800)
  fireEvent.keyDown(stage, { key: '1' })
  expect(image.style.width).toBe('1600px')
  fireEvent.keyDown(stage, { key: '-' })
  expect(image.style.width).toBe('1280px')
  fireEvent.doubleClick(stage)
  expect(fitSelected()).toBe('true')
})

// AC: @donut-create-plugin ac-preview-fit
test('preserves touch pinch and drag controls before returning to centered Fit', () => {
  render(<CreateResultPreview src="synthetic" imageKey="image" alt="Synthetic output" />)
  const image = loadImage(1600, 1200)
  const stage = screen.getByLabelText('Image preview. Scroll to zoom, drag to pan.')
  const pointer = (type, pointerId, clientX, clientY) => {
    const event = new Event(type, { bubbles: true })
    Object.assign(event, { pointerId, pointerType: 'touch', clientX, clientY })
    fireEvent(stage, event)
  }
  pointer('pointerdown', 1, 300, 300)
  pointer('pointerdown', 2, 500, 300)
  pointer('pointermove', 2, 700, 300)
  expect(image.style.width).toBe('1600px')
  expect(image.style.left).toBe('calc(50% + 100px)')
  pointer('pointerup', 2, 700, 300)
  pointer('pointermove', 1, 340, 300)
  expect(image.style.left).toBe('calc(50% + 140px)')
  pointer('pointerup', 1, 340, 300)
  fireEvent.click(screen.getByRole('button', { name: 'Fit', exact: true }))
  expect(fitSelected()).toBe('true')
  expect(image.style.left).toBe('')
})
