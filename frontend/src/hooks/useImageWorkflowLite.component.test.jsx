import { afterEach, expect, test, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import { EditionProvider } from '../contexts/EditionContext'
import { useImageWorkflow } from './useImageWorkflow'
import { adjustmentLocator } from '../utils/imageAdjustments'

const workflow = vi.hoisted(() => ({ getImageWorkflow: vi.fn(), openImageWorkflow: vi.fn(), createErrorMessage: String }))
vi.mock('../services/donutCreate', () => workflow)
afterEach(() => { cleanup(); delete window.__DMC_EDITION_CAPABILITIES__; vi.clearAllMocks() })

test('lite image browsing neither probes nor launches the executable workflow controller', async () => {
  window.__DMC_EDITION_CAPABILITIES__ = { edition: 'store-lite', downloaded_executable_addons: false }
  const image = { id: 1, directory_id: 2, library_id: 'synthetic', file_hash: 'abcdef1234567890', filename: 'synthetic.png' }
  expect(adjustmentLocator(image).imageId).toBe(1)
  const { result } = renderHook(() => useImageWorkflow(image, true), { wrapper: EditionProvider })
  expect(result.current.available).toBe(false)
  await act(async () => { expect(await result.current.loadWorkflow()).toBe(false) })
  expect(workflow.getImageWorkflow).not.toHaveBeenCalled()
  expect(workflow.openImageWorkflow).not.toHaveBeenCalled()
})
