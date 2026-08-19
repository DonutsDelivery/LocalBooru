import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  absorbWd14Sidecars: vi.fn(),
  exportWd14Sidecars: vi.fn(),
  fetchDirectories: vi.fn(),
  fetchLibraries: vi.fn(),
  getAddons: vi.fn(),
  importWd14Sidecars: vi.fn(),
}))

vi.mock('../api', async importOriginal => ({
  ...await importOriginal(),
  ...api,
}))

import AddonSettings from './AddonSettings'
import WD14SidecarSettings from './WD14SidecarSettings'

function response(operation, overrides = {}) {
  return {
    operation,
    summary: {
      directories: 2,
      media_candidates: 3,
      sidecars_found: 2,
      sidecars_succeeded: 1,
      sidecars_skipped: 0,
      sidecars_failed: 1,
      tags_parsed: 4,
      tags_added: 2,
      sidecars_written: 0,
      sidecars_removed: 0,
      ...overrides.summary,
    },
    results: overrides.results || [],
  }
}

function setDirectories(count = 2) {
  api.fetchLibraries.mockResolvedValue({
    libraries: [
      { uuid: 'library-a', name: 'Primary', mounted: true },
      { uuid: 'library-b', name: 'Archive', mounted: true },
    ],
  })
  api.fetchDirectories.mockResolvedValue({
    directories: Array.from({ length: count }, (_, index) => ({
      id: (index % 51) + 1,
      library_id: index % 2 === 0 ? 'library-a' : 'library-b',
      name: `Dataset ${index + 1}`,
      path: `/managed/dataset-${index + 1}`,
      image_count: index + 1,
      path_exists: true,
    })),
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  setDirectories()
  globalThis.confirm = vi.fn(() => true)
})

afterEach(() => {
  cleanup()
})

describe('WD14 sidecar settings', () => {
  // AC: @wd14-addon-settings ac-installed-visibility
  test('renders the panel only when the WD14 add-on is installed', async () => {
    api.getAddons.mockResolvedValueOnce({
      addons: [{ id: 'wd14-sidecar', installed: false }],
    })
    const firstRender = render(<AddonSettings />)
    await screen.findByText(/Install a configurable add-on/)
    expect(screen.queryByRole('heading', { name: 'WD14 Text Sidecars' })).toBeNull()
    firstRender.unmount()

    api.getAddons.mockResolvedValueOnce({
      addons: [{ id: 'wd14-sidecar', installed: true }],
    })
    render(<AddonSettings />)
    expect(await screen.findByRole('heading', { name: 'WD14 Text Sidecars' })).toBeTruthy()
  })

  // AC: @wd14-addon-settings ac-managed-selection
  test('caps select-all at the backend limit and clears compound selections', async () => {
    setDirectories(102)
    render(<WD14SidecarSettings />)

    await screen.findByText('Dataset 102')
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }))

    expect(screen.getByText('100 of 102 selected (maximum 100)')).toBeTruthy()
    const directoryCheckboxes = screen.getAllByRole('checkbox')
      .filter(checkbox => checkbox !== screen.getByLabelText(/Overwrite existing sidecars/))
    expect(directoryCheckboxes.filter(checkbox => checkbox.checked)).toHaveLength(100)
    expect(directoryCheckboxes.filter(checkbox => checkbox.disabled)).toHaveLength(2)

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(directoryCheckboxes.filter(checkbox => checkbox.checked)).toHaveLength(0)
  })

  // AC: @wd14-addon-settings ac-operation-controls
  test('locks duplicate submissions and sends only compound identities', async () => {
    let finishImport
    api.importWd14Sidecars.mockReturnValue(new Promise(resolve => { finishImport = resolve }))
    render(<WD14SidecarSettings />)

    await screen.findByText('Dataset 2')
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }))
    const importButton = screen.getByRole('button', { name: 'Import' })
    fireEvent.click(importButton)
    fireEvent.click(importButton)

    expect(api.importWd14Sidecars).toHaveBeenCalledTimes(1)
    expect(api.importWd14Sidecars).toHaveBeenCalledWith([
      { library_id: 'library-b', directory_id: 2 },
      { library_id: 'library-a', directory_id: 1 },
    ])
    expect(JSON.stringify(api.importWd14Sidecars.mock.calls[0][0])).not.toContain('path')
    expect(screen.getByRole('button', { name: 'Import running...' }).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Absorb' }).disabled).toBe(true)

    finishImport(response('import', { summary: { sidecars_succeeded: 2, sidecars_failed: 0 } }))
    await screen.findByText('Import results')
    expect(screen.getByRole('status').textContent).toMatch(/2 succeeded, 0 skipped, and 0 failed/)
  })

  // AC: @wd14-addon-settings ac-operation-controls
  test('honors absorb cancellation and confirms overwrite-enabled export', async () => {
    api.exportWd14Sidecars.mockResolvedValue(response('export'))
    render(<WD14SidecarSettings />)

    await screen.findByText('Dataset 2')
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }))

    globalThis.confirm.mockReturnValueOnce(false)
    fireEvent.click(screen.getByRole('button', { name: 'Absorb' }))
    expect(api.absorbWd14Sidecars).not.toHaveBeenCalled()
    expect(globalThis.confirm).toHaveBeenCalledWith(expect.stringMatching(/permanently deleted/))

    fireEvent.click(screen.getByLabelText(/Overwrite existing sidecars/))
    globalThis.confirm.mockReturnValueOnce(true)
    fireEvent.click(screen.getByRole('button', { name: 'Export' }))

    await waitFor(() => expect(api.exportWd14Sidecars).toHaveBeenCalledTimes(1))
    expect(globalThis.confirm).toHaveBeenLastCalledWith(expect.stringMatching(/atomically overwritten/))
    expect(api.exportWd14Sidecars.mock.calls[0][1]).toBe(true)
  })

  // AC: @wd14-addon-settings ac-visible-results
  test('renders aggregate results and actionable failures without disabling the panel', async () => {
    api.importWd14Sidecars.mockResolvedValue(response('import', {
      results: [{
        sidecar_path: '/managed/dataset-1/broken.txt',
        status: 'failed_read',
        error: 'Permission denied',
      }],
    }))
    render(<WD14SidecarSettings />)

    await screen.findByText('Dataset 2')
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }))
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await screen.findByText('Sidecars requiring attention')
    expect(screen.getByText('Permission denied')).toBeTruthy()
    expect(screen.getByText('/managed/dataset-1/broken.txt')).toBeTruthy()
    expect(screen.getByText('Failed Read')).toBeTruthy()
    expect(screen.getByText('Media candidates').nextElementSibling.textContent).toBe('3')
    expect(screen.getByRole('button', { name: 'Import' }).disabled).toBe(false)
    expect(screen.getByRole('status').textContent).toMatch(/1 succeeded, 0 skipped, and 1 failed/)
  })
})
