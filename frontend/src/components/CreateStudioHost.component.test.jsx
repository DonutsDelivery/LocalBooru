import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { StrictMode } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

const bridgeFixture = vi.hoisted(() => ({ current: null }))
vi.mock('../services/createStudioBridge', async importOriginal => {
  const original = await importOriginal()
  return { ...original, createStudioBridge: options => bridgeFixture.current || original.createStudioBridge(options) }
})

const api = vi.hoisted(() => ({
  apiClient: { get: vi.fn(), post: vi.fn() },
  getApiUrl: vi.fn(() => '/remote/api'),
  getAddon: vi.fn(), getAddons: vi.fn(),
  installAddon: vi.fn(), startAddon: vi.fn(),
  fetchDirectories: vi.fn(), fetchLibraries: vi.fn(),
  fetchCollections: vi.fn(), getSavedSearches: vi.fn(),
}))
vi.mock('../api', async importOriginal => ({ ...await importOriginal(), ...api }))
vi.mock('./FamilyModeLock', () => ({ default: () => null }))
vi.mock('./Music/PersistentMusicPlayer', () => ({ default: () => null }))
vi.mock('./Sidebar/TagSearch', () => ({ default: () => null }))

import CreateStudioHost from './CreateStudioHost'
import CreateSettings from './CreateSettings'
import Sidebar from './Sidebar/Sidebar'
import AddonSettings from './AddonSettings'
import ToastContainer, { toast } from './Toast'
import { getCreateStatus, startCreateBackend, stopCreateBackend } from '../services/donutCreate'

const CREATE_API = '/addons/donut-create/api/create'
let status
let session

function openStudio() {
  act(() => { window.dispatchEvent(new CustomEvent('donut-create-open')) })
}

function serverEvent(name) {
  act(() => { window.dispatchEvent(new CustomEvent(name)) })
}

function deferred() {
  let resolve
  const promise = new Promise(done => { resolve = done })
  return { promise, resolve }
}

function selectNewServer() {
  session = { ...session, id: 'new-server-session', outputs: [
    { id: 'new-output', filename: 'new-server.png', media_type: 'image/png', type: 'output', prompt_id: 'new-job' },
  ] }
  api.fetchLibraries.mockResolvedValue({ libraries: [{ uuid: 'library-new', name: 'New server', mounted: true, accessible: true }] })
  api.fetchDirectories.mockResolvedValue({ directories: [{ id: 77, library_id: 'library-new', name: 'Fresh images', show_images: true, enabled: true, path_exists: true }] })
}

async function reconnectStudio() {
  const launch = await screen.findByRole('button', { name: 'Open studio', exact: true })
  await waitFor(() => expect(launch.disabled).toBe(false))
  fireEvent.click(launch)
  return screen.findByTitle('DonutUI creation studio')
}

beforeEach(() => {
  vi.clearAllMocks()
  bridgeFixture.current = null
  status = {
    mode: 'managed', backend_url: 'http://127.0.0.1:18010',
    backend: { ready: true, running: true, owned: true },
    setup: {
      state: 'idle', running: false, default_runtime: 'cpu', default_profile: 'base',
      supported_runtimes: ['cpu'], disk: { available_bytes: 100_000_000_000 },
      catalog: { profiles: [
        { id: 'base', label: 'Neutral starter', description: 'Workflow v5 with no aesthetic LoRA.', model_count: 6, total_bytes: 32_273_722_868 },
        { id: 'workflow', label: 'Original workflow v5', description: 'Original v5 model and LoRA selections.', model_count: 8, total_bytes: 45_872_604_519 },
        { id: 'all', label: 'Complete catalog', description: 'All catalog models.', model_count: 16, total_bytes: 63_162_364_113 },
      ] },
    },
  }
  session = {
    id: 'scoped-session', backend_url: status.backend_url,
    studio_path: '/api/create/studio/scoped-session/',
    jobs: [{ id: 'owned-job', status: 'running' }],
    outputs: [
      { id: 'output-1', filename: 'synthetic.png', media_type: 'image/png', type: 'output', prompt_id: 'owned-job' },
      { id: 'video-1', filename: 'synthetic.mp4', media_type: 'video/mp4', type: 'output', prompt_id: 'owned-job' },
    ],
  }
  api.getAddon.mockResolvedValue({ addon: { id: 'donut-create', installed: true, status: 'running' }, status: 'running' })
  api.getAddons.mockResolvedValue({ addons: [{ id: 'donut-create', installed: true }] })
  api.installAddon.mockResolvedValue({ success: true })
  api.startAddon.mockResolvedValue({ success: true })
  api.fetchCollections.mockResolvedValue({ collections: [] })
  api.getSavedSearches.mockResolvedValue({ searches: [] })
  api.fetchLibraries.mockResolvedValue({ libraries: [
    { uuid: 'library-a', name: 'Primary', mounted: true, accessible: true },
    { uuid: 'library-b', name: 'Archive', mounted: true, accessible: true },
    { uuid: 'offline', name: 'Offline', mounted: false },
    { uuid: 'readonly', name: 'Read only', mounted: true, read_only: true },
  ] })
  api.fetchDirectories.mockResolvedValue({ directories: [
    { id: 1, library_id: 'library-a', name: 'Images A', show_images: true, path_exists: true },
    { id: 1, library_id: 'library-b', name: 'Images B', show_images: true, path_exists: true },
    { id: 2, library_id: 'library-a', name: 'Videos only', show_images: false, path_exists: true },
    { id: 3, library_id: 'library-a', name: 'Unavailable', show_images: true, path_exists: false },
    { id: 4, library_id: 'library-a', name: 'Disabled', show_images: true, enabled: false },
    { id: 5, library_id: 'offline', name: 'Offline images', show_images: true },
    { id: 6, library_id: 'readonly', name: 'Read only images', show_images: true },
  ] })
  api.apiClient.get.mockImplementation(async url => ({ data: url.endsWith('/status') ? status : session }))
  api.apiClient.post.mockImplementation(async (url, body) => {
    if (url === `${CREATE_API}/sessions`) return { data: session }
    if (url === '/create/import') return { data: {
      library_id: body.library_id, directory_id: body.directory_id,
      image_id: 42, sha256: 'synthetic-sha256', status: 'imported',
    } }
    return { data: { success: true } }
  })
})

afterEach(() => { cleanup(); vi.useRealTimers() })

describe('Donut Create studio', () => {
  test('shows an acknowledged queue submission before the next session poll', async () => {
    session.jobs = []
    session.outputs = []
    const snapshot = { ready: true, fields: { prompt: { value: 'Synthetic landscape', available: true } } }
    bridgeFixture.current = {
      dispose: vi.fn(),
      request: vi.fn(async action => action === 'generate'
        ? { ...snapshot, lastQueuedPromptIds: ['new-queued-id'] } : snapshot),
    }
    render(<CreateStudioHost />)
    openStudio()
    const generate = await screen.findByRole('button', { name: 'Generate image', exact: true })
    await waitFor(() => expect(generate.disabled).toBe(false))
    const pollsBefore = api.apiClient.get.mock.calls.filter(([url]) => url.includes('/sessions/')).length
    fireEvent.click(generate)
    await screen.findByText('1 active')
    const details = screen.getByText('Studio jobs').closest('details')
    details.open = true
    expect(within(details).getByText('new-queu')).toBeTruthy()
    expect(within(details).getByText('queued')).toBeTruthy()
    expect(api.apiClient.get.mock.calls.filter(([url]) => url.includes('/sessions/'))).toHaveLength(pollsBefore)
  })

  test('does not let a session poll started before submission erase its queue acknowledgement', async () => {
    session.jobs = []
    session.outputs = []
    const oldPoll = deferred()
    api.apiClient.get.mockImplementation(async url => ({ data: url.endsWith('/status') ? status : await oldPoll.promise }))
    const snapshot = { ready: true, fields: { prompt: { value: 'Synthetic landscape', available: true } } }
    bridgeFixture.current = { dispose: vi.fn(), request: vi.fn(async action => action === 'generate'
      ? { ...snapshot, lastQueuedPromptIds: ['new-queued-id'] } : snapshot) }
    render(<CreateStudioHost />)
    openStudio()
    const generate = await screen.findByRole('button', { name: 'Generate image', exact: true })
    await waitFor(() => expect(generate.disabled).toBe(false))
    await waitFor(() => expect(api.apiClient.get.mock.calls.some(([url]) => url.includes('/sessions/'))).toBe(true))
    fireEvent.click(generate)
    await screen.findByText('1 active')
    await act(async () => { oldPoll.resolve(session); await oldPoll.promise })
    expect(screen.getByText('1 active')).toBeTruthy()
  })

  test('does not let an older session poll reopen an acknowledged cancellation', async () => {
    const oldPoll = deferred()
    api.apiClient.get.mockImplementation(async url => ({ data: url.endsWith('/status') ? status : await oldPoll.promise }))
    api.apiClient.post.mockImplementation(async url => ({ data: url.endsWith('/cancel')
      ? { cancelled: ['owned-job'], session: { ...session, jobs: [{ id: 'owned-job', status: 'cancelled' }] } }
      : session }))
    render(<CreateStudioHost />)
    openStudio()
    const cancel = await screen.findByRole('button', { name: 'Cancel studio jobs' })
    await waitFor(() => expect(api.apiClient.get.mock.calls.some(([url]) => url.includes('/sessions/'))).toBe(true))
    fireEvent.click(cancel)
    await screen.findByText('cancelled')
    await act(async () => { oldPoll.resolve(session); await oldPoll.promise })
    expect(screen.getByText('cancelled')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Cancel studio jobs' })).toBeNull()
  })

  test('keeps window controls and notifications interactive while the gallery is inert', async () => {
    const windowAction = vi.fn()
    const { container } = render(<>
      <div className="title-bar"><button onClick={windowAction}>Window action</button></div>
      <main data-testid="gallery">Gallery</main>
      <ToastContainer />
      <CreateStudioHost />
    </>)
    const titlebar = container.querySelector('.title-bar')
    const gallery = screen.getByTestId('gallery')
    titlebar.inert = false
    gallery.inert = false
    act(() => { toast.info('Synthetic window notification', 0) })
    openStudio()
    await screen.findByTitle('DonutUI creation studio')
    expect(titlebar.inert).toBe(false)
    expect(container.querySelector('.toast-container').inert).not.toBe(true)
    expect(gallery.inert).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Window action' }))
    expect(windowAction).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole('button', { name: 'Exit', exact: true }))
    expect(gallery.inert).toBe(false)
  })

  test('places studio navigation over the sidebars and preserves it through setup and advanced mode', async () => {
    const { container } = render(<CreateStudioHost />)
    openStudio()
    const frame = await screen.findByTitle('DonutUI creation studio')
    const exit = screen.getByRole('button', { name: 'Exit', exact: true })
    expect(exit.closest('.create-exit-slot')).toBeTruthy()
    expect(exit.closest('.create-simple-workspace')).toBeTruthy()
    const advanced = screen.getByRole('button', { name: 'Advanced', exact: true })
    expect(advanced.closest('.create-mode-actions')).toBeTruthy()
    expect(container.querySelector('.create-studio-header')).toBeNull()
    advanced.focus()
    fireEvent.click(advanced)
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Advanced', exact: true }))
    expect(frame.hidden).toBe(false)
    expect(screen.getByRole('button', { name: 'Exit', exact: true }).closest('.create-simple-workspace')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Setup', exact: true }))
    await screen.findByRole('button', { name: 'Return to studio' })
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Setup', exact: true }))
    expect(screen.getByRole('button', { name: 'Exit', exact: true })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Simple', exact: true }))
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Simple', exact: true }))
    expect(screen.getByTitle('DonutUI creation studio')).toBe(frame)
    expect(screen.getByRole('button', { name: 'Exit', exact: true }).closest('.create-simple-workspace')).toBeTruthy()
  })

  test('keeps generation progress inside the compact badge while inspecting a previous result', async () => {
    let socket
    vi.stubGlobal('WebSocket', class {
      constructor() { socket = this }
      close() {}
    })
    try {
      const { container } = render(<CreateStudioHost />)
      openStudio()
      await screen.findByTitle('DonutUI creation studio')
      await waitFor(() => expect(socket?.onmessage).toBeTypeOf('function'))
      act(() => socket.onmessage({ data: JSON.stringify({ type: 'progress', data: { value: 88, max: 100 } }) }))
      const progress = await screen.findByRole('progressbar', { name: 'Generation progress' })
      expect(progress.value).toBe(88)
      expect(progress.closest('.create-preview-run-status')).toBeTruthy()
      expect(container.querySelector('.create-large-preview > progress')).toBeNull()
      fireEvent.click(screen.getByRole('tab', { name: 'History', exact: true }))
      fireEvent.click(await screen.findByRole('button', { name: 'Preview synthetic.png', exact: true }))
      expect(screen.getByRole('progressbar', { name: 'Generation progress' }).value).toBe(88)
      expect(screen.getByText('Selected image')).toBeTruthy()
    } finally { cleanup(); vi.unstubAllGlobals() }
  })

  test('groups final batch images by run and collapses intermediate stages', async () => {
    session.outputs = [
      { id: 'stage-a', filename: 'first-pass.png', prompt_id: 'run-a', final: false, media_type: 'image/png' },
      { id: 'final-a1', filename: 'final-one.png', prompt_id: 'run-a', final: true, media_type: 'image/png' },
      { id: 'final-a2', filename: 'final-two.png', prompt_id: 'run-a', final: true, media_type: 'image/png' },
      { id: 'final-b', filename: 'next-run.png', prompt_id: 'run-b', final: true, media_type: 'image/png' },
    ]
    render(<CreateStudioHost />)
    openStudio()
    await screen.findByTitle('DonutUI creation studio')
    fireEvent.click(screen.getByRole('tab', { name: 'History', exact: true }))
    const runs = screen.getAllByRole('region', { name: /Generation run/ })
    expect(runs).toHaveLength(2)
    expect(runs[0].getAttribute('aria-label')).toBe('Generation run run-b')
    const run = screen.getByRole('region', { name: 'Generation run run-a' })
    expect(within(run).getByText('2 results')).toBeTruthy()
    expect(within(run).getByRole('button', { name: 'Preview final-one.png' })).toBeTruthy()
    expect(within(run).getByRole('button', { name: 'Preview final-two.png' })).toBeTruthy()
    const stages = within(run).getByText('Stages (1)').closest('details')
    expect(stages.open).toBe(false)
    stages.open = true
    fireEvent.click(within(stages).getByRole('button', { name: 'Preview first-pass.png' }))
    expect(screen.getByText('Selected image')).toBeTruthy()
  })

  test('notifies failures through existing toast UI without adding a studio banner', async () => {
    const { container } = render(<><ToastContainer /><CreateStudioHost /></>)
    openStudio()
    const cancel = await screen.findByRole('button', { name: 'Cancel studio jobs' })
    api.apiClient.post.mockRejectedValueOnce(new Error('Synthetic cancellation failure'))
    fireEvent.click(cancel)
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('Synthetic cancellation failure')
    expect(alert.closest('.toast-container')).toBeTruthy()
    expect(alert.closest('.toast-container').inert).not.toBe(true)
    expect(container.querySelector('.create-studio-dialog > .create-message.error')).toBeNull()
  })

  test('offers owned backend stop and restart from the right sidebar without opening Setup', async () => {
    render(<CreateStudioHost />)
    openStudio()
    await screen.findByTitle('DonutUI creation studio')
    const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
    expect(summary.closest('.create-mode-actions')).toBeTruthy()
    summary.closest('details').open = true
    const restart = await screen.findByRole('button', { name: 'Restart ComfyUI' })
    await waitFor(() => expect(restart.disabled).toBe(false))
    fireEvent.click(restart)
    await waitFor(() => expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/backend/restart'))).toBe(true))
    expect(screen.getByTitle('DonutUI creation studio')).toBeTruthy()
  })

  test('shows disabled backend actions and management instructions for a URL-only connection', async () => {
    status = { ...status, mode: 'existing', backend: { ...status.backend, owned: false } }
    render(<CreateSettings backendOnly />)
    const summary = await screen.findByText('ComfyUI', { selector: 'summary' })
    expect(summary.getAttribute('aria-label')).toBe('ComfyUI backend controls')
    expect(summary.querySelector('svg')).toBeTruthy()
    summary.closest('details').open = true
    const start = await screen.findByRole('button', { name: 'Start ComfyUI' })
    expect(start.disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Stop ComfyUI' }).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Restart ComfyUI' }).disabled).toBe(true)
    expect(screen.getByText(/select Manage existing local installation/)).toBeTruthy()
    fireEvent.click(start)
    expect(api.apiClient.post).not.toHaveBeenCalled()
  })

  test.each(['before', 'during'])('discards status polls begun %s a backend stop across all views', async when => {
    const oldPoll = deferred(), stopping = deferred()
    api.apiClient.get.mockReturnValueOnce(oldPoll.promise)
    api.apiClient.post.mockReturnValueOnce(stopping.promise)
    let poll, stop
    if (when === 'before') {
      poll = getCreateStatus()
      stop = stopCreateBackend()
    } else {
      stop = stopCreateBackend()
      poll = getCreateStatus()
    }
    const rejected = expect(poll).rejects.toMatchObject({ name: 'AbortError' })
    stopping.resolve({ data: { ...status, backend: { ready: false, running: false, owned: false } } })
    expect((await stop).backend.owned).toBe(false)
    oldPoll.resolve({ data: status })
    await rejected
  })

  test('a delayed start cannot overwrite a newer stop acknowledgement', async () => {
    const starting = deferred()
    api.apiClient.post.mockReturnValueOnce(starting.promise)
    const start = startCreateBackend()
    const rejected = expect(start).rejects.toMatchObject({ name: 'AbortError' })
    api.apiClient.post.mockResolvedValueOnce({ data: { ...status, backend: { owned: false, ready: false, running: false } } })
    expect((await stopCreateBackend()).backend.owned).toBe(false)
    starting.resolve({ data: status })
    await rejected
  })

  // AC: @donut-create-plugin ac-save-gallery
  test('offers exclusive ComfyUI saving and never automatically copies permanent outputs', async () => {
    render(<CreateStudioHost />)
    openStudio()
    await screen.findByTitle('DonutUI creation studio')
    fireEvent.click(await screen.findByRole('tab', { name: 'Finalize' }))
    const destination = await screen.findByLabelText('Output destination')
    fireEvent.change(destination, { target: { value: '__comfy_output__' } })
    const automatic = screen.getByRole('switch', { name: 'Save output' })
    expect(automatic.disabled).toBe(false)
    fireEvent.click(automatic)
    expect(screen.getByText(/Save output keeps images in ComfyUI only/)).toBeTruthy()
    expect(api.apiClient.post.mock.calls.filter(([url]) => url === '/create/import')).toEqual([])
    fireEvent.change(destination, { target: { value: 'library-b:1' } })
    session = {...session, outputs: [...session.outputs, {id:'permanent-new',filename:'comfy.png',type:'output',
      storage:'comfy',final:true,prompt_id:'owned-job',media_type:'image/png'}]}
    fireEvent.click(screen.getByRole('tab', { name: 'History' }))
    await screen.findAllByText('comfy.png', {}, {timeout:5000})
    expect(api.apiClient.post.mock.calls.filter(([url]) => url === '/create/import')).toEqual([])
    session = {...session, outputs: [...session.outputs, {id:'temporary-new',filename:'staged.png',type:'temp',
      storage:'temporary',final:true,prompt_id:'owned-job',media_type:'image/png'}]}
    await waitFor(() => expect(api.apiClient.post.mock.calls.filter(([url]) => url === '/create/import'))
      .toEqual([[ '/create/import', expect.objectContaining({output_id:'temporary-new',library_id:'library-b',directory_id:1}),
        expect.objectContaining({signal:expect.any(AbortSignal)}) ]]), {timeout:5000})
  }, 10000)
  // AC: @donut-create-plugin ac-image-entry
  test('opens from Images without navigation or changing the gallery scroll', async () => {
    const { container } = render(<MemoryRouter initialEntries={['/?tags=landscape&directory=1']}>
      <div className="masonry-container" />
      <Sidebar tags={[]} onSearch={vi.fn()} mediaType="image" />
      <CreateStudioHost />
    </MemoryRouter>)
    const gallery = container.querySelector('.masonry-container')
    gallery.scrollTop = 820
    const createButton = screen.getByRole('button', { name: 'Create', exact: true })
    createButton.focus()
    fireEvent.click(createButton)
    expect(await screen.findByRole('dialog', { name: 'Create images' })).toBeTruthy()
    await screen.findByTitle('DonutUI creation studio')
    expect(gallery.scrollTop).toBe(820)
    expect(screen.getByRole('link', { name: 'Images' }).getAttribute('aria-current')).toBe('page')
    fireEvent.click(screen.getByRole('button', { name: 'Exit' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(gallery.scrollTop).toBe(820)
    expect(document.activeElement).toBe(createButton)
  })

  // AC: @donut-create-plugin ac-image-entry
  test.each([['/videos', 'video'], ['/music', 'music']])('does not offer creation on %s', async (path, mediaType) => {
    render(<MemoryRouter initialEntries={[path]}><Sidebar tags={[]} onSearch={vi.fn()} mediaType={mediaType} /></MemoryRouter>)
    await waitFor(() => expect(api.fetchDirectories).toHaveBeenCalled())
    expect(screen.queryByRole('button', { name: 'Create', exact: true })).toBeNull()
  })

  // AC: @donut-create-plugin ac-workflow-state
  test('keeps the edited iframe and owned queue mounted through setup, close and reopen', async () => {
    render(<CreateStudioHost />)
    openStudio()
    const frame = await screen.findByTitle('DonutUI creation studio')
    frame.contentDocument.open()
    frame.contentDocument.write('<html><body><input aria-label="Workflow prompt" value="initial" /></body></html>')
    frame.contentDocument.close()
    const workflowPrompt = frame.contentDocument.querySelector('input')
    workflowPrompt.value = 'edited studio prompt'
    fireEvent.click(screen.getByRole('button', { name: 'Setup', exact: true }))
    expect(frame.isConnected).toBe(true)
    expect(frame.hidden).toBe(true)
    await screen.findByRole('button', { name: 'Return to studio' })
    fireEvent.click(screen.getByRole('button', { name: 'Return to studio' }))
    expect(frame.hidden).toBe(true) // Basic view retains the workflow engine without showing its graph.
    fireEvent.click(screen.getByRole('button', { name: 'Advanced', exact: true }))
    expect(frame.hidden).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Exit' }))
    openStudio()
    expect(screen.getByTitle('DonutUI creation studio')).toBe(frame)
    expect(frame.contentDocument.querySelector('input').value).toBe('edited studio prompt')
    expect(screen.getByText('owned-jo')).toBeTruthy()
    expect(api.apiClient.post.mock.calls.filter(([url]) => url === `${CREATE_API}/sessions`)).toHaveLength(1)
    expect(frame.getAttribute('src')).toBe('/remote/api/create/studio/scoped-session/')
    expect(frame.getAttribute('src')).not.toMatch(/token=|auth=|Bearer/)
  })

  // AC: @donut-create-plugin ac-save-gallery
  test('requires an available image destination and saves a qualified result with a gallery event', async () => {
    const imported = vi.fn()
    window.addEventListener('donut-create-imported', imported)
    try {
      render(<CreateStudioHost />)
      openStudio()
      const save = await screen.findByRole('button', { name: 'Save to library', exact: true })
      expect(save.disabled).toBe(true)
      fireEvent.click(await screen.findByRole('tab', { name: 'Finalize' }))
      const destination = await screen.findByLabelText('Output destination')
      expect(Array.from(destination.options).map(option => option.text)).toEqual([
        'Choose a directory', 'ComfyUI output folder', 'Archive · Images B', 'Primary · Images A',
      ])
      fireEvent.change(destination, { target: { value: 'library-b:1' } })
      fireEvent.click(save)
      await screen.findByRole('button', { name: 'Saved to library' })
      expect(api.apiClient.post).toHaveBeenCalledWith('/create/import', {
        session_id: 'scoped-session', output_id: 'output-1', library_id: 'library-b', directory_id: 1,
      }, expect.objectContaining({ signal: expect.any(AbortSignal) }))
      expect(imported).toHaveBeenCalledTimes(1)
      expect(imported.mock.calls[0][0].detail).toMatchObject({ library_id: 'library-b', directory_id: 1, image_id: 42 })
      expect(screen.getByRole('img', { name: 'synthetic.png' }).getAttribute('src')).toBe('/remote/api/create/output/scoped-session/output-1')
      expect(screen.queryByRole('img', { name: 'synthetic.mp4' })).toBeNull()
      expect(screen.getByRole('button', { name: 'Saved to library' }).disabled).toBe(true)
    } finally {
      window.removeEventListener('donut-create-imported', imported)
    }
  })

  // AC: @donut-create-plugin ac-job-ownership
  test('cancels only the active studio session jobs', async () => {
    render(<CreateStudioHost />)
    openStudio()
    const cancel = await screen.findByRole('button', { name: 'Cancel studio jobs' })
    api.apiClient.post.mockResolvedValueOnce({ data: {
      cancelled: ['owned-job'], errors: [],
      session: { ...session, jobs: [{ id: 'owned-job', status: 'cancelled' }] },
    } })
    fireEvent.click(cancel)
    await waitFor(() => expect(api.apiClient.post).toHaveBeenCalledWith(`${CREATE_API}/sessions/scoped-session/cancel`, undefined, expect.objectContaining({ signal: expect.any(AbortSignal) })))
    await screen.findByText('Cancelled 1 studio job.')
    expect(screen.getByText('cancelled')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Cancel studio jobs' })).toBeNull()
    expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/interrupt'))).toBe(false)
  })

  // AC: @donut-create-plugin ac-job-ownership
  test('shows cancellation errors when the running prompt was left unchanged', async () => {
    const message = 'This backend lacks prompt-specific interruption. The running prompt was left unchanged.'
    render(<><ToastContainer /><CreateStudioHost /></>)
    openStudio()
    const cancel = await screen.findByRole('button', { name: 'Cancel studio jobs' })
    api.apiClient.post.mockResolvedValueOnce({ data: { cancelled: [], errors: [message] } })
    fireEvent.click(cancel)
    expect((await screen.findByRole('alert')).querySelector('.toast-message').textContent).toBe(message)
    expect(screen.getAllByRole('status').some(element => element.textContent === message)).toBe(true)
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Cancel studio jobs' }).disabled).toBe(false)
    expect(screen.queryByText(/Cancellation requested|Cancelled \d+ studio job/)).toBeNull()
    expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/interrupt'))).toBe(false)
  })

  // AC: @donut-create-plugin ac-workflow-state
  test('invalidates the old session when the backend changes', async () => {
    render(<><ToastContainer /><CreateStudioHost /></>)
    openStudio()
    const oldFrame = await screen.findByTitle('DonutUI creation studio')
    act(() => { window.dispatchEvent(new CustomEvent('donut-create-backend-changed')) })
    expect(oldFrame.isConnected).toBe(false)
    expect((await screen.findByRole('alert')).querySelector('.toast-message').textContent).toMatch(/backend changed/)
    session = { ...session, id: 'new-scoped-session', backend_url: 'http://127.0.0.1:8288' }
    status = { ...status, backend_url: session.backend_url }
    const launch = await screen.findByRole('button', { name: 'Open studio', exact: true })
    await waitFor(() => expect(launch.disabled).toBe(false))
    fireEvent.click(launch)
    expect((await screen.findByTitle('DonutUI creation studio')).getAttribute('src')).toContain('new-scoped-session')
  })

  // AC: @donut-create-plugin ac-access-boundary
  // AC: @donut-create-plugin ac-workflow-state
  test('discards the active session and late output replies when the paired server changes at the same proxy URL', async () => {
    const oldSession = { ...session, outputs: [{ id: 'late-output', filename: 'old-server.png', media_type: 'image/png' }] }
    const oldPoll = deferred()
    api.apiClient.get.mockImplementation(async url => {
      if (url === `${CREATE_API}/sessions/scoped-session`) return oldPoll.promise
      return { data: status }
    })
    render(<CreateStudioHost />)
    openStudio()
    const oldFrame = await screen.findByTitle('DonutUI creation studio')
    await waitFor(() => expect(api.apiClient.get.mock.calls.some(([url]) => url === `${CREATE_API}/sessions/scoped-session`)).toBe(true))
    const requestsBeforeSwitch = api.apiClient.get.mock.calls.length
    serverEvent('donut-create-server-changing')
    expect(oldFrame.isConnected).toBe(false)
    expect(screen.queryByRole('img', { name: 'synthetic.png' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Open studio', exact: true }).disabled).toBe(true)
    selectNewServer()
    serverEvent('donut-create-server-changed')
    const newFrame = await reconnectStudio()
    await act(async () => { oldPoll.resolve({ data: oldSession }); await oldPoll.promise })
    expect(newFrame.getAttribute('src')).toBe('/remote/api/create/studio/new-server-session/')
    expect(screen.queryByRole('img', { name: 'old-server.png' })).toBeNull()
    expect(screen.getByRole('img', { name: 'new-server.png' })).toBeTruthy()
    fireEvent.click(await screen.findByRole('tab', { name: 'Finalize' }))
    expect(Array.from((await screen.findByLabelText('Output destination')).options).map(option => option.text)).toEqual(['Choose a directory', 'ComfyUI output folder', 'New server · Fresh images'])
    expect(api.apiClient.get.mock.calls.slice(requestsBeforeSwitch).some(([url]) => url.includes('/sessions/scoped-session'))).toBe(false)
    expect(api.getApiUrl()).toBe('/remote/api')
  })

  // AC: @donut-create-plugin ac-access-boundary
  test('rejects pending session and destination responses from the previous server', async () => {
    const oldSession = { ...session }
    const oldCreation = deferred()
    const oldDirectories = deferred()
    api.apiClient.post.mockImplementationOnce(() => oldCreation.promise)
    api.fetchDirectories.mockImplementationOnce(() => oldDirectories.promise)
    render(<CreateStudioHost />)
    openStudio()
    await waitFor(() => expect(api.apiClient.post).toHaveBeenCalled())
    const oldSignal = api.apiClient.post.mock.calls[0][2].signal
    serverEvent('donut-create-server-changing')
    expect(oldSignal.aborted).toBe(true)
    selectNewServer()
    serverEvent('donut-create-server-changed')
    const newFrame = await reconnectStudio()
    await act(async () => {
      oldCreation.resolve({ data: oldSession })
      oldDirectories.resolve({ directories: [{ id: 1, library_id: 'library-a', name: 'Old destination', show_images: true }] })
      await Promise.all([oldCreation.promise, oldDirectories.promise])
    })
    expect(screen.getByTitle('DonutUI creation studio')).toBe(newFrame)
    expect(newFrame.getAttribute('src')).toContain('new-server-session')
    expect(screen.queryByRole('option', { name: /Old destination/ })).toBeNull()
    fireEvent.click(await screen.findByRole('tab', { name: 'Finalize' }))
    expect(await screen.findByRole('option', { name: 'New server · Fresh images' })).toBeTruthy()
  })

  // AC: @donut-create-plugin ac-access-boundary
  // AC: @donut-create-plugin ac-save-gallery
  test('does not publish a late import response into the newly selected server gallery', async () => {
    const oldImport = deferred()
    const imported = vi.fn()
    window.addEventListener('donut-create-imported', imported)
    try {
      api.apiClient.post.mockImplementation(async url => url === '/create/import' ? oldImport.promise : { data: session })
      render(<CreateStudioHost />)
      openStudio()
      await screen.findByRole('button', { name: 'Save to library', exact: true })
      fireEvent.click(await screen.findByRole('tab', { name: 'Finalize' }))
      fireEvent.change(await screen.findByLabelText('Output destination'), { target: { value: 'library-a:1' } })
      fireEvent.click(screen.getByRole('button', { name: 'Save to library', exact: true }))
      await waitFor(() => expect(api.apiClient.post.mock.calls.some(([url]) => url === '/create/import')).toBe(true))
      serverEvent('donut-create-server-changing')
      selectNewServer()
      serverEvent('donut-create-server-changed')
      await reconnectStudio()
      await act(async () => {
        oldImport.resolve({ data: { library_id: 'library-a', directory_id: 1, image_id: 42, status: 'imported' } })
        await oldImport.promise
      })
      expect(imported).not.toHaveBeenCalled()
      expect(screen.getByRole('button', { name: 'Save to library', exact: true }).disabled).toBe(true)
      expect(screen.queryByRole('button', { name: 'Saved to library' })).toBeNull()
    } finally {
      window.removeEventListener('donut-create-imported', imported)
    }
  })

  // AC: @donut-create-plugin ac-image-entry
  test('launches normally after the initial server configuration events', async () => {
    render(<StrictMode><CreateStudioHost /></StrictMode>)
    serverEvent('donut-create-server-changing')
    serverEvent('donut-create-server-changed')
    openStudio()
    expect((await screen.findByTitle('DonutUI creation studio')).getAttribute('src')).toContain('scoped-session')
    expect(api.apiClient.post.mock.calls.filter(([url]) => url === `${CREATE_API}/sessions`)).toHaveLength(1)
  })
})

describe('Donut Create setup', () => {
  // AC: @donut-create-plugin ac-setup-recovery
  test('serializes slow readiness polls per server generation and ignores stale completions', async () => {
    vi.useFakeTimers()
    const oldStatus = deferred()
    const newStatus = deferred()
    api.apiClient.get.mockImplementationOnce(() => oldStatus.promise)
    api.apiClient.get.mockImplementationOnce(() => newStatus.promise)
    render(<CreateSettings />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(api.apiClient.get).toHaveBeenCalledTimes(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(7500) })
    expect(api.apiClient.get).toHaveBeenCalledTimes(1)
    serverEvent('donut-create-server-changing')
    serverEvent('donut-create-server-changed')
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(api.apiClient.get).toHaveBeenCalledTimes(2)
    await act(async () => { oldStatus.resolve({ data: status }); await oldStatus.promise })
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(api.apiClient.get).toHaveBeenCalledTimes(2)
    await act(async () => { newStatus.resolve({ data: status }); await newStatus.promise })
    await act(async () => { await vi.advanceTimersByTimeAsync(2500) })
    expect(api.apiClient.get).toHaveBeenCalledTimes(3)
    expect(screen.getByLabelText('Runtime').value).toBe('cpu')
  })

  // AC: @donut-create-plugin ac-setup-recovery
  test('can stop the owned backend while its start request is still pending', async () => {
    const starting = deferred()
    status.backend = { ready: false, running: false, owned: false }
    api.apiClient.post.mockImplementation(async url => {
      if (url.endsWith('/backend/start')) {
        status = { ...status, backend: { ready: false, running: false, owned: true } }
        return starting.promise
      }
      if (url.endsWith('/backend/stop')) status = { ...status, backend: { ready: false, running: false, owned: false } }
      return { data: { success: true } }
    })
    render(<CreateSettings />)
    await screen.findByText('Creator add-on is running')
    fireEvent.click(screen.getByRole('button', { name: 'Start ComfyUI' }))
    await waitFor(() => expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/backend/start'))).toBe(true))
    act(() => { window.dispatchEvent(new CustomEvent('localbooru-addons-changed')) })
    const stop = screen.getByRole('button', { name: 'Stop ComfyUI' })
    await waitFor(() => expect(stop.disabled).toBe(false))
    fireEvent.click(stop)
    await waitFor(() => expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/backend/stop'))).toBe(true))
    expect(api.apiClient.post.mock.calls.find(([url]) => url.endsWith('/backend/start'))[2].signal.aborted).toBe(true)
    await act(async () => { starting.resolve({ data: { backend: { ready: true, running: true, owned: true } } }); await starting.promise })
    expect(screen.getByText('Backend is stopped or unavailable')).toBeTruthy()
    expect(stop.disabled).toBe(true)
  })

  // AC: @donut-create-plugin ac-setup-recovery
  test('displays catalog sizes before setup and submits transient tokens with the selected profile', async () => {
    render(<CreateSettings />)
    await screen.findByText('Creator add-on is running')
    expect(screen.getByText(/32.3 GB download/).textContent).toMatch(/6 models.*100.0 GB available/)
    expect(screen.getByLabelText('Runtime').value).toBe('cpu')
    fireEvent.change(screen.getByLabelText('Models'), { target: { value: 'workflow' } })
    expect(screen.getByText(/45.9 GB download/).textContent).toContain('8 models')
    fireEvent.click(screen.getByText('Model access tokens (optional)'))
    fireEvent.change(screen.getByLabelText('Hugging Face token'), { target: { value: 'synthetic-hf-token' } })
    fireEvent.change(screen.getByLabelText('Civitai API key'), { target: { value: 'synthetic-civitai-key' } })
    fireEvent.click(screen.getByRole('button', { name: 'Install selected runtime and models' }))
    await waitFor(() => expect(api.apiClient.post).toHaveBeenCalledWith(`${CREATE_API}/setup`, {
      runtime: 'cpu', profile: 'workflow', hf_token: 'synthetic-hf-token', civitai_api_key: 'synthetic-civitai-key',
    }, expect.objectContaining({ signal: expect.any(AbortSignal) })))
    expect(screen.getByLabelText('Hugging Face token').value).toBe('')
    expect(screen.getByLabelText('Civitai API key').value).toBe('')
    expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/config'))).toBe(false)
  })

  // AC: @donut-create-plugin ac-setup-recovery
  test('shows progress, cancellation and reusable downloads before retrying', async () => {
    status.setup = { ...status.setup, state: 'installing', running: true, phase: 'models',
      downloaded_bytes: 2_000_000_000, total_bytes: 8_000_000_000, progress: 0.25, current_file: 'synthetic-model.safetensors' }
    api.apiClient.post.mockImplementation(async url => {
      if (url.endsWith('/cancel-setup')) status = { ...status, setup: { ...status.setup, state: 'cancelled', running: false } }
      return { data: { success: true } }
    })
    render(<CreateSettings />)
    await screen.findByText('synthetic-model.safetensors')
    expect(screen.getByRole('progressbar', { name: 'Setup download progress' }).value).toBe(25)
    expect(screen.getByRole('button', { name: 'Install selected runtime and models' }).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel setup' }))
    await screen.findByText(/Verified downloads will be reused/)
    fireEvent.click(screen.getByRole('button', { name: 'Retry setup' }))
    await waitFor(() => expect(api.apiClient.post).toHaveBeenCalledWith(`${CREATE_API}/setup`, { runtime: 'cpu', profile: 'base' }, expect.objectContaining({ signal: expect.any(AbortSignal) })))
  })

  // AC: @donut-create-plugin ac-setup-recovery
  test('shows a failed download and allows an explicit retry', async () => {
    status.setup = { ...status.setup, state: 'error', error: 'Model download requires a Hugging Face token.' }
    render(<CreateSettings />)
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(screen.getByRole('alert').textContent).toMatch(/requires a Hugging Face token/)
    fireEvent.click(screen.getByRole('button', { name: 'Retry setup' }))
    await waitFor(() => expect(api.apiClient.post).toHaveBeenCalledWith(`${CREATE_API}/setup`, { runtime: 'cpu', profile: 'base' }, expect.objectContaining({ signal: expect.any(AbortSignal) })))
  })

  // AC: @donut-create-plugin ac-managed-setup
  test('activates only the optional controller until managed setup is explicitly requested', async () => {
    api.getAddon.mockResolvedValueOnce({ addon: { id: 'donut-create', installed: false, status: 'not_installed' }, status: 'not_installed' })
    render(<CreateSettings />)
    const install = await screen.findByRole('button', { name: 'Install creator add-on' })
    await waitFor(() => expect(install.disabled).toBe(false))
    expect(screen.getByText(/neutral starter requires about 32 GB/)).toBeTruthy()
    fireEvent.click(install)
    await waitFor(() => expect(api.startAddon).toHaveBeenCalledWith('donut-create', expect.objectContaining({ signal: expect.any(AbortSignal) })))
    expect(api.installAddon).toHaveBeenCalledWith('donut-create', undefined, expect.objectContaining({ signal: expect.any(AbortSignal) }))
    expect(api.apiClient.post).not.toHaveBeenCalled()
  })

  // AC: @donut-create-plugin ac-access-boundary
  test('does not start an add-on on the new server after an old installation finishes', async () => {
    const installation = deferred()
    api.installAddon.mockImplementationOnce(() => installation.promise)
    api.getAddon.mockResolvedValue({ addon: { id: 'donut-create', installed: false, status: 'not_installed' }, status: 'not_installed' })
    render(<CreateSettings />)
    const install = await screen.findByRole('button', { name: 'Install creator add-on' })
    await waitFor(() => expect(install.disabled).toBe(false))
    fireEvent.click(install)
    await waitFor(() => expect(api.installAddon).toHaveBeenCalled())
    serverEvent('donut-create-server-changing')
    serverEvent('donut-create-server-changed')
    await act(async () => { installation.resolve({ success: true }); await installation.promise })
    expect(api.startAddon).not.toHaveBeenCalled()
    expect(api.apiClient.post).not.toHaveBeenCalled()
  })

  // AC: @donut-create-plugin ac-managed-setup
  test('connects to an existing backend without requesting managed downloads', async () => {
    status.backend = { ready: false, running: false, owned: false }
    render(<CreateSettings />)
    await screen.findByText('Creator add-on is running')
    fireEvent.click(screen.getByLabelText('Connect existing ComfyUI'))
    fireEvent.change(screen.getByLabelText('Backend URL'), { target: { value: 'http://127.0.0.1:8288' } })
    fireEvent.click(screen.getByRole('button', { name: 'Connect backend' }))
    await waitFor(() => expect(api.apiClient.post).toHaveBeenCalledWith(`${CREATE_API}/config`, { mode: 'existing', backend_url: 'http://127.0.0.1:8288' }, expect.objectContaining({ signal: expect.any(AbortSignal) })))
    expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/setup'))).toBe(false)
    expect(screen.queryByRole('button', { name: 'Install selected runtime and models' })).toBeNull()
  })

  // AC: @donut-create-plugin ac-image-entry
  test('registers the installed creator in add-on settings', async () => {
    render(<AddonSettings />)
    expect(await screen.findByRole('heading', { name: 'Donut Create' })).toBeTruthy()
  })
})
