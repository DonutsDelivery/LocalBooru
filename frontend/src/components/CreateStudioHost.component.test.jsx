import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { StrictMode } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

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
    fireEvent.click(screen.getByRole('button', { name: 'Close studio' }))
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
    expect(frame.hidden).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Close studio' }))
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
      const destination = screen.getByLabelText('Save to image directory')
      expect(Array.from(destination.options).map(option => option.text)).toEqual([
        'Choose a directory', 'Archive · Images B', 'Primary · Images A',
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
    render(<CreateStudioHost />)
    openStudio()
    const cancel = await screen.findByRole('button', { name: 'Cancel studio jobs' })
    api.apiClient.post.mockResolvedValueOnce({ data: { cancelled: [], errors: [message] } })
    fireEvent.click(cancel)
    expect((await screen.findByRole('alert')).textContent).toBe(message)
    expect(screen.getByRole('status').textContent).toBe(message)
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Cancel studio jobs' }).disabled).toBe(false)
    expect(screen.queryByText(/Cancellation requested|Cancelled \d+ studio job/)).toBeNull()
    expect(api.apiClient.post.mock.calls.some(([url]) => url.endsWith('/interrupt'))).toBe(false)
  })

  // AC: @donut-create-plugin ac-workflow-state
  test('invalidates the old session when the backend changes', async () => {
    render(<CreateStudioHost />)
    openStudio()
    const oldFrame = await screen.findByTitle('DonutUI creation studio')
    act(() => { window.dispatchEvent(new CustomEvent('donut-create-backend-changed')) })
    expect(oldFrame.isConnected).toBe(false)
    expect(screen.getByRole('alert').textContent).toMatch(/backend changed/)
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
    expect(Array.from(screen.getByLabelText('Save to image directory').options).map(option => option.text)).toEqual(['Choose a directory', 'New server · Fresh images'])
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
    expect(screen.getByRole('option', { name: 'New server · Fresh images' })).toBeTruthy()
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
      fireEvent.change(screen.getByLabelText('Save to image directory'), { target: { value: 'library-a:1' } })
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
