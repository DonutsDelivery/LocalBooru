import { afterEach, expect, test, vi } from 'vitest'
import { apiClient, fetchCreateDestinations } from './api'
import { toast } from './components/Toast'
import { getCreateStatus, getStudioSession, stopCreateBackend } from './services/donutCreate'

const adapter = apiClient.defaults.adapter
afterEach(() => { apiClient.defaults.adapter = adapter; vi.restoreAllMocks() })

// AC: @donut-create-plugin ac-family-mode-save
test('Create fetches save destinations independently of gallery directory filters', async () => {
  const signal = new AbortController().signal
  const data = { directories: [{ id: 7, library_id: 'synthetic-library', name: 'Hidden output', show_images: true }] }
  apiClient.defaults.adapter = async config => {
    expect(config.url).toBe('/create/destinations')
    expect(config.signal).toBe(signal)
    return { data, status: 200, statusText: 'OK', headers: {}, config }
  }
  await expect(fetchCreateDestinations({ signal })).resolves.toEqual(data)
})

// AC: @donut-create-plugin ac-quiet-backend-stop
test('real API interceptor keeps repeated background shutdown errors quiet but rejects them', async () => {
  vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 60_000)
  const notify = vi.spyOn(toast, 'error')
  apiClient.defaults.adapter = async config => {
    throw { config, response: { status: 502, data: { detail: 'Synthetic ComfyUI shutdown' } } }
  }
  for (let i = 0; i < 3; i++) {
    await expect(getCreateStatus()).rejects.toMatchObject({ response: { status: 502 } })
    await expect(getStudioSession('synthetic-session')).rejects.toMatchObject({ response: { status: 502 } })
  }
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(notify).not.toHaveBeenCalled()
  await expect(stopCreateBackend()).rejects.toMatchObject({ response: { status: 502 } })
  await vi.waitFor(() => expect(notify).toHaveBeenCalledTimes(1))
  expect(notify.mock.calls[0][0]).toContain('POST /addons/donut-create/api/create/backend/stop')
  await expect(apiClient.get('/synthetic-unrelated')).rejects.toMatchObject({ response: { status: 502 } })
  await vi.waitFor(() => expect(notify).toHaveBeenCalledTimes(2))
})
