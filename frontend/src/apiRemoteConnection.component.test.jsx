import { beforeEach, expect, it, vi } from 'vitest'
import { apiClient } from './api'

const { toastError } = vi.hoisted(() => ({ toastError: vi.fn() }))
vi.mock('./components/Toast', () => ({ toast: { error: toastError } }))

const interceptor = apiClient.interceptors.response.handlers.at(-1)
const remoteBase = 'http://127.0.0.1:8790/remote/api'
const failure = (url = '/library/stats', method = 'get') => ({
  config: { url, method, baseURL: remoteBase },
  response: { status: 502, data: 'Proxy error: error sending request for url (http://example.invalid/api)' },
})
async function reject(error) {
  await expect(interceptor.rejected(error)).rejects.toBe(error)
  await vi.dynamicImportSettled()
}

beforeEach(() => {
  vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 60_000)
  toastError.mockClear()
  interceptor.fulfilled({ config: { baseURL: remoteBase } })
})

it('rejects each failed read but displays only one actionable remote outage notice', async () => {
  await reject(failure())
  await reject(failure('/images?page=1'))
  await reject(failure('/library/stats'))
  expect(toastError).toHaveBeenCalledTimes(1)
  expect(toastError.mock.calls[0][0]).toContain('Open Servers')
})

it('local health responses do not reset a remote outage; remote recovery does', async () => {
  await reject(failure())
  interceptor.fulfilled({ config: { baseURL: 'http://127.0.0.1:8790/api' } })
  await reject(failure())
  expect(toastError).toHaveBeenCalledTimes(1)
  interceptor.fulfilled({ config: { baseURL: remoteBase } })
  await reject(failure())
  expect(toastError).toHaveBeenCalledTimes(2)
})

it('mutation and non-proxy errors retain their individual reports', async () => {
  await reject(failure('/collections', 'post'))
  await reject(failure('/collections', 'post'))
  const ordinary = failure()
  ordinary.response.data = 'Bad Gateway'
  await reject(ordinary)
  expect(toastError).toHaveBeenCalledTimes(3)
})
