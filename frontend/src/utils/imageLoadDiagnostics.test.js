import test from 'node:test'
import assert from 'node:assert/strict'
import { diagnoseImageLoad } from './imageLoadDiagnostics.js'

const source = 'http://127.0.0.1:8790/api/images/7/file?directory_id=2&media_token=private-token'
const jsonResponse = (status, detail) => new Response(JSON.stringify({ detail }), {
  status,
  headers: { 'content-type': 'application/json' },
})

test('ambiguous duplicate response identifies the route failure without exposing source credentials', async () => {
  let requestedSource
  let options
  const result = await diagnoseImageLoad(source, async (url, init) => {
    requestedSource = url
    options = init
    return jsonResponse(400, 'Media target is ambiguous because multiple existing file paths share this image')
  })
  assert.equal(requestedSource, source)
  assert.equal(options.cache, 'no-store')
  assert.ok(options.signal instanceof AbortSignal)
  assert.equal(result.status, 400)
  assert.match(result.reason, /Several copies share this image ID/)
  assert.doesNotMatch(JSON.stringify(result), /private-token/)
})

test('known missing and offline source responses explain what failed', async () => {
  const missing = await diagnoseImageLoad(source, async () => jsonResponse(404, 'File not found on disk'))
  assert.match(missing.reason, /original image file is missing/)
  const offline = await diagnoseImageLoad(source, async () => jsonResponse(503, 'Drive is offline'))
  assert.match(offline.reason, /drive.*offline/)
})

test('unexpected server detail is not displayed or logged as a private file path', async () => {
  const result = await diagnoseImageLoad(source, async () => jsonResponse(500, 'Failed to serve /Users/person/private/image.png'))
  assert.equal(result.status, 500)
  assert.match(result.reason, /server failed/)
  assert.doesNotMatch(JSON.stringify(result), /person|private|media_token/)
})

test('malformed error response keeps the HTTP status and a useful fallback', async () => {
  const result = await diagnoseImageLoad(source, async () => new Response('not json', {
    status: 404,
    headers: { 'content-type': 'application/json' },
  }))
  assert.equal(result.status, 404)
  assert.match(result.reason, /could not find/)
})

test('successful repeat request does not consume the original image and does not claim the initial cause', async () => {
  let cancelled = false
  const result = await diagnoseImageLoad(source, async () => ({
    ok: true,
    status: 200,
    headers: new Headers({ 'content-type': 'image/jpeg', 'content-length': '12345' }),
    body: { cancel: async () => { cancelled = true } },
  }))
  assert.equal(cancelled, true)
  assert.equal(result.status, 200)
  assert.equal(result.mediaType, 'image/jpeg')
  assert.equal(result.contentLength, '12345')
  assert.match(result.reason, /original failure may have been transient/)
})

test('non-image repeat response and network failure remain distinct', async () => {
  const wrongContent = await diagnoseImageLoad(source, async () => new Response('<html></html>', {
    status: 200,
    headers: { 'content-type': 'text/html' },
  }))
  assert.match(wrongContent.reason, /non-image content/)
  const network = await diagnoseImageLoad(source, async () => { throw new TypeError('Failed to fetch private-token') })
  assert.equal(network.status, null)
  assert.match(network.reason, /diagnostic request failed/)
  assert.doesNotMatch(JSON.stringify(network), /private-token/)
})
