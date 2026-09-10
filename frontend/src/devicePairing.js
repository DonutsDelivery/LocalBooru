import { addOrUpdateServer, isTauriApp, probeServer } from './serverManager.js'

const encoder = new TextEncoder()
const decoder = new TextDecoder()
export async function getLocalPairingApiBase() {
  if (isTauriApp()) {
    const { invoke } = await import('@tauri-apps/api/core')
    return localPairingApiBaseForPort(await invoke('backend_get_port'))
  }
  return '/api/device-pairing'
}

export function localPairingApiBaseForPort(port) {
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error('The local pairing service reported an invalid port')
  }
  return `http://127.0.0.1:${port}/api/device-pairing`
}

function bytesToBase64(bytes) {
  let binary = ''
  for (const byte of new Uint8Array(bytes)) binary += String.fromCharCode(byte)
  return btoa(binary)
}

function base64ToBytes(value) {
  const binary = atob(value)
  return Uint8Array.from(binary, char => char.charCodeAt(0))
}

function base64UrlToBytes(value) {
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/')
  return base64ToBytes(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '='))
}

function bytesToBase64Url(bytes) {
  return bytesToBase64(bytes).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

export function formatPairingFingerprint(value) {
  return (value || '').slice(0, 24).match(/.{1,4}/g)?.join(' ') || ''
}

export async function confirmDeviceAuthorization() {
  const bridge = window.AndroidDeviceApproval
  if (!bridge?.request) return
  const requestId = crypto.randomUUID()
  await new Promise((resolve, reject) => {
    const pending = window.__localBooruDeviceApprovalPending ||= {}
    const timeout = window.setTimeout(() => {
      delete pending[requestId]
      reject(new Error('Device confirmation timed out'))
    }, 60_000)
    pending[requestId] = { resolve, reject, timeout }
    window.__localBooruDeviceApprovalResolve ||= (id, approved, error) => {
      const entry = window.__localBooruDeviceApprovalPending?.[id]
      if (!entry) return
      window.clearTimeout(entry.timeout)
      delete window.__localBooruDeviceApprovalPending[id]
      if (approved) entry.resolve()
      else entry.reject(new Error(error || 'Device confirmation was cancelled'))
    }
    bridge.request(requestId)
  })
}

async function directFetch(url, options = {}) {
  if (isTauriApp() && !isLoopbackPairingUrl(url)) {
    const { fetch: tauriFetch } = await import('@tauri-apps/plugin-http')
    return tauriFetch(url, options)
  }
  return fetch(url, options)
}

export function isLoopbackPairingUrl(value) {
  try {
    const url = new URL(value, globalThis.location?.href || 'http://localhost')
    return url.hostname === '127.0.0.1' || url.hostname === 'localhost' || url.hostname === '::1'
  } catch {
    return false
  }
}

export function selectDesktopCallbackUrls(callbackUrls = []) {
  const normalized = callbackUrls.map(value => value.replace(/\/+$/, ''))
  const reachableFromPhone = normalized.filter(value => !isLoopbackPairingUrl(value))
  return reachableFromPhone.length > 0 ? reachableFromPhone : normalized
}

async function jsonRequest(url, options = {}) {
  const response = await directFetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    signal: options.signal || AbortSignal.timeout(10_000),
  })
  let body = null
  try { body = await response.json() } catch { /* handled below */ }
  if (!response.ok) {
    throw new Error(body?.detail || body?.error || `Server returned ${response.status}`)
  }
  return body
}

export function isEligiblePairingServerUrl(value) {
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) return false
    if (url.pathname && url.pathname !== '/') return false
    return url.protocol === 'https:' || isPrivateNetworkHost(url.hostname)
  } catch {
    return false
  }
}

function normalizePairingServerUrl(value) {
  if (!isEligiblePairingServerUrl(value)) {
    throw new Error('Desktop authorization requires HTTPS or an authenticated private-LAN server')
  }
  return new URL(value).origin
}

export function pairingServerEndpoints(server, workingUrl) {
  const baseUrl = normalizePairingServerUrl(workingUrl)
  const alternatives = [server.url, server.fallbackUrl]
    .filter(Boolean)
    .flatMap(value => {
      try { return [normalizePairingServerUrl(value)] } catch { return [] }
    })
    .filter(url => url !== baseUrl)
  return { baseUrl, fallbackUrl: alternatives[0] || null }
}

export async function createDesktopPairingSession(displayName) {
  const [signingKeys, encryptionKeys] = await Promise.all([
    crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify']),
    crypto.subtle.generateKey({ name: 'RSA-OAEP', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' }, true, ['encrypt', 'decrypt']),
  ])
  const [signingSpki, encryptionSpki] = await Promise.all([
    crypto.subtle.exportKey('spki', signingKeys.publicKey),
    crypto.subtle.exportKey('spki', encryptionKeys.publicKey),
  ])
  const signingPublicKeySpki = bytesToBase64(signingSpki)
  const encryptionPublicKeySpki = bytesToBase64(encryptionSpki)
  const fingerprintInput = new Uint8Array(signingPublicKeySpki.length + 1 + encryptionPublicKeySpki.length)
  fingerprintInput.set(encoder.encode(signingPublicKeySpki), 0)
  fingerprintInput[signingPublicKeySpki.length] = 0
  fingerprintInput.set(encoder.encode(encryptionPublicKeySpki), signingPublicKeySpki.length + 1)
  const publicKeyFingerprint = bytesToBase64Url(await crypto.subtle.digest('SHA-256', fingerprintInput))
  const apiBase = await getLocalPairingApiBase()
  const session = await jsonRequest(`${apiBase}/desktop-sessions`, {
    method: 'POST',
    body: JSON.stringify({ displayName, signingPublicKeySpki, encryptionPublicKeySpki, publicKeyFingerprint }),
  })
  return { session, signingPrivateKey: signingKeys.privateKey, encryptionPrivateKey: encryptionKeys.privateKey }
}

export async function pollDesktopPairingSession(sessionId, sessionSecret) {
  const apiBase = await getLocalPairingApiBase()
  return jsonRequest(`${apiBase}/desktop-sessions/${encodeURIComponent(sessionId)}`, {
    headers: { 'X-LocalBooru-Pairing-Secret': sessionSecret },
  })
}

export async function cancelDesktopPairingSession(sessionId, sessionSecret) {
  try {
    const apiBase = await getLocalPairingApiBase()
    await jsonRequest(`${apiBase}/desktop-sessions/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
      headers: { 'X-LocalBooru-Pairing-Secret': sessionSecret },
    })
  } catch { /* expiry and cancellation are both terminal */ }
}

async function encryptForDesktop(publicKeySpki, value) {
  const publicKey = await crypto.subtle.importKey(
    'spki',
    base64ToBytes(publicKeySpki),
    { name: 'RSA-OAEP', hash: 'SHA-256' },
    false,
    ['encrypt'],
  )
  const contentKey = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt'])
  const rawContentKey = await crypto.subtle.exportKey('raw', contentKey)
  const iv = crypto.getRandomValues(new Uint8Array(12))
  const [wrappedKey, ciphertext] = await Promise.all([
    crypto.subtle.encrypt({ name: 'RSA-OAEP' }, publicKey, rawContentKey),
    crypto.subtle.encrypt({ name: 'AES-GCM', iv }, contentKey, encoder.encode(JSON.stringify(value))),
  ])
  return JSON.stringify({
    version: 1,
    wrappedKey: bytesToBase64(wrappedKey),
    iv: bytesToBase64(iv),
    ciphertext: bytesToBase64(ciphertext),
  })
}

export async function authorizeServerForDesktop(server, pairingRequest) {
  if (!server.token) throw new Error('This server does not have a token-based authenticated session')
  if (!isTauriApp()) throw new Error('Desktop authorization requires LocalBooru protected native networking')
  const probe = await probeServer(server)
  if (!probe.success || !probe.url) {
    throw new Error(`The selected server is not reachable: ${probe.error || 'connection failed'}`)
  }
  const { baseUrl, fallbackUrl: secureFallbackUrl } = pairingServerEndpoints(server, probe.url)
  const callbackUrl = await verifyDesktopCallback(pairingRequest)
  const grant = await jsonRequest(`${baseUrl}/api/device-pairing/grants`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${server.token}` },
    body: JSON.stringify({
      desktopSessionId: pairingRequest.sessionId,
      displayName: pairingRequest.displayName,
      signingPublicKeySpki: pairingRequest.signingPublicKeySpki,
      encryptionPublicKeySpki: pairingRequest.encryptionPublicKeySpki,
      publicKeyFingerprint: pairingRequest.publicKeyFingerprint,
      requestedScope: pairingRequest.requestedScope,
    }),
  })
  const encryptedPayload = await encryptForDesktop(pairingRequest.encryptionPublicKeySpki, {
    serverId: grant.serverId,
    serverName: grant.serverName,
    url: baseUrl,
    fallbackUrl: secureFallbackUrl,
    grantId: grant.grantId,
    grantSecret: grant.grantSecret,
    challenge: grant.challenge,
    expiresAt: grant.expiresAt,
  })
  await jsonRequest(`${callbackUrl}/api/device-pairing/desktop-sessions/${encodeURIComponent(pairingRequest.sessionId)}/deliver`, {
    method: 'POST',
    body: JSON.stringify({
      sessionSecret: pairingRequest.sessionSecret,
      serverId: grant.serverId,
      encryptedPayload,
    }),
  })
  return { serverId: grant.serverId, serverName: grant.serverName }
}

async function verifyDesktopCallback(pairingRequest) {
  const callbackErrors = []
  const callbackPublicKey = await crypto.subtle.importKey(
    'spki',
    base64ToBytes(pairingRequest.callbackPublicKeySpki),
    { name: 'ECDSA', namedCurve: 'P-256' },
    false,
    ['verify'],
  )
  for (const callbackUrl of selectDesktopCallbackUrls(pairingRequest.callbackUrls)) {
    try {
      const callbackChallenge = crypto.randomUUID()
      const proof = await jsonRequest(`${callbackUrl}/api/device-pairing/desktop-sessions/${encodeURIComponent(pairingRequest.sessionId)}/verify`, {
        method: 'POST',
        body: JSON.stringify({
          sessionSecret: pairingRequest.sessionSecret,
          publicKeyFingerprint: pairingRequest.publicKeyFingerprint,
          callbackChallenge,
        }),
      })
      if (proof.sessionId !== pairingRequest.sessionId
          || proof.publicKeyFingerprint !== pairingRequest.publicKeyFingerprint
          || proof.callbackChallenge !== callbackChallenge
          || Number(proof.expiresAt) <= Math.floor(Date.now() / 1000)) {
        throw new Error('Desktop callback identity did not match the QR request')
      }
      const callbackMessage = encoder.encode(`localbooru-callback-v1\0${proof.sessionId}\0${proof.publicKeyFingerprint}\0${callbackChallenge}\0${proof.expiresAt}`)
      const verified = await crypto.subtle.verify(
        { name: 'ECDSA', hash: 'SHA-256' },
        callbackPublicKey,
        base64UrlToBytes(proof.callbackSignature),
        callbackMessage,
      )
      if (!verified) throw new Error('Desktop callback proof signature was invalid')
      return callbackUrl
    } catch (error) {
      callbackErrors.push(`${callbackUrl}: ${error.message}`)
      console.warn('[DesktopPairing] Callback verification failed:', callbackUrl, error.message)
    }
  }
  throw new Error(`Desktop callback could not prove this pairing session. Tried: ${callbackErrors.join('; ') || 'no reachable desktop callback was advertised'}`)
}

async function decryptDesktopDelivery(privateKey, encryptedPayload) {
  const envelope = JSON.parse(encryptedPayload)
  if (envelope.version !== 1) throw new Error('Unsupported encrypted pairing payload')
  const rawContentKey = await crypto.subtle.decrypt({ name: 'RSA-OAEP' }, privateKey, base64ToBytes(envelope.wrappedKey))
  const contentKey = await crypto.subtle.importKey('raw', rawContentKey, { name: 'AES-GCM' }, false, ['decrypt'])
  const plaintext = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: base64ToBytes(envelope.iv) },
    contentKey,
    base64ToBytes(envelope.ciphertext),
  )
  return JSON.parse(decoder.decode(plaintext))
}

export async function inspectDesktopDelivery(delivery, privateKeys) {
  const payload = await decryptDesktopDelivery(privateKeys.encryptionPrivateKey, delivery.encryptedPayload)
  if (payload.serverId !== delivery.serverId || payload.expiresAt < Math.floor(Date.now() / 1000)) {
    throw new Error('Pairing delivery identity or expiry check failed')
  }
  payload.url = normalizePairingServerUrl(payload.url)
  if (payload.fallbackUrl) payload.fallbackUrl = normalizePairingServerUrl(payload.fallbackUrl)
  return payload
}

export async function redeemDesktopPayload(payload, privateKeys) {
  if (payload.expiresAt < Math.floor(Date.now() / 1000)) throw new Error('Pairing grant has expired')
  if (!isTauriApp()) throw new Error('Paired credentials require protected native storage')
  const signature = await crypto.subtle.sign(
    { name: 'ECDSA', hash: 'SHA-256' },
    privateKeys.signingPrivateKey,
    encoder.encode(payload.challenge),
  )
  const exchange = await jsonRequest(`${payload.url}/api/device-pairing/exchange`, {
    method: 'POST',
    body: JSON.stringify({
      grantId: payload.grantId,
      grantSecret: payload.grantSecret,
      signature: bytesToBase64Url(signature),
    }),
  })
  if (exchange.serverId !== payload.serverId) {
    throw new Error('Pairing server identity changed during credential exchange')
  }
  return addOrUpdateServer({
    id: exchange.serverId,
    name: exchange.serverName || payload.serverName,
    url: payload.url,
    fallbackUrl: payload.fallbackUrl,
    token: exchange.token,
    deviceId: exchange.deviceId,
    username: null,
    password: null,
    lastConnected: new Date().toISOString(),
  })
}

export async function redeemDesktopDelivery(delivery, privateKeys) {
  return redeemDesktopPayload(await inspectDesktopDelivery(delivery, privateKeys), privateKeys)
}

export async function validateDesktopPairingRequest(value) {
  if (!value || value.type !== 'localbooru-desktop-pairing' || value.version !== 1) return false
  if (!value.sessionId || !value.sessionSecret || !value.displayName || !value.publicKeyFingerprint) return false
  if (!value.signingPublicKeySpki || !value.encryptionPublicKeySpki || !value.callbackPublicKeySpki) return false
  if (value.requestedScope !== 'local_network_write') return false
  if (value.sessionId.length > 128 || value.sessionSecret.length > 128 || value.displayName.length > 80) return false
  if (value.signingPublicKeySpki.length > 2048 || value.encryptionPublicKeySpki.length > 4096 || value.callbackPublicKeySpki.length > 2048) return false
  if (!Array.isArray(value.callbackUrls) || value.callbackUrls.length === 0 || value.callbackUrls.length > 4) return false
  if (!value.callbackUrls.every(isPrivatePairingCallback)) return false
  const now = Math.floor(Date.now() / 1000)
  if (Number(value.expiresAt) <= now || Number(value.expiresAt) > now + 180) return false
  try {
    await Promise.all([
      crypto.subtle.importKey('spki', base64ToBytes(value.signingPublicKeySpki), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']),
      crypto.subtle.importKey('spki', base64ToBytes(value.encryptionPublicKeySpki), { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['encrypt']),
      crypto.subtle.importKey('spki', base64ToBytes(value.callbackPublicKeySpki), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']),
    ])
    const fingerprintInput = encoder.encode(`${value.signingPublicKeySpki}\0${value.encryptionPublicKeySpki}`)
    const expected = bytesToBase64Url(await crypto.subtle.digest('SHA-256', fingerprintInput))
    return expected === value.publicKeyFingerprint
  } catch {
    return false
  }
}

function isPrivatePairingCallback(value) {
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return false
    if ((url.pathname && url.pathname !== '/') || url.search || url.hash) return false
    return isPrivateNetworkHost(url.hostname)
  } catch {
    return false
  }
}

function isPrivateNetworkHost(value) {
  const host = value.toLowerCase().replace(/^\[|\]$/g, '')
  if (host === 'localhost' || host === '::1') return true
  const octets = host.split('.').map(Number)
  if (octets.length !== 4 || octets.some(octet => !Number.isInteger(octet) || octet < 0 || octet > 255)) return false
  return octets[0] === 10
    || octets[0] === 127
    || (octets[0] === 192 && octets[1] === 168)
    || (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31)
    || (octets[0] === 100 && octets[1] >= 64 && octets[1] <= 127)
}
