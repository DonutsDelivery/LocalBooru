export async function scanQrCode({ idPrefix = 'qr-scanner' } = {}) {
  if (window.AndroidQrScanner?.scan) {
    return scanQrCodeWithAndroidBridge()
  }

  return scanQrCodeInWeb(idPrefix)
}

const androidScanRequests = new Map()

if (typeof window !== 'undefined') {
  window.__localBooruQrScannerResolve = (requestId, content, error) => {
    const request = androidScanRequests.get(requestId)
    if (!request) return
    androidScanRequests.delete(requestId)
    if (error) request.reject(new Error(error))
    else if (!content) request.reject(new Error('The QR scanner returned no content.'))
    else request.resolve(content)
  }
}

function scanQrCodeWithAndroidBridge() {
  const requestId = globalThis.crypto?.randomUUID?.()
    || `localbooru-qr-${Date.now()}-${Math.random().toString(16).slice(2)}`

  return new Promise((resolve, reject) => {
    androidScanRequests.set(requestId, { resolve, reject })
    try {
      window.AndroidQrScanner.scan(requestId)
    } catch (error) {
      androidScanRequests.delete(requestId)
      reject(error)
    }
  })
}

async function scanQrCodeInWeb(idPrefix) {
  const { Html5Qrcode } = await import('html5-qrcode')
  const scannerId = `${idPrefix}-${Date.now()}`
  const container = document.createElement('div')
  container.id = scannerId
  container.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:10000;background:#000;'
  document.body.appendChild(container)

  const closeButton = document.createElement('button')
  closeButton.textContent = 'Cancel'
  closeButton.style.cssText = 'position:fixed;bottom:40px;left:50%;transform:translateX(-50%);z-index:10001;padding:12px 32px;font-size:16px;background:#333;color:#fff;border:none;border-radius:8px;cursor:pointer;'
  document.body.appendChild(closeButton)

  const scanner = new Html5Qrcode(scannerId)
  let settled = false
  const cleanup = async () => {
    try { await scanner.stop() } catch { /* scanner may still be starting */ }
    container.remove()
    closeButton.remove()
  }

  try {
    return await new Promise((resolve, reject) => {
      closeButton.onclick = () => {
        if (settled) return
        settled = true
        reject(new Error('QR scanning was cancelled.'))
      }
      scanner.start(
        { facingMode: 'environment' },
        { fps: 10, qrbox: { width: 250, height: 250 } },
        decodedText => {
          if (settled) return
          settled = true
          resolve(decodedText)
        },
        () => {},
      ).catch(reject)
    })
  } finally {
    await cleanup()
  }
}
