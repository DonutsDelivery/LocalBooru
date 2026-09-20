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

async function scanQrCodeInWeb() {
  let stream = null
  let scanInterval = null
  let overlay = null

  try {
    // Keep the camera in an explicit, top-level foreground overlay.  The
    // html5-qrcode widget can create a video surface behind the Tauri Android
    // WebView, while this direct stream path is the one that works reliably on
    // the phone.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    })

    overlay = document.createElement('div')
    overlay.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:#000;display:flex;flex-direction:column;'

    const video = document.createElement('video')
    video.setAttribute('autoplay', '')
    video.setAttribute('playsinline', '')
    video.setAttribute('muted', '')
    video.muted = true
    video.style.cssText = 'flex:1;width:100%;height:100%;object-fit:cover;background:#000;'
    video.srcObject = stream
    overlay.appendChild(video)

    const indicator = document.createElement('div')
    indicator.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:250px;height:250px;border:3px solid rgba(99,102,241,0.8);border-radius:16px;pointer-events:none;'
    overlay.appendChild(indicator)

    const bottomBar = document.createElement('div')
    bottomBar.style.cssText = 'padding:16px;display:flex;justify-content:center;background:rgba(0,0,0,0.7);'
    const closeButton = document.createElement('button')
    closeButton.textContent = 'Cancel'
    closeButton.style.cssText = 'padding:12px 32px;font-size:16px;background:#333;color:#fff;border:none;border-radius:8px;cursor:pointer;'
    bottomBar.appendChild(closeButton)
    overlay.appendChild(bottomBar)
    document.body.appendChild(overlay)

    await video.play()

    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d', { willReadFrequently: true })
    let decoder = null
    let scanFile = null
    if ('BarcodeDetector' in window) {
      decoder = new BarcodeDetector({ formats: ['qr_code'] })
    } else {
      const { Html5Qrcode } = await import('html5-qrcode')
      scanFile = (file) => Html5Qrcode.scanFile(file, false)
    }

    let settled = false
    const cleanup = () => {
      if (settled) return
      settled = true
      clearInterval(scanInterval)
      stream?.getTracks().forEach(track => track.stop())
      video.srcObject = null
      overlay?.remove()
    }

    return await new Promise((resolve, reject) => {
      const finish = (error, value) => {
        if (settled) return
        cleanup()
        if (error) reject(error)
        else resolve(value)
      }

      closeButton.onclick = () => finish(new Error('QR scanning was cancelled.'))

      scanInterval = setInterval(async () => {
        if (settled || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return
        try {
          if (decoder) {
            const barcodes = await decoder.detect(video)
            if (barcodes.length > 0) finish(null, barcodes[0].rawValue)
            return
          }

          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          context.drawImage(video, 0, 0, canvas.width, canvas.height)
          const blob = await new Promise(resolveBlob => canvas.toBlob(resolveBlob, 'image/png'))
          if (!blob || settled) return
          const file = new File([blob], 'frame.png', { type: 'image/png' })
          try {
            finish(null, await scanFile(file))
          } catch {
            // No QR code in this frame; keep scanning.
          }
        } catch {
          // Ignore transient camera/decode errors and keep the foreground view alive.
        }
      }, 200)
    })
  } catch (error) {
    if (scanInterval) clearInterval(scanInterval)
    stream?.getTracks().forEach(track => track.stop())
    overlay?.remove()
    throw error
  }
}
