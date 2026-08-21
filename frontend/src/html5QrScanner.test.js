import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

test('both mobile QR entry points use the foreground native Android scanner', async () => {
  const [selectSource, settingsSource, scannerSource] = await Promise.all([
    readFile(new URL('./components/ServerSelectScreen.jsx', import.meta.url), 'utf8'),
    readFile(new URL('./components/ServerSettings.jsx', import.meta.url), 'utf8'),
    readFile(new URL('./qrScanner.js', import.meta.url), 'utf8')
  ])

  for (const source of [selectSource, settingsSource]) {
    assert.match(source, /scanQrCode/)
    assert.doesNotMatch(source, /new Html5Qrcode|getUserMedia|BarcodeDetector|setInterval/)
  }
  assert.match(scannerSource, /AndroidQrScanner\?\.scan/)
  assert.match(scannerSource, /window\.AndroidQrScanner\.scan\(requestId\)/)
  assert.match(scannerSource, /__localBooruQrScannerResolve/)
  assert.doesNotMatch(scannerSource, /plugin-barcode-scanner|windowed:/)
})

test('the Android package contains the proven foreground GMS Code Scanner', async () => {
  const [gradleSource, kotlinSource, manifestSource, packageSource] = await Promise.all([
    readFile(new URL('../../src-tauri/gen/android/app/build.gradle.kts', import.meta.url), 'utf8'),
    readFile(new URL('../../src-tauri/gen/android/app/src/main/java/com/localbooru/app/MainActivity.kt', import.meta.url), 'utf8'),
    readFile(new URL('../../src-tauri/gen/android/app/src/main/AndroidManifest.xml', import.meta.url), 'utf8'),
    readFile(new URL('../package.json', import.meta.url), 'utf8')
  ])

  assert.match(gradleSource, /play-services-code-scanner:16\.1\.0/)
  assert.match(kotlinSource, /GmsBarcodeScanning\.getClient/)
  assert.match(kotlinSource, /GmsBarcodeScannerOptions\.Builder/)
  assert.match(kotlinSource, /setBarcodeFormats\(Barcode\.FORMAT_QR_CODE\)/)
  assert.match(kotlinSource, /enableAutoZoom\(\)/)
  assert.match(kotlinSource, /\.startScan\(\)/)
  assert.match(manifestSource, /com\.google\.mlkit\.vision\.DEPENDENCIES/)
  assert.match(manifestSource, /android:value="barcode_ui"/)
  assert.equal(JSON.parse(packageSource).dependencies['@tauri-apps/plugin-barcode-scanner'], undefined)
})

test('the Android bridge returns decoded text and propagates native cancellation', async () => {
  const calls = []
  globalThis.window = {
    AndroidQrScanner: {
      scan(requestId) {
        calls.push(requestId)
        queueMicrotask(() => window.__localBooruQrScannerResolve(requestId, 'decoded-qr', null))
      }
    }
  }

  try {
    const { scanQrCode } = await import(`./qrScanner.js?bridge-test=${Date.now()}`)
    assert.equal(await scanQrCode(), 'decoded-qr')
    assert.equal(calls.length, 1)

    window.AndroidQrScanner.scan = requestId => {
      queueMicrotask(() => window.__localBooruQrScannerResolve(
        requestId,
        null,
        'QR scanning was cancelled.'
      ))
    }
    await assert.rejects(scanQrCode(), /QR scanning was cancelled/)
  } finally {
    delete globalThis.window
  }
})
