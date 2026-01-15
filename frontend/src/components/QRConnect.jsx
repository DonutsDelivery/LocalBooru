import { useState, useEffect } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { isMobileApp } from '../serverManager'
import { getQRData } from '../api'
import './QRConnect.css'

export default function QRConnect() {
  const [qrData, setQrData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchQRData()
  }, [])

  async function fetchQRData() {
    try {
      setLoading(true)
      setError(null)
      const data = await getQRData()
      setQrData(data)
    } catch (err) {
      setError('Could not fetch server info')
      console.error('QR data fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  // Don't show on mobile app
  if (isMobileApp()) {
    return null
  }

  if (loading) {
    return (
      <div className="qr-connect">
        <div className="qr-loading">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="qr-connect">
        <div className="qr-error">{error}</div>
        <button className="retry-btn" onClick={fetchQRData}>Retry</button>
      </div>
    )
  }

  if (!qrData?.local) {
    return (
      <div className="qr-connect">
        <div className="qr-notice">
          <p>Enable local network access in Network settings to generate a QR code.</p>
        </div>
      </div>
    )
  }

  const qrString = JSON.stringify(qrData)

  return (
    <div className="qr-connect">
      <h3>Connect Mobile App</h3>

      <div className="qr-container">
        <QRCodeSVG
          value={qrString}
          size={200}
          level="M"
          bgColor="#ffffff"
          fgColor="#000000"
        />
      </div>

      <div className="qr-instructions">
        <p>Scan this QR code with the LocalBooru mobile app to connect.</p>
        <ol>
          <li>Open the LocalBooru app on your phone</li>
          <li>Go to Settings &gt; Servers</li>
          <li>Tap "Scan QR Code"</li>
        </ol>
      </div>

      <div className="qr-details">
        <div className="detail-row">
          <span className="detail-label">Local:</span>
          <span className="detail-value">{qrData.local}</span>
        </div>
        {qrData.public && (
          <div className="detail-row">
            <span className="detail-label">Public:</span>
            <span className="detail-value">{qrData.public}</span>
          </div>
        )}
        {qrData.auth && (
          <div className="detail-row auth-notice">
            Authentication required - enter credentials in the app
          </div>
        )}
      </div>
    </div>
  )
}
