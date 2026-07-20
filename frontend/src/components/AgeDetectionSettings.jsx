import { useEffect, useState } from 'react'
import { detectAgesRetrospective, getAgeDetectionStatus } from '../api'
import { toast } from './Toast'
import './OpticalFlowSettings.css'

export default function AgeDetectionSettings() {
  const [status, setStatus] = useState(null)
  const [running, setRunning] = useState(false)

  const loadStatus = async () => {
    try {
      setStatus(await getAgeDetectionStatus())
    } catch (error) {
      console.error('Failed to load age detector status:', error)
    }
  }

  useEffect(() => {
    loadStatus()
  }, [])

  const runRetrospective = async () => {
    setRunning(true)
    try {
      const result = await detectAgesRetrospective()
      toast.success(result.message || `Queued ${result.queued} images for age detection`)
    } catch (error) {
      toast.error(`Failed to queue age detection: ${error.message}`)
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="optical-flow-settings">
      <h2>Age Detector</h2>
      <p className="settings-description">
        Detect faces and estimate ages in images. Enable automatic processing for each watched directory from the Directories page.
      </p>
      <div className="backend-status">
        <strong>Status:</strong>
        <span className={`backend-badge ${status?.installed ? 'available' : 'unavailable'}`}>
          {status?.installed ? 'Installed' : 'Not installed'}
        </span>
      </div>
      <button onClick={runRetrospective} disabled={running || !status?.installed}>
        {running ? 'Queuing...' : 'Run on existing images'}
      </button>
    </section>
  )
}
