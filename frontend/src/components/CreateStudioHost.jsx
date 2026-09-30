import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDirectories, fetchLibraries, getAddon } from '../api'
import {
  cancelStudioJobs,
  createDirectoryOptions,
  createErrorMessage,
  createStudioSession,
  getCreateStatus,
  getStudioSession,
  importStudioOutput,
  outputUrl,
  studioUrl,
} from '../services/donutCreate'
import CreateSettings from './CreateSettings'
import './CreateStudio.css'

export default function CreateStudioHost() {
  const [opened, setOpened] = useState(false)
  const [started, setStarted] = useState(false)
  const [showSetup, setShowSetup] = useState(false)
  const [status, setStatus] = useState(null)
  const [session, setSession] = useState(null)
  const [frameUrl, setFrameUrl] = useState('')
  const [directories, setDirectories] = useState([])
  const [destinationKey, setDestinationKey] = useState('')
  const [opening, setOpening] = useState(false)
  const [saving, setSaving] = useState(null)
  const [cancelling, setCancelling] = useState(false)
  const [saved, setSaved] = useState({})
  const [error, setError] = useState('')
  const [announcement, setAnnouncement] = useState('')
  const [switchingServer, setSwitchingServer] = useState(false)
  const [serverRevision, setServerRevision] = useState(0)
  const sessionRef = useRef(null)
  const generation = useRef(0)
  const requests = useRef(new AbortController())
  const serverChanging = useRef(false)
  const initialLaunch = useRef(false)
  const openingLock = useRef(false)
  const saveLock = useRef(false)
  const mounted = useRef(false)
  const visible = useRef(false)
  const trigger = useRef(null)
  const closeButton = useRef(null)
  const host = useRef(null)

  useEffect(() => {
    mounted.current = true
    requests.current = new AbortController()
    const openStudio = () => {
      trigger.current = document.activeElement
      visible.current = true
      setStarted(true)
      setOpened(true)
    }
    window.addEventListener('donut-create-open', openStudio)
    return () => {
      mounted.current = false
      generation.current += 1
      requests.current.abort()
      window.removeEventListener('donut-create-open', openStudio)
    }
  }, [])

  function closeStudio() {
    visible.current = false
    setOpened(false)
  }

  useEffect(() => {
    if (!opened) return
    const siblings = Array.from(host.current.parentElement.children)
      .filter(element => element !== host.current)
      .map(element => ({ element, inert: element.inert }))
    siblings.forEach(({ element }) => { element.inert = true })
    closeButton.current?.focus()
    const escape = event => {
      if (event.key === 'Escape') {
        visible.current = false
        setOpened(false)
      }
    }
    window.addEventListener('keydown', escape)
    return () => {
      window.removeEventListener('keydown', escape)
      siblings.forEach(({ element, inert }) => { element.inert = inert })
      trigger.current?.focus?.()
    }
  }, [opened])

  const invalidateSession = useCallback(message => {
    generation.current += 1
    requests.current.abort()
    requests.current = new AbortController()
    initialLaunch.current = true
    openingLock.current = false
    saveLock.current = false
    sessionRef.current = null
    setStatus(null)
    setSession(null)
    setFrameUrl('')
    setSaved({})
    setOpening(false)
    setSaving(null)
    setCancelling(false)
    setShowSetup(true)
    setError(message)
  }, [])

  const launchStudio = useCallback(async () => {
    if (openingLock.current || serverChanging.current) return
    openingLock.current = true
    initialLaunch.current = true
    const requestGeneration = generation.current
    setOpening(true)
    setError('')
    try {
      const nextSession = await createStudioSession(requests.current.signal)
      if (!mounted.current || requestGeneration !== generation.current) return
      sessionRef.current = nextSession
      setSession(nextSession)
      setFrameUrl(studioUrl(nextSession.id))
      setShowSetup(false)
      setAnnouncement('Studio connected. Use the workflow controls to create an image.')
    } catch (openError) {
      if (mounted.current && requestGeneration === generation.current) {
        setError(createErrorMessage(openError))
        setShowSetup(true)
      }
    } finally {
      if (requestGeneration === generation.current) {
        openingLock.current = false
        if (mounted.current) setOpening(false)
      }
    }
  }, [])

  const handleStatus = useCallback(nextStatus => {
    if (serverChanging.current) return
    setStatus(nextStatus)
    const current = sessionRef.current
    if (current?.backend_url && nextStatus?.backend_url && current.backend_url !== nextStatus.backend_url) {
      invalidateSession('The backend changed. Open a new studio session to continue.')
    }
  }, [invalidateSession])

  useEffect(() => {
    const changed = () => invalidateSession('The backend changed. Open a new studio session to continue.')
    const changingServer = () => {
      const hadStudio = !!sessionRef.current || openingLock.current || initialLaunch.current
      serverChanging.current = true
      setSwitchingServer(true)
      invalidateSession('The selected server changed. Open a new studio session to continue.')
      if (!hadStudio) initialLaunch.current = false
      setDirectories([])
      setDestinationKey('')
    }
    const changedServer = () => {
      serverChanging.current = false
      setSwitchingServer(false)
      setServerRevision(previous => previous + 1)
    }
    window.addEventListener('donut-create-backend-changed', changed)
    window.addEventListener('donut-create-server-changing', changingServer)
    window.addEventListener('donut-create-server-changed', changedServer)
    return () => {
      window.removeEventListener('donut-create-backend-changed', changed)
      window.removeEventListener('donut-create-server-changing', changingServer)
      window.removeEventListener('donut-create-server-changed', changedServer)
    }
  }, [invalidateSession])

  useEffect(() => {
    if (!started) return
    let active = true
    let polling = false
    const poll = async () => {
      if (polling || serverChanging.current) return
      polling = true
      const requestGeneration = generation.current
      const signal = requests.current.signal
      try {
        const response = await getAddon('donut-create', { signal })
        const addon = response.addon
        if (!active || requestGeneration !== generation.current) return
        if (!addon.installed || addon.status !== 'running') {
          handleStatus(null)
          setShowSetup(true)
          return
        }
        const nextStatus = await getCreateStatus(signal)
        if (!active || requestGeneration !== generation.current) return
        handleStatus(nextStatus)
        if (requestGeneration !== generation.current) return
        if (!sessionRef.current && !initialLaunch.current && visible.current) {
          if (nextStatus.backend?.ready) await launchStudio()
          else setShowSetup(true)
        }
        const current = sessionRef.current
        if (current) {
          const nextSession = await getStudioSession(current.id, signal)
          if (active && requestGeneration === generation.current && sessionRef.current?.id === current.id) {
            sessionRef.current = nextSession
            setSession(nextSession)
          }
        }
      } catch (pollError) {
        if (!active || requestGeneration !== generation.current || signal.aborted) return
        if ([404, 410].includes(pollError.response?.status) && sessionRef.current) {
          invalidateSession('The studio session expired or became unavailable. Open the studio again to reconnect.')
        } else {
          setError(createErrorMessage(pollError))
        }
      } finally {
        polling = false
      }
    }
    poll()
    const timer = setInterval(poll, 2000)
    return () => { active = false; clearInterval(timer) }
  }, [started, serverRevision, handleStatus, invalidateSession, launchStudio])

  useEffect(() => {
    if (!opened || switchingServer) return
    let active = true
    const requestGeneration = generation.current
    Promise.all([fetchLibraries(), fetchDirectories(true, null, 'image')])
      .then(([libraryData, directoryData]) => {
        if (active && requestGeneration === generation.current) setDirectories(createDirectoryOptions(directoryData.directories || [], libraryData.libraries || []))
      })
      .catch(directoryError => { if (active && requestGeneration === generation.current) setError(`Could not load save directories: ${createErrorMessage(directoryError)}`) })
    return () => { active = false }
  }, [opened, switchingServer, serverRevision])

  async function saveOutput(output) {
    const destination = directories.find(directory => directory.key === destinationKey)
    const current = sessionRef.current
    if (!destination || !current || saveLock.current || serverChanging.current) return
    saveLock.current = true
    setSaving(output.id)
    setError('')
    const requestGeneration = generation.current
    try {
      const result = await importStudioOutput(current.id, output.id, destination, requests.current.signal)
      if (mounted.current && requestGeneration === generation.current) {
        setSaved(previous => ({ ...previous, [`${output.id}:${destination.key}`]: result }))
        setAnnouncement(`Image saved to ${destination.label}.`)
      }
    } catch (saveError) {
      if (mounted.current && requestGeneration === generation.current) setError(createErrorMessage(saveError))
    } finally {
      if (requestGeneration === generation.current) {
        saveLock.current = false
        if (mounted.current) setSaving(null)
      }
    }
  }

  async function cancelJobs() {
    const current = sessionRef.current
    if (!current || cancelling || serverChanging.current) return
    setCancelling(true)
    setError('')
    const requestGeneration = generation.current
    try {
      const result = await cancelStudioJobs(current.id, requests.current.signal)
      if (mounted.current && requestGeneration === generation.current) {
        if (result.session?.id === current.id) {
          sessionRef.current = result.session
          setSession(result.session)
        }
        if (result.errors?.length) {
          const message = result.errors.join(' ')
          setError(message)
          setAnnouncement(message)
        } else {
          const cancelledCount = result.cancelled?.length || 0
          setAnnouncement(cancelledCount > 0
            ? `Cancelled ${cancelledCount} studio job${cancelledCount === 1 ? '' : 's'}.`
            : 'No studio jobs were cancelled.')
        }
      }
    } catch (cancelError) {
      if (mounted.current && requestGeneration === generation.current) setError(createErrorMessage(cancelError))
    } finally {
      if (mounted.current && requestGeneration === generation.current) setCancelling(false)
    }
  }

  if (!started) return null
  const jobs = session?.jobs || []
  const outputs = (session?.outputs || []).filter(output => !output.media_type
    || output.media_type === 'image' || output.media_type.startsWith('image/'))
  const hasActiveJobs = jobs.some(job => job.status === 'queued' || job.status === 'running')
  const destination = directories.find(directory => directory.key === destinationKey)

  return (
    <div className="create-studio-host" hidden={!opened} ref={host}>
      <section className="create-studio-dialog" role="dialog" aria-modal="true" aria-labelledby="create-studio-title">
        <header className="create-studio-header">
          <div><h1 id="create-studio-title">Create images</h1><span>DonutUI · workflow v5</span></div>
          <div className="create-actions">
            {session && <button type="button" aria-pressed={!showSetup} onClick={() => setShowSetup(false)}>Studio</button>}
            <button type="button" aria-pressed={showSetup} onClick={() => setShowSetup(true)}>Setup</button>
            <button type="button" ref={closeButton} onClick={closeStudio}>Close studio</button>
          </div>
        </header>
        <p className="create-sr-status" role="status" aria-live="polite">{announcement}</p>
        {error && <p className="create-message error" role="alert">{error}</p>}
        {switchingServer && <p className="create-message" role="status">Connecting to the selected server…</p>}
        <div className="create-studio-body">
          <div className="create-studio-main">
            {/* Keep the actual editor mounted through close and Setup so its
                current graph, controls and websocket connection survive. */}
            {session && <iframe title="DonutUI creation studio" src={frameUrl} className="create-studio-frame" hidden={showSetup} referrerPolicy="no-referrer" />}
            <div className="create-studio-setup" hidden={!showSetup && !!session}>
              {showSetup && !switchingServer && <CreateSettings key={serverRevision} onStatusChange={handleStatus} />}
              <button type="button" className="create-primary create-launch" disabled={switchingServer || opening || (!session && !status?.backend?.ready)} onClick={() => session ? setShowSetup(false) : launchStudio()}>
                {opening ? 'Connecting studio…' : session ? 'Return to studio' : 'Open studio'}
              </button>
              {!showSetup && !session && <p>Connecting to the creator…</p>}
            </div>
          </div>
          {session && (
            <aside className="create-results" aria-label="Creation results">
              <h2>Studio jobs</h2>
              {jobs.length === 0 ? <p>Generate from the studio controls. Results will appear here.</p> : (
                <ul className="create-job-list">{jobs.map(job => (
                  <li key={job.id}><span>{job.id.slice(0, 8)}</span><strong>{job.status}</strong>{job.error && <p role="alert">{job.error}</p>}</li>
                ))}</ul>
              )}
              {hasActiveJobs && <button type="button" disabled={cancelling} onClick={cancelJobs}>{cancelling ? 'Cancelling jobs…' : 'Cancel studio jobs'}</button>}
              <h2>Save results</h2>
              <label className="create-field">Save to image directory
                <select value={destination ? destinationKey : ''} onChange={event => setDestinationKey(event.target.value)}>
                  <option value="">Choose a directory</option>
                  {directories.map(directory => <option key={directory.key} value={directory.key}>{directory.label}</option>)}
                </select>
              </label>
              {directories.length === 0 && <p>Add or enable an Images directory in Directories to save results.</p>}
              <div className="create-output-list">{outputs.map(output => {
                const imported = saved[`${output.id}:${destinationKey}`]
                return <article className="create-output" key={output.id}>
                  <img src={outputUrl(session.id, output.id)} alt={output.filename || 'Generated image'} loading="lazy" />
                  <span className="create-output-name">{output.filename || 'Generated image'}</span>
                  <button type="button" className="create-primary" disabled={!destination || saving !== null || !!imported} onClick={() => saveOutput(output)}>
                    {saving === output.id ? 'Saving…' : imported ? 'Saved to library' : 'Save to library'}
                  </button>
                </article>
              })}</div>
            </aside>
          )}
        </div>
      </section>
    </div>
  )
}
