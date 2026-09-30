import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDirectories, fetchLibraries, getAddon } from '../api'
import {
  cancelStudioJobs,
  createDirectoryOptions,
  createLibraryOptions,
  createOutputDirectory,
  createErrorMessage,
  createStudioSession,
  getCreateStatus,
  getStudioSession,
  importStudioOutput,
  outputUrl,
  studioUrl,
} from '../services/donutCreate'
import CreateSettings from './CreateSettings'
import CreateEditCanvas from './CreateEditCanvas'
import SimpleCreateControls from './SimpleCreateControls'
import { createStudioBridge } from '../services/createStudioBridge'
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
  const [libraries, setLibraries] = useState([])
  const [outputLibraryId, setOutputLibraryId] = useState('')
  const [creatingDirectory, setCreatingDirectory] = useState(false)
  const [workflowPending, setWorkflowPending] = useState(false)
  const [opening, setOpening] = useState(false)
  const [saving, setSaving] = useState(null)
  const [cancelling, setCancelling] = useState(false)
  const [saved, setSaved] = useState({})
  const [error, setError] = useState('')
  const [announcement, setAnnouncement] = useState('')
  const [switchingServer, setSwitchingServer] = useState(false)
  const [serverRevision, setServerRevision] = useState(0)
  const [studioView, setStudioView] = useState('simple')
  const [controlTab, setControlTab] = useState('create')
  const [snapshot, setSnapshot] = useState(null)
  const [bridgeConnecting, setBridgeConnecting] = useState(true)
  const [bridgeBusy, setBridgeBusy] = useState('')
  const [bridgeError, setBridgeError] = useState('')
  const [controlsRevision, setControlsRevision] = useState(0)
  const [editRevision, setEditRevision] = useState(0)
  const [mobilePane, setMobilePane] = useState('controls')
  const sessionRef = useRef(null)
  const generation = useRef(0)
  const requests = useRef(new AbortController())
  const serverChanging = useRef(false)
  const initialLaunch = useRef(false)
  const openingLock = useRef(false)
  const saveLock = useRef(false)
  const directoryCreateLock = useRef(false)
  const directoryRequestVersion = useRef(0)
  const pendingWorkflow = useRef(null)
  const mounted = useRef(false)
  const visible = useRef(false)
  const trigger = useRef(null)
  const closeButton = useRef(null)
  const host = useRef(null)
  const studioFrame = useRef(null)
  const bridge = useRef(null)
  const snapshotRef = useRef(null)
  const snapshotRequest = useRef(null)
  const studioActions = useRef(Promise.resolve())
  const pendingActions = useRef(0)
  const actionErrorVisible = useRef(false)
  const studioSessionId = session?.id

  useEffect(() => {
    mounted.current = true
    requests.current = new AbortController()
    const openStudio = () => {
      trigger.current = document.activeElement
      visible.current = true
      setStarted(true)
      setOpened(true)
    }
    const loadWorkflow = event => {
      if (serverChanging.current) return
      const workflow = event.detail?.workflow
      if (!workflow || typeof workflow !== 'object' || Array.isArray(workflow)) return
      pendingWorkflow.current = { workflow, generation: generation.current, loading: false }
      setWorkflowPending(true)
      setError('')
      setBridgeError('')
      if (!sessionRef.current && !openingLock.current) initialLaunch.current = false
      setAnnouncement('Opening the saved workflow. It will load when the studio is ready.')
      openStudio()
    }
    window.addEventListener('donut-create-open', openStudio)
    window.addEventListener('donut-create-load-workflow', loadWorkflow)
    return () => {
      mounted.current = false
      generation.current += 1
      requests.current.abort()
      bridge.current?.dispose()
      window.removeEventListener('donut-create-open', openStudio)
      window.removeEventListener('donut-create-load-workflow', loadWorkflow)
    }
  }, [])

  function closeStudio() {
    visible.current = false
    pendingWorkflow.current = null
    setWorkflowPending(false)
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
        pendingWorkflow.current = null
        setWorkflowPending(false)
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
    bridge.current?.dispose()
    bridge.current = null
    snapshotRef.current = null
    snapshotRequest.current = null
    studioActions.current = Promise.resolve()
    pendingActions.current = 0
    actionErrorVisible.current = false
    initialLaunch.current = true
    openingLock.current = false
    saveLock.current = false
    directoryCreateLock.current = false
    directoryRequestVersion.current += 1
    pendingWorkflow.current = null
    sessionRef.current = null
    setStatus(null)
    setSession(null)
    setFrameUrl('')
    setSaved({})
    setOpening(false)
    setSaving(null)
    setCancelling(false)
    setCreatingDirectory(false)
    setWorkflowPending(false)
    setSnapshot(null)
    setBridgeBusy('')
    setBridgeConnecting(true)
    setBridgeError('')
    setStudioView('simple')
    setControlTab('create')
    setMobilePane('controls')
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
      setSnapshot(null)
      setBridgeConnecting(true)
      setBridgeError('')
      setStudioView('simple')
      setControlTab('create')
      setMobilePane('controls')
      setShowSetup(false)
      setAnnouncement('Studio connected. Describe an image to begin, or choose Edit to upload a reference.')
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
      setLibraries([])
      setOutputLibraryId('')
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
          setStatus(previous => previous ? { ...previous, backend: { ...previous.backend, ready: false } } : null)
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

  const refreshDestinations = useCallback(async select => {
    const requestGeneration = generation.current
    const version = ++directoryRequestVersion.current
    const signal = requests.current.signal
    const [libraryData, directoryData] = await Promise.all([fetchLibraries({ signal }), fetchDirectories(true, null, 'image', { signal })])
    if (!mounted.current || signal.aborted || requestGeneration !== generation.current || version !== directoryRequestVersion.current) return
    const libraryOptions = createLibraryOptions(libraryData.libraries || [])
    const directoryOptions = createDirectoryOptions(directoryData.directories || [], libraryData.libraries || [])
    setLibraries(libraryOptions)
    setDirectories(directoryOptions)
    setOutputLibraryId(previous => libraryOptions.some(library => library.id === previous) ? previous : libraryOptions.find(library => library.primary)?.id || libraryOptions[0]?.id || '')
    if (select) {
      const key = `${select.library_id}:${select.directory_id}`
      if (!directoryOptions.some(directory => directory.key === key)) throw new Error('The output directory was created, but is not available for saving. Refresh the studio destinations to try again.')
      setDestinationKey(key)
    }
  }, [])

  useEffect(() => {
    if (!opened || switchingServer || directoryCreateLock.current) return
    const requestGeneration = generation.current
    const version = directoryRequestVersion.current + 1
    refreshDestinations().catch(directoryError => {
      if (mounted.current && requestGeneration === generation.current && version === directoryRequestVersion.current) setError(`Could not load save directories: ${createErrorMessage(directoryError)}`)
    })
  }, [opened, switchingServer, serverRevision, studioSessionId, refreshDestinations])

  const acceptSnapshot = useCallback(nextSnapshot => {
    snapshotRef.current = nextSnapshot
    setSnapshot(nextSnapshot)
    setBridgeConnecting(!nextSnapshot.ready && !nextSnapshot.error)
    if (!actionErrorVisible.current) setBridgeError(nextSnapshot.error || '')
    if (nextSnapshot.ready && nextSnapshot.fields?.editing?.available !== false) {
      setControlTab(previous => previous === 'tuning' ? previous : nextSnapshot.fields?.editing?.value === true ? 'edit' : 'create')
    }
  }, [])

  const syncSnapshot = useCallback(() => {
    const currentBridge = bridge.current
    const currentSession = sessionRef.current
    const requestGeneration = generation.current
    if (!currentBridge || !currentSession || serverChanging.current) return Promise.resolve(null)
    if (snapshotRequest.current?.bridge === currentBridge) return snapshotRequest.current.promise
    const promise = currentBridge.request('snapshot', undefined, { timeout: 7000 })
      .then(nextSnapshot => {
        if (!mounted.current || generation.current !== requestGeneration || bridge.current !== currentBridge) return null
        acceptSnapshot(nextSnapshot)
        return nextSnapshot
      })
      .catch(syncError => {
        if (mounted.current && generation.current === requestGeneration && bridge.current === currentBridge && syncError.name !== 'AbortError') {
          setBridgeConnecting(false)
          if (!actionErrorVisible.current) setBridgeError(syncError.message)
        }
        return null
      })
      .finally(() => {
        if (snapshotRequest.current?.bridge === currentBridge) snapshotRequest.current = null
      })
    snapshotRequest.current = { bridge: currentBridge, promise }
    return promise
  }, [acceptSnapshot])

  useEffect(() => {
    if (!studioSessionId || !frameUrl || !studioFrame.current) return
    const currentBridge = createStudioBridge({
      iframe: studioFrame.current, sessionId: studioSessionId, url: frameUrl, signal: requests.current.signal,
    })
    bridge.current = currentBridge
    const handshake = () => {
      if ((!snapshotRef.current?.ready || visible.current) && pendingActions.current === 0) syncSnapshot()
    }
    handshake()
    const timer = setInterval(handshake, 2000)
    return () => {
      clearInterval(timer)
      currentBridge.dispose()
      if (bridge.current === currentBridge) bridge.current = null
    }
  }, [studioSessionId, frameUrl, syncSnapshot])

  const studioAction = useCallback((action, payload, ownsRequest) => {
    if (action !== 'load-workflow' && pendingWorkflow.current) {
      pendingWorkflow.current = null
      setWorkflowPending(false)
      setAnnouncement('Saved workflow loading was cancelled to keep your current edits.')
    }
    const currentBridge = bridge.current
    const currentSession = sessionRef.current
    const requestGeneration = generation.current
    if (!currentBridge || !currentSession || serverChanging.current) return Promise.reject(new Error('Connect the studio before continuing.'))
    pendingActions.current += 1
    const request = studioActions.current.catch(() => {}).then(async () => {
      try {
        if (generation.current !== requestGeneration || bridge.current !== currentBridge || serverChanging.current || (ownsRequest && !ownsRequest())) {
          throw new DOMException('The studio session changed. Reconnect to continue.', 'AbortError')
        }
        setBridgeBusy(action)
        actionErrorVisible.current = false
        setBridgeError('')
        const nextSnapshot = await currentBridge.request(action, payload, { timeout: action === 'upload-reference' ? 120000 : ['generate', 'load-workflow'].includes(action) ? 60000 : 15000 })
        if (!mounted.current || generation.current !== requestGeneration || bridge.current !== currentBridge) throw new DOMException('The studio session changed.', 'AbortError')
        acceptSnapshot(nextSnapshot)
        if (action === 'load-workflow') {
          setControlsRevision(previous => previous + 1)
          setEditRevision(previous => previous + 1)
        }
        if (action === 'generate') setAnnouncement('Image added to your studio queue.')
        return nextSnapshot
      } catch (actionError) {
        if (mounted.current && generation.current === requestGeneration && actionError.name !== 'AbortError') {
          actionErrorVisible.current = true
          setBridgeError(actionError.message)
        }
        throw actionError
      } finally {
        if (generation.current === requestGeneration && bridge.current === currentBridge) {
          pendingActions.current -= 1
          if (mounted.current) setBridgeBusy('')
        }
      }
    })
    studioActions.current = request
    return request
  }, [acceptSnapshot])

  const loadPendingWorkflow = useCallback(async () => {
    const request = pendingWorkflow.current
    if (!request || request.loading || !visible.current || request.generation !== generation.current || serverChanging.current) return
    request.loading = true
    try {
      const nextSnapshot = await studioAction('load-workflow', { workflow: request.workflow }, () => visible.current && pendingWorkflow.current === request)
      if (!mounted.current || request.generation !== generation.current || pendingWorkflow.current !== request) return
      setControlTab(nextSnapshot.fields?.editing?.value === true ? 'edit' : 'create')
      setAnnouncement('Saved workflow loaded. Review it before generating an image.')
    } catch { /* The bridge reports errors while keeping the previous graph. */ }
    finally {
      if (pendingWorkflow.current === request) {
        pendingWorkflow.current = null
        if (mounted.current) setWorkflowPending(false)
      }
    }
  }, [studioAction])

  useEffect(() => {
    if (opened && workflowPending && snapshot?.ready && status?.backend?.ready && !bridgeConnecting && !bridgeBusy && !switchingServer) loadPendingWorkflow()
  }, [opened, workflowPending, snapshot?.ready, status?.backend?.ready, bridgeConnecting, bridgeBusy, switchingServer, loadPendingWorkflow])

  const patchStudio = useCallback(patch => studioAction('patch', patch), [studioAction])
  const generateImage = useCallback(async () => {
    const nextSnapshot = await studioAction('generate')
    setMobilePane('results')
    return nextSnapshot
  }, [studioAction])
  const uploadReference = useCallback(async file => {
    const nextSnapshot = await studioAction('upload-reference', { file })
    setControlTab('edit')
    setMobilePane('results')
    return nextSnapshot
  }, [studioAction])

  function refreshControls() {
    const requestGeneration = generation.current
    setBridgeConnecting(true)
    actionErrorVisible.current = false
    setBridgeError('')
    return studioActions.current.catch(() => {}).then(() => requestGeneration === generation.current ? syncSnapshot() : null).then(nextSnapshot => {
      if (nextSnapshot && mounted.current && requestGeneration === generation.current) {
        setControlsRevision(previous => previous + 1)
        setControlTab(previous => previous === 'tuning' ? previous : nextSnapshot.fields?.editing?.value === true ? 'edit' : 'create')
      }
    })
  }

  function showSimpleStudio() {
    setShowSetup(false)
    setStudioView('simple')
    if (studioView === 'advanced') refreshControls()
  }

  function showAdvancedStudio() {
    if (pendingWorkflow.current && !pendingWorkflow.current.loading) {
      pendingWorkflow.current = null
      setWorkflowPending(false)
      setAnnouncement('Saved workflow loading was cancelled to keep your Advanced editor draft.')
    }
    setShowSetup(false)
    setStudioView('advanced')
  }

  async function loadLatestRun() {
    try {
      const nextSnapshot = await studioAction('load-latest')
      setControlsRevision(previous => previous + 1)
      setEditRevision(previous => previous + 1)
      setControlTab(nextSnapshot.fields?.editing?.value === true ? 'edit' : 'create')
      setAnnouncement('Loaded the latest run into your working draft.')
    } catch { /* The host reports the error without replacing the current draft. */ }
  }

  function frameLoaded() {
    snapshotRef.current = null
    setSnapshot(null)
    setBridgeConnecting(true)
    syncSnapshot()
  }

  async function changeMask(patch) {
    const requestGeneration = generation.current
    const payload = { ...patch }
    if (payload.editMask && typeof payload.editMask === 'object') payload.editMask = JSON.stringify(payload.editMask)
    if (payload.editMask === undefined) delete payload.editMask
    try {
      await patchStudio(payload)
    } catch {
      if (mounted.current && requestGeneration === generation.current) {
        await syncSnapshot()
        if (mounted.current && requestGeneration === generation.current) setEditRevision(previous => previous + 1)
      }
    }
  }

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

  async function createSaveDirectory() {
    const destination = directories.find(directory => directory.key === destinationKey)
    const libraryId = destination?.libraryId || outputLibraryId
    if (!libraryId || directoryCreateLock.current || serverChanging.current) return
    directoryCreateLock.current = true
    const requestGeneration = generation.current
    setCreatingDirectory(true)
    setError('')
    try {
      const result = await createOutputDirectory(libraryId, requests.current.signal)
      if (!mounted.current || requestGeneration !== generation.current) return
      await refreshDestinations(result)
      if (mounted.current && requestGeneration === generation.current) setAnnouncement(`${result.name || 'Created Images'} is ready and selected for saving.`)
    } catch (directoryError) {
      if (mounted.current && requestGeneration === generation.current) setError(createErrorMessage(directoryError))
    } finally {
      if (requestGeneration === generation.current) {
        directoryCreateLock.current = false
        if (mounted.current) setCreatingDirectory(false)
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
  const activeJobs = jobs.filter(job => job.status === 'queued' || job.status === 'running')
  const destination = directories.find(directory => directory.key === destinationKey)
  const connected = !!status?.backend?.ready && !!snapshot?.ready && !bridgeConnecting && !switchingServer
  const reference = snapshot?.fields?.referenceA?.value || ''
  const referenceId = typeof reference === 'string' && /^donutref:[a-f0-9]{64}$/.test(reference) ? reference.slice('donutref:'.length) : null
  const referenceUrl = referenceId && frameUrl ? new URL(`donut/edit-studio/reference/${referenceId}`, new URL(frameUrl, window.location.href)).href : ''
  const maskData = snapshot?.fields?.editMask?.value || null
  const canMask = snapshot?.fields?.editing?.value === true && snapshot?.fields?.inpaint?.available !== false && !!snapshot?.fields?.inpaint
  const simpleView = !showSetup && studioView === 'simple'

  return (
    <div className="create-studio-host" hidden={!opened} ref={host}>
      <section className="create-studio-dialog" role="dialog" aria-modal="true" aria-labelledby="create-studio-title">
        <header className="create-studio-header">
          <div className="create-studio-brand"><div className="create-brand-mark" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z" /></svg>
          </div><div><h1 id="create-studio-title">Create images</h1><span>Donut workflow v5 <span aria-hidden="true">·</span> Your creative workspace</span></div></div>
          <div className="create-actions">
            {session && <div className="create-view-switch">
              <button type="button" aria-pressed={simpleView} onClick={showSimpleStudio}>Simple studio</button>
              <button type="button" aria-pressed={!showSetup && studioView === 'advanced'} onClick={showAdvancedStudio}>Advanced editor</button>
            </div>}
            <button type="button" aria-pressed={showSetup} onClick={() => setShowSetup(true)}>Setup</button>
            <button type="button" ref={closeButton} onClick={closeStudio}>Close studio</button>
          </div>
        </header>
        <p className="create-sr-status" role="status" aria-live="polite">{announcement}</p>
        {error && <p className="create-message error" role="alert">{error}</p>}
        {bridgeError && snapshot?.ready && <p className="create-message error" role="alert">{bridgeError}</p>}
        {switchingServer && <p className="create-message" role="status">Connecting to the selected server…</p>}
        {workflowPending && <p className="create-message" role="status">{bridgeBusy === 'load-workflow' ? 'Loading the saved workflow…' : 'The saved workflow is waiting for the studio to be ready.'}</p>}
        {session && simpleView && <div className="create-mobile-workspace-nav" role="group" aria-label="Workspace panel">
          <button type="button" aria-pressed={mobilePane === 'controls'} onClick={() => setMobilePane('controls')}>Controls</button>
          <button type="button" aria-pressed={mobilePane === 'results'} onClick={() => setMobilePane('results')}>{controlTab === 'edit' ? 'Canvas & results' : `Results${outputs.length ? ` (${outputs.length})` : ''}`}</button>
        </div>}
        <div className={`create-studio-body create-view-${showSetup ? 'setup' : studioView} create-mobile-${mobilePane}`}>
          {session && <div className="create-controls-slot" hidden={!simpleView}>
            <SimpleCreateControls key={`${session.id}:${controlsRevision}`} snapshot={snapshot} activeTab={controlTab} onTabChange={setControlTab}
              connected={connected} connecting={bridgeConnecting} busy={workflowPending ? bridgeBusy || 'load-workflow' : bridgeBusy} error={bridgeError}
              onPatch={patchStudio} onGenerate={generateImage} onUpload={uploadReference} onRetry={refreshControls} onAdvanced={showAdvancedStudio} />
          </div>}
          <div className="create-studio-main" hidden={!!session && simpleView}>
            {/* One live workflow backs both views. Hiding this frame preserves
                advanced edits and keeps its bridge and websocket connected. */}
            {session && <iframe ref={studioFrame} title="DonutUI creation studio" src={frameUrl} className="create-studio-frame" hidden={showSetup || studioView !== 'advanced'} referrerPolicy="no-referrer" onLoad={frameLoaded} />}
            <div className="create-studio-setup" hidden={!showSetup && !!session}>
              {showSetup && !switchingServer && <CreateSettings key={serverRevision} onStatusChange={handleStatus} />}
              <button type="button" className="create-primary create-launch" disabled={switchingServer || opening || (!session && !status?.backend?.ready)} onClick={() => session ? showSimpleStudio() : launchStudio()}>
                {opening ? 'Connecting studio…' : session ? 'Return to studio' : 'Open studio'}
              </button>
              {!showSetup && !session && <p>Connecting to the creator…</p>}
            </div>
          </div>
          {session && (
            <section className="create-results" aria-label="Creation results">
              <div className="create-results-toolbar">
                <div><h2>{simpleView && controlTab === 'edit' ? 'Editing workspace' : 'Results'} <span className="create-count">{outputs.length}</span></h2>
                  <p>{outputs.length ? 'Save your favorites to the library.' : 'Images from this workspace will appear here.'}</p></div>
                <div className="create-save-tools">
                  <label className="create-field create-destination">Save to image directory
                    <select value={destination ? destinationKey : ''} disabled={creatingDirectory} onChange={event => setDestinationKey(event.target.value)}>
                      <option value="">Choose a directory</option>
                      {directories.map(directory => <option key={directory.key} value={directory.key}>{directory.label}</option>)}
                    </select>
                  </label>
                  {!destination && libraries.length > 0 && <label className="create-field create-output-library">Output library
                    <select value={outputLibraryId} disabled={creatingDirectory} onChange={event => setOutputLibraryId(event.target.value)}>
                      {libraries.map(library => <option key={library.id} value={library.id}>{library.label}</option>)}
                    </select>
                  </label>}
                  <button type="button" className="create-directory-button" title="Create or reuse an image-only Created Images folder in this library"
                    disabled={creatingDirectory || switchingServer || (!destination && !outputLibraryId)} onClick={createSaveDirectory}>
                    {creatingDirectory ? 'Creating output directory…' : 'Create output directory'}
                  </button>
                </div>
              </div>
              <div className="create-results-scroll">
                {snapshot?.latestAvailable && <div className="create-latest-note"><div><strong>A newer run is available</strong><span>Load it to replace this working draft.</span></div>
                  <button type="button" disabled={!connected || !!bridgeBusy} onClick={loadLatestRun}>{bridgeBusy === 'load-latest' ? 'Loading latest run…' : 'Load latest run'}</button>
                </div>}
                {directories.length === 0 && <p className="create-control-note">Add or enable an Images directory in Directories to save results.</p>}
                {(jobs.length > 0 || hasActiveJobs) && <div className="create-queue-bar">
                  <details className="create-queue-details">
                    <summary><span className={hasActiveJobs ? 'create-spinner' : 'create-connection-dot'} aria-hidden="true" />
                      <span>Studio jobs</span><strong>{hasActiveJobs ? `${activeJobs.length} active` : `${jobs.length} recent`}</strong></summary>
                    <ul className="create-job-list">{jobs.map(job => (
                      <li key={job.id}><span>{job.id.slice(0, 8)}</span><strong>{job.status}</strong>{job.error && <p role="alert">{job.error}</p>}</li>
                    ))}</ul>
                  </details>
                  {hasActiveJobs && <button type="button" disabled={cancelling} onClick={cancelJobs}>{cancelling ? 'Cancelling jobs…' : 'Cancel studio jobs'}</button>}
                </div>}
                {simpleView && controlTab === 'edit' && <CreateEditCanvas key={`${session.id}:${editRevision}`} reference={reference} imageUrl={referenceUrl} maskData={maskData}
                  enabled={canMask} selectedArea={snapshot?.fields?.inpaint?.value === true} disabled={!connected || !!bridgeBusy || workflowPending}
                  onChange={changeMask} onUpload={file => uploadReference(file).catch(() => {})} />}
                {simpleView && controlTab === 'edit' && outputs.length > 0 && <h3 className="create-results-heading">Your results</h3>}
                {outputs.length === 0 && (controlTab !== 'edit' || !simpleView) && <div className="create-results-empty">
                  <div className="create-empty-art" aria-hidden="true"><svg width="62" height="62" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="1.3"><rect x="8" y="10" width="44" height="44" rx="7" /><path d="m9 44 12-12 11 10 8-8 12 12" /><circle cx="39" cy="24" r="4" /><path d="m55 5 1.5 4.5L61 11l-4.5 1.5L55 17l-1.5-4.5L49 11l4.5-1.5L55 5Z" /></svg></div>
                  <h3>{hasActiveJobs ? 'Your idea is taking shape' : 'Your next favorite image starts here'}</h3>
                  <p>{hasActiveJobs ? 'You can keep creating while your images are in the queue.' : 'Write a prompt and choose Generate. Your images will appear here, ready to save.'}</p>
                  {!hasActiveJobs && <span className="create-empty-tip">Try a subject, a setting, and a little atmosphere.</span>}
                </div>}
                <div className="create-output-list">{outputs.slice().reverse().map(output => {
                const imported = saved[`${output.id}:${destinationKey}`]
                return <article className="create-output" key={output.id}>
                  <div className="create-output-preview"><img src={outputUrl(session.id, output.id)} alt={output.filename || 'Generated image'} loading="lazy" />
                    {imported && <span className="create-saved-badge">Saved</span>}</div>
                  <div className="create-output-footer"><span className="create-output-name">{output.filename || 'Generated image'}</span>
                    <button type="button" className="create-primary" disabled={!destination || saving !== null || creatingDirectory || !!imported} onClick={() => saveOutput(output)}>
                      {saving === output.id ? 'Saving…' : imported ? 'Saved to library' : 'Save to library'}
                    </button>
                  </div>
                </article>
                })}</div>
              </div>
            </section>
          )}
        </div>
      </section>
    </div>
  )
}
