import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  readStudioPreview,
  stagePreviewUrl,
  studioEventsUrl,
  studioUrl,
} from '../services/donutCreate'
import CreateSettings from './CreateSettings'
import CreateEditCanvas from './CreateEditCanvas'
import SimpleCreateControls from './SimpleCreateControls'
import { createStudioBridge } from '../services/createStudioBridge'
import './CreateStudio.css'

const CONNECTION_WAIT_MS = 5 * 60 * 1000

export default function CreateStudioHost() {
  const [opened, setOpened] = useState(false)
  const [started, setStarted] = useState(false)
  const [showSetup, setShowSetup] = useState(false)
  const [status, setStatus] = useState(null)
  const [session, setSession] = useState(null)
  const [frameUrl, setFrameUrl] = useState('')
  const [frameRevision, setFrameRevision] = useState(0)
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
  const [autoSave, setAutoSave] = useState(false)
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
  const [selectedOutputId, setSelectedOutputId] = useState('')
  const [runInstant, setRunInstant] = useState(false)
  const [graphRevision, setGraphRevision] = useState(0)
  const [draftPending, setDraftPending] = useState(false)
  const [livePreview, setLivePreview] = useState(null)
  const [liveProgress, setLiveProgress] = useState(null)
  const [previewConnecting, setPreviewConnecting] = useState(false)
  const [canvasScope, setCanvasScope] = useState('edit')
  const [canvasReference, setCanvasReference] = useState('A')
  const [centerView, setCenterView] = useState('preview')
  const [failedPreview, setFailedPreview] = useState('')
  const [previewRetry, setPreviewRetry] = useState(0)
  const sessionRef = useRef(null)
  const generation = useRef(0)
  const requests = useRef(new AbortController())
  const serverChanging = useRef(false)
  const initialLaunch = useRef(false)
  const openingLock = useRef(false)
  const saveLock = useRef(false)
  const saveAttempts = useRef(new Map())
  const saveBaselines = useRef(new Map())
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
  const connectionDeadline = useRef(0)
  const backendWasReady = useRef(null)
  const studioActions = useRef(Promise.resolve())
  const pendingActions = useRef(0)
  const actionErrorVisible = useRef(false)
  const instantEnabled = useRef(false)
  const instantEpoch = useRef(0)
  const authoredRevision = useRef(0)
  const instantVersion = useRef(null)
  const draftsWaiting = useRef(false)
  const draftSources = useRef(new Set())
  const generateLock = useRef(false)
  const acknowledgedPrompts = useRef(new Set())
  const studioSessionId = session?.id

  const stopInstant = useCallback(() => {
    instantEnabled.current = false
    instantEpoch.current += 1
    setRunInstant(false)
  }, [])

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
      stopInstant()
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
  }, [stopInstant])

  function closeStudio() {
    stopInstant()
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
        stopInstant()
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
  }, [opened, stopInstant])

  const invalidateSession = useCallback(message => {
    stopInstant()
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
    authoredRevision.current = 0
    instantVersion.current = null
    draftsWaiting.current = false
    draftSources.current.clear()
    generateLock.current = false
    acknowledgedPrompts.current.clear()
    initialLaunch.current = true
    openingLock.current = false
    saveLock.current = false
    saveAttempts.current.clear()
    saveBaselines.current.clear()
    directoryCreateLock.current = false
    directoryRequestVersion.current += 1
    pendingWorkflow.current = null
    sessionRef.current = null
    setStatus(null)
    setSession(null)
    setFrameUrl('')
    setSaved({})
    setAutoSave(false)
    setSelectedOutputId('')
    setLivePreview(null)
    setLiveProgress(null)
    setCanvasScope('edit')
    setCanvasReference('A')
    setCenterView('preview')
    setOpening(false)
    setSaving(null)
    setCancelling(false)
    setCreatingDirectory(false)
    setWorkflowPending(false)
    setSnapshot(null)
    setGraphRevision(0)
    setDraftPending(false)
    setBridgeBusy('')
    setBridgeConnecting(true)
    setBridgeError('')
    setStudioView('simple')
    setControlTab('create')
    setMobilePane('controls')
    setShowSetup(true)
    setError(message)
  }, [stopInstant])

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
            for (const job of nextSession.jobs || []) acknowledgedPrompts.current.delete(job.id)
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
      saveBaselines.current.set(`${sessionRef.current?.id}:${key}`, new Set((sessionRef.current?.outputs || []).map(output => output.id)))
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
    if (!currentBridge || !currentSession || serverChanging.current || !visible.current) return Promise.resolve(null)
    if (snapshotRequest.current?.bridge === currentBridge) return snapshotRequest.current.promise
    const promise = currentBridge.request('snapshot', undefined, { timeout: 7000 })
      .then(nextSnapshot => {
        if (!mounted.current || generation.current !== requestGeneration || bridge.current !== currentBridge) return null
        acceptSnapshot(nextSnapshot)
        if (!nextSnapshot.ready && !nextSnapshot.error && Date.now() >= connectionDeadline.current) {
          setBridgeConnecting(false)
          setBridgeError('ComfyUI did not finish starting within five minutes. Retry connection to keep waiting.')
        }
        return nextSnapshot
      })
      .catch(syncError => {
        if (mounted.current && generation.current === requestGeneration && bridge.current === currentBridge && syncError.name !== 'AbortError') {
          const waiting = !snapshotRef.current?.ready && Date.now() < connectionDeadline.current
          setBridgeConnecting(waiting)
          if (!actionErrorVisible.current) setBridgeError(waiting ? '' : syncError.message)
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
    if (!opened || !studioSessionId || !frameUrl || !studioFrame.current) return
    connectionDeadline.current = Date.now() + CONNECTION_WAIT_MS
    if (!snapshotRef.current?.ready) {
      setBridgeConnecting(true)
      setBridgeError('')
    }
    const currentBridge = createStudioBridge({
      iframe: studioFrame.current, sessionId: studioSessionId, url: frameUrl, signal: requests.current.signal,
    })
    bridge.current = currentBridge
    const handshake = () => {
      if (!visible.current) return
      if (!snapshotRef.current?.ready && Date.now() >= connectionDeadline.current) {
        setBridgeConnecting(false)
        if (!actionErrorVisible.current) setBridgeError('ComfyUI did not finish starting within five minutes. Retry connection to keep waiting.')
        return
      }
      if (pendingActions.current === 0) syncSnapshot()
    }
    handshake()
    const timer = setInterval(handshake, 2000)
    return () => {
      clearInterval(timer)
      currentBridge.dispose()
      if (bridge.current === currentBridge) {
        bridge.current = null
        pendingActions.current = 0
        studioActions.current = Promise.resolve()
        if (mounted.current) setBridgeBusy('')
      }
    }
  }, [opened, studioSessionId, frameUrl, frameRevision, syncSnapshot])

  useEffect(() => {
    const wasReady = backendWasReady.current
    const ready = status?.backend?.ready
    backendWasReady.current = ready
    // A failed iframe document cannot recover through postMessage alone.
    if (opened && studioSessionId && wasReady === false && ready === true && !snapshotRef.current?.ready && Date.now() < connectionDeadline.current) {
      setFrameRevision(previous => previous + 1)
    }
  }, [opened, studioSessionId, status?.backend?.ready])

  const studioAction = useCallback((action, payload, ownsRequest) => {
    if (['load-workflow', 'load-latest', 'load-preset'].includes(action)) stopInstant()
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
        const nextSnapshot = await currentBridge.request(action, payload, { timeout: ['upload-reference', 'upload-mask'].includes(action) ? 120000 : ['generate', 'load-workflow'].includes(action) ? 60000 : 15000 })
        if (!mounted.current || generation.current !== requestGeneration || bridge.current !== currentBridge) throw new DOMException('The studio session changed.', 'AbortError')
        acceptSnapshot(nextSnapshot)
        if (['patch', 'upload-reference', 'upload-mask'].includes(action) && nextSnapshot.mutationChanged === true) {
          authoredRevision.current += 1
          setGraphRevision(authoredRevision.current)
        }
        if (action === 'load-workflow') {
          setControlsRevision(previous => previous + 1)
          setEditRevision(previous => previous + 1)
        }
        if (action === 'generate') setAnnouncement('Image added to your studio queue.')
        return nextSnapshot
      } catch (actionError) {
        if (mounted.current && generation.current === requestGeneration && actionError.name !== 'AbortError') {
          stopInstant()
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
  }, [acceptSnapshot, stopInstant])

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
  const generateImage = useCallback(async (instantRequest) => {
    if (generateLock.current) throw new Error('An image is already being added to the queue.')
    const requestGeneration = generation.current
    const epoch = instantEpoch.current
    const version = authoredRevision.current
    const isInstant = instantRequest === true
    generateLock.current = true
    instantVersion.current = version
    try {
      const ownsRequest = isInstant ? () => instantEnabled.current && instantEpoch.current === epoch
        && authoredRevision.current === version && !draftsWaiting.current && visible.current
        && !acknowledgedPrompts.current.size && !(sessionRef.current?.jobs || []).some(job => ['queued', 'running'].includes(job.status)) : undefined
      const nextSnapshot = await studioAction('generate', undefined, ownsRequest)
      for (const id of nextSnapshot.lastQueuedPromptIds || []) acknowledgedPrompts.current.add(id)
      if (!isInstant) setMobilePane('results')
      return nextSnapshot
    } catch (generateError) {
      if (requestGeneration === generation.current && generateError.name !== 'AbortError') stopInstant()
      throw generateError
    } finally {
      if (requestGeneration === generation.current) generateLock.current = false
    }
  }, [studioAction, stopInstant])
  const uploadReference = useCallback(async (file, target = 'referenceA') => {
    const nextSnapshot = await studioAction('upload-reference', { file, target })
    if (!target.startsWith('guidance')) setControlTab('edit')
    setCanvasScope(target.startsWith('guidance') ? 'guidance' : 'edit')
    setCanvasReference(target.endsWith('B') ? 'B' : 'A')
    setCenterView('canvas')
    setMobilePane('results')
    return nextSnapshot
  }, [studioAction])

  const uploadSubjectMask = useCallback(file => studioAction('upload-mask', { file }), [studioAction])

  useEffect(() => {
    if (!studioSessionId || !status?.backend?.ready || switchingServer) return
    const requestGeneration = generation.current
    let active = true, socket, retry, failures = 0, ownPreviewUrl = null
    const signal = requests.current.signal
    const valid = () => active && !signal.aborted && generation.current === requestGeneration
      && sessionRef.current?.id === studioSessionId
    const receive = async event => {
      if (!valid()) return
      if (typeof event.data === 'string') {
        let message
        try { message = JSON.parse(event.data) } catch { return }
        if (message.type === 'progress' && Number.isFinite(message.data?.value) && Number.isFinite(message.data?.max) && message.data.max > 0) {
          setLiveProgress({ value: Math.max(0, Math.min(message.data.value, message.data.max)), max: message.data.max })
        } else if (message.type === 'execution_start') {
          if (ownPreviewUrl) URL.revokeObjectURL(ownPreviewUrl)
          ownPreviewUrl = null
          setLivePreview(previous => previous?.sessionId === studioSessionId ? null : previous)
          setLiveProgress(null)
        }
        else if (message.type === 'executing' && message.data?.node === null) setLiveProgress(null)
        return
      }
      const buffer = event.data instanceof Blob ? await event.data.arrayBuffer() : event.data
      if (!valid()) return
      const image = readStudioPreview(buffer)
      if (!image) return
      const url = URL.createObjectURL(image)
      const previous = ownPreviewUrl
      ownPreviewUrl = url
      setLivePreview({ url, sessionId: studioSessionId })
      if (previous) URL.revokeObjectURL(previous)
    }
    const connect = () => {
      if (!valid()) return
      setPreviewConnecting(true)
      try { socket = new WebSocket(studioEventsUrl(studioSessionId)) }
      catch {
        retry = setTimeout(connect, Math.min(10000, 1000 * 2 ** failures++))
        return
      }
      socket.binaryType = 'arraybuffer'
      socket.onopen = () => { if (valid()) { failures = 0; setPreviewConnecting(false) } }
      socket.onmessage = event => { receive(event).catch(() => {}) }
      socket.onclose = () => {
        if (!valid()) return
        setPreviewConnecting(true)
        retry = setTimeout(connect, Math.min(10000, 1000 * 2 ** failures++))
      }
      socket.onerror = () => { if (valid()) setPreviewConnecting(true) }
    }
    const close = () => {
      if (valid()) {
        const url = ownPreviewUrl
        setLivePreview(previous => previous?.sessionId === studioSessionId && previous.url === url ? null : previous)
        setLiveProgress(null)
      }
      active = false
      clearTimeout(retry)
      if (socket) { socket.onclose = null; socket.close() }
      if (ownPreviewUrl) URL.revokeObjectURL(ownPreviewUrl)
      ownPreviewUrl = null
    }
    connect()
    signal.addEventListener('abort', close, { once: true })
    return () => { signal.removeEventListener('abort', close); close() }
  }, [studioSessionId, status?.backend?.ready, switchingServer])

  const reportDrafts = useCallback((pending, source) => {
    if (pending) draftSources.current.add(source)
    else draftSources.current.delete(source)
    draftsWaiting.current = draftSources.current.size > 0
    setDraftPending(draftsWaiting.current)
  }, [])
  const reportControlDrafts = useCallback(pending => reportDrafts(pending, 'controls'), [reportDrafts])
  const reportCanvasDrafts = useCallback(pending => reportDrafts(pending, 'canvas'), [reportDrafts])

  function changeRunInstant(enabled) {
    instantEpoch.current += 1
    instantEnabled.current = enabled
    instantVersion.current = null
    setRunInstant(enabled)
    if (enabled) {
      actionErrorVisible.current = false
      setBridgeError('')
      setAnnouncement('Run Instant is on. The current acknowledged draft will run when the studio queue is empty.')
    }
  }

  useEffect(() => {
    if (!runInstant || !opened || showSetup || studioView !== 'simple' || workflowPending || switchingServer
      || draftPending || bridgeBusy || bridgeConnecting || bridgeError || !snapshot?.ready || !status?.backend?.ready
      || instantVersion.current === graphRevision || acknowledgedPrompts.current.size
      || (session?.jobs || []).some(job => ['queued', 'running'].includes(job.status))) return
    const timer = setTimeout(() => {
      if (pendingActions.current || generateLock.current || serverChanging.current || !instantEnabled.current) return
      generateImage(true).catch(() => {})
    }, 650)
    return () => clearTimeout(timer)
  }, [runInstant, opened, showSetup, studioView, workflowPending, switchingServer, draftPending, bridgeBusy,
    bridgeConnecting, bridgeError, snapshot?.ready, status?.backend?.ready, graphRevision, session?.jobs, generateImage])

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

  function retryConnection() {
    setBridgeConnecting(true)
    actionErrorVisible.current = false
    setBridgeError('')
    connectionDeadline.current = Date.now() + CONNECTION_WAIT_MS
    if (!snapshotRef.current?.ready) setFrameRevision(previous => previous + 1)
    else refreshControls()
  }

  function showSimpleStudio() {
    setShowSetup(false)
    setStudioView('simple')
    if (studioView === 'advanced') refreshControls()
  }

  function showAdvancedStudio() {
    stopInstant()
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

  async function loadUpdatedV5() {
    try {
      const nextSnapshot = await studioAction('load-preset')
      setControlsRevision(previous => previous + 1)
      setEditRevision(previous => previous + 1)
      setControlTab(nextSnapshot.fields?.editing?.value === true ? 'edit' : 'create')
      setCenterView('preview')
      setAnnouncement('Updated v5 recipe loaded. Review its controls before generating.')
    } catch { /* Keep the previous draft and show the bridge error. */ }
  }

  function frameLoaded() {
    stopInstant()
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
    if (!snapshotRef.current?.fields?.outputCanvas || snapshotRef.current.fields.outputCanvas.available === false) delete payload.outputCanvas
    try {
      await patchStudio(payload)
    } catch {
      if (mounted.current && requestGeneration === generation.current) {
        await syncSnapshot()
        if (mounted.current && requestGeneration === generation.current) setEditRevision(previous => previous + 1)
      }
    }
  }

  async function changeCrop(value) {
    const prefix = canvasScope === 'guidance' ? 'guidance' : ''
    const key = prefix ? `${prefix}Crop${canvasReference}` : `crop${canvasReference}`
    const mode = prefix ? 'guidanceGeometryMode' : 'geometryMode'
    const payload = { [key]: value }
    if (snapshotRef.current?.fields?.[mode]?.options?.includes('Independent crops')) payload[mode] = 'Independent crops'
    await changeMask(payload)
  }

  function openReferenceTools(scope, reference) {
    setCanvasScope(scope)
    setCanvasReference(reference)
    setCenterView('canvas')
    setMobilePane('results')
  }

  function changeControlTab(tab) {
    setControlTab(tab)
    if (tab === 'edit') openReferenceTools('edit', 'A')
    else if (tab === 'create') setCenterView('preview')
  }

  const saveOutput = useCallback(async (output, automatic = false) => {
    const destination = directories.find(directory => directory.key === destinationKey)
    const current = sessionRef.current
    if (!destination || !current || saveLock.current || serverChanging.current) return
    const key = `${current.id}:${output.id}:${destination.key}`
    if (automatic && saveAttempts.current.has(key)) return
    saveLock.current = true
    saveAttempts.current.set(key, 'saving')
    setSaving(output.id)
    setError('')
    const requestGeneration = generation.current
    try {
      const result = await importStudioOutput(current.id, output.id, destination, requests.current.signal)
      if (mounted.current && requestGeneration === generation.current) {
        saveAttempts.current.set(key, 'saved')
        setSaved(previous => ({ ...previous, [`${output.id}:${destination.key}`]: result }))
        setAnnouncement(`Image saved to ${destination.label}.`)
      }
    } catch (saveError) {
      if (mounted.current && requestGeneration === generation.current) {
        saveAttempts.current.set(key, 'error')
        setError(createErrorMessage(saveError))
        if (automatic) {
          setAutoSave(false)
          setAnnouncement('Automatic saving paused. Check the error and try saving again.')
        }
      }
    } finally {
      if (requestGeneration === generation.current) {
        saveLock.current = false
        if (mounted.current) setSaving(null)
      }
    }
  }, [directories, destinationKey])

  const outputs = useMemo(() => (session?.outputs || []).filter(output => !output.media_type
    || output.media_type === 'image' || output.media_type.startsWith('image/')), [session?.outputs])

  useEffect(() => {
    if (!autoSave || saving !== null || creatingDirectory || switchingServer || !session || !destinationKey) return
    const context = `${session.id}:${destinationKey}`
    if (!saveBaselines.current.has(context)) {
      saveBaselines.current.set(context, new Set(outputs.map(output => output.id)))
      return
    }
    const baseline = saveBaselines.current.get(context)
    const next = outputs.find(output => output.final === true && !baseline.has(output.id) && !saved[`${output.id}:${destinationKey}`]
      && !saveAttempts.current.has(`${session.id}:${output.id}:${destinationKey}`))
    if (next) saveOutput(next, true)
  }, [autoSave, saving, creatingDirectory, switchingServer, session, destinationKey, outputs, saved, saveOutput])

  function changeAutoSave(enabled) {
    if (enabled) {
      const current = sessionRef.current
      if (!current || !destinationKey) return
      const context = `${current.id}:${destinationKey}`
      const baseline = saveBaselines.current.get(context) || new Set()
      for (const output of current.outputs || []) {
        const key = `${current.id}:${output.id}:${destinationKey}`
        if (saveAttempts.current.get(key) === 'error') saveAttempts.current.delete(key)
        else baseline.add(output.id)
      }
      saveBaselines.current.set(context, baseline)
    }
    setAutoSave(enabled)
  }

  function changeDestination(key) {
    if (key && key !== destinationKey) saveBaselines.current.set(`${sessionRef.current?.id}:${key}`, new Set((sessionRef.current?.outputs || []).map(output => output.id)))
    setDestinationKey(key)
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
    stopInstant()
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
  const activeJobs = jobs.filter(job => job.status === 'queued' || job.status === 'running')
  const hasActiveJobs = activeJobs.length > 0
  const destination = directories.find(directory => directory.key === destinationKey)
  const connected = !!status?.backend?.ready && !!snapshot?.ready && !bridgeConnecting && !switchingServer
  const fields = snapshot?.fields || {}
  const simpleView = !showSetup && studioView === 'simple'
  const referenceUrls = {}
  for (const key of ['referenceA', 'referenceB', 'guidanceReferenceA', 'guidanceReferenceB']) {
    const reference = fields[key]?.value
    if (typeof reference === 'string' && /^donutref:[a-f0-9]{64}$/.test(reference) && frameUrl) {
      referenceUrls[key] = new URL('donut/edit-studio/reference/' + reference.slice('donutref:'.length), new URL(frameUrl, window.location.href)).href
    }
  }
  const referenceKey = canvasScope === 'guidance' ? 'guidanceReference' + canvasReference : 'reference' + canvasReference
  const cropKey = canvasScope === 'guidance' ? 'guidanceCrop' + canvasReference : 'crop' + canvasReference
  const reference = fields[referenceKey]?.value || ''
  const canMask = canvasScope === 'edit' && canvasReference === 'A' && fields.editing?.value === true
    && fields.inpaint?.available !== false && !!fields.inpaint
  const canCrop = !!fields[cropKey] && fields[cropKey].available !== false
  const latestOutput = outputs.slice().reverse().find(output => output.final === true) || outputs.at(-1)
  const selectedOutput = outputs.find(output => output.id === selectedOutputId) || latestOutput
  const stagePreview = (session?.previews || []).slice().reverse().find(preview => activeJobs.some(job => job.id === preview.prompt_id))
  const liveImage = !selectedOutputId && hasActiveJobs && status?.backend?.ready
    ? livePreview?.sessionId === session?.id ? livePreview.url : stagePreview ? stagePreviewUrl(session.id, stagePreview) : '' : ''
  const previewUrl = liveImage || (selectedOutput ? outputUrl(session.id, selectedOutput.id) : '')
  const imported = selectedOutput && saved[selectedOutput.id + ':' + destinationKey]
  const settings = Object.fromEntries(['editing', 'geometryMode', 'outputCanvas', 'resolutionMode', 'aspectRatio', 'megapixels', 'width', 'height', 'pixelGrid', 'outputMultiple', 'referenceB']
    .map(key => [key, fields[key]?.value]))

  const saveTools = <details className="create-save-tools">
    <summary><span>Save destination</span><strong>{destination?.label || 'Choose a directory'}</strong></summary>
    <div className="create-save-popover">
      <label className="create-field create-destination">Save to image directory
        <select value={destination ? destinationKey : ''} disabled={creatingDirectory || switchingServer} onChange={event => changeDestination(event.target.value)}>
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
      {directories.length === 0 && <p className="create-control-note">Create an output directory here, or enable an Images directory in Directories.</p>}
    </div>
  </details>

  const historyContent = <div className="create-history-grid">
    {outputs.length === 0 && <div className="create-history-empty"><strong>Your image history starts here</strong><p>Completed images stay in this workspace until the session expires.</p></div>}
    {outputs.slice().reverse().map(output => {
      const result = saved[output.id + ':' + destinationKey]
      return <article className="create-history-item" key={output.id}>
        <button type="button" className="create-history-preview" aria-label={'Preview ' + (output.filename || 'generated image')}
          aria-pressed={selectedOutput?.id === output.id && !liveImage} onClick={() => { setSelectedOutputId(output.id); setCenterView('preview'); setMobilePane('results') }}>
          <img src={outputUrl(session.id, output.id)} alt={output.filename || 'Generated image'} loading="lazy" />
          {result && <span className="create-saved-badge">Saved</span>}
          {output.final === false && <span className="create-stage-badge">Stage</span>}
        </button>
        <div className="create-history-footer"><span title={output.filename}>{output.filename || 'Generated image'}</span>
          <button type="button" disabled={!destination || saving !== null || creatingDirectory || !!result} onClick={() => saveOutput(output)}>
            {saving === output.id ? 'Saving…' : result ? 'Saved to library' : 'Save to library'}
          </button>
        </div>
      </article>
    })}
  </div>

  const resultsContent = <section className="create-results" aria-label="Creation results">
    <div className="create-results-toolbar">
      <div className="create-center-tabs" role="group" aria-label="Center workspace">
        <button type="button" aria-pressed={centerView === 'preview'} onClick={() => setCenterView('preview')}>Preview <span className="create-count">{outputs.length}</span></button>
        {simpleView && <button type="button" aria-pressed={centerView === 'canvas'} onClick={() => setCenterView('canvas')}>Image tools</button>}
      </div>
      <div className="create-save-toolbar">
        <label className="create-auto-save"><input type="checkbox" role="switch" checked={autoSave} disabled={!destination || creatingDirectory || switchingServer}
          onChange={event => changeAutoSave(event.target.checked)} />Save output</label>
        {saveTools}
      </div>
    </div>
    <div className="create-results-scroll">
      {snapshot?.workflowUpgradeAvailable && <div className="create-latest-note"><div><strong>Updated v5 controls are available</strong><span>Load the updated preset to replace this recipe.</span></div>
        <button type="button" disabled={!connected || !!bridgeBusy || workflowPending} onClick={loadUpdatedV5}>{bridgeBusy === 'load-preset' ? 'Loading updated v5…' : 'Load updated v5'}</button>
      </div>}
      {snapshot?.latestAvailable && <div className="create-latest-note"><div><strong>A newer run is available</strong><span>Load it to replace this working draft.</span></div>
        <button type="button" disabled={!connected || !!bridgeBusy || workflowPending} onClick={loadLatestRun}>{bridgeBusy === 'load-latest' ? 'Loading latest run…' : 'Load latest run'}</button>
      </div>}
      {(jobs.length > 0 || hasActiveJobs) && <div className="create-queue-bar">
        <details className="create-queue-details"><summary><span className={hasActiveJobs ? 'create-spinner' : 'create-connection-dot'} aria-hidden="true" /><span>Studio jobs</span>
          <strong>{hasActiveJobs ? activeJobs.length + ' active' : jobs.length + ' recent'}</strong></summary>
          <ul className="create-job-list">{jobs.map(job => <li key={job.id}><span>{job.id.slice(0, 8)}</span><strong>{job.status}</strong>{job.error && <p role="alert">{job.error}</p>}</li>)}</ul>
        </details>
        {hasActiveJobs && <button type="button" disabled={cancelling} onClick={cancelJobs}>{cancelling ? 'Cancelling jobs…' : 'Cancel studio jobs'}</button>}
      </div>}
      {simpleView && centerView === 'canvas' ? <CreateEditCanvas key={session.id + ':' + editRevision + ':' + canvasScope + ':' + canvasReference}
        reference={reference} imageUrl={referenceUrls[referenceKey] || ''} maskData={canMask ? fields.editMask?.value : null}
        enabled={canMask} selectedArea={canMask && fields.inpaint?.value === true} disabled={!connected || !!bridgeBusy || workflowPending}
        cropData={fields[cropKey]?.value} cropEnabled={canCrop} onCropChange={changeCrop}
        referenceLabel={(canvasScope === 'guidance' ? 'Guidance reference ' : 'Reference ') + canvasReference}
        referenceOptions={['A', ...(fields[canvasScope === 'guidance' ? 'guidanceUseReferenceB' : 'useReferenceB']?.value === true ? ['B'] : [])].map(id => ({ id, label: id, active: id === canvasReference }))}
        onReferenceChange={setCanvasReference} onChange={changeMask} onUpload={file => uploadReference(file, referenceKey).catch(() => {})}
        onDraftChange={reportCanvasDrafts} outputSettings={settings} referenceBUrl={referenceUrls.referenceB || ''} cropBData={fields.cropB?.value}
        outputSize={[Number(fields.width?.value) || 1152, Number(fields.height?.value) || 896]} pixelGrid={Number(fields.pixelGrid?.value || fields.outputMultiple?.value) || 64}
      /> : <div className="create-large-preview">
        <div className="create-preview-heading"><div><h2>{liveImage ? 'Live preview' : selectedOutputId ? 'Selected image' : 'Latest result'}</h2>
          <span>{liveImage ? 'Your image is taking shape' : selectedOutput?.filename || 'Your creative canvas'}</span></div>
          {selectedOutputId && <button type="button" onClick={() => setSelectedOutputId('')}>Follow latest</button>}
        </div>
        {previewUrl ? <div className="create-preview-image-wrap">
          {failedPreview === previewUrl ? <div className="create-preview-error"><strong>The preview could not be loaded.</strong><p>Your workflow and completed results are still available.</p>
            <button type="button" onClick={() => { setFailedPreview(''); setPreviewRetry(previous => previous + 1) }}>Retry preview</button></div>
            : <img key={previewUrl + ':' + previewRetry} className="create-preview-image" src={previewUrl} alt={liveImage ? 'Live generation preview' : selectedOutput?.filename || 'Generated image'} onError={() => setFailedPreview(previewUrl)} />}
          {!liveImage && imported && <span className="create-saved-badge">Saved</span>}
          {hasActiveJobs && !selectedOutputId && <div className="create-preview-run-status"><span className="create-spinner" aria-hidden="true" />
            {liveProgress ? Math.round(liveProgress.value / liveProgress.max * 100) + '% · Generating' : activeJobs.some(job => job.status === 'running') ? 'Generating' : 'Queued'}
            {previewConnecting && <small>Live preview reconnecting…</small>}</div>}
        </div> : <div className="create-results-empty">
          <div className="create-empty-art" aria-hidden="true"><svg width="62" height="62" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="1.3"><rect x="8" y="10" width="44" height="44" rx="7" /><path d="m9 44 12-12 11 10 8-8 12 12" /><circle cx="39" cy="24" r="4" /><path d="m55 5 1.5 4.5L61 11l-4.5 1.5L55 17l-1.5-4.5L49 11l4.5-1.5L55 5Z" /></svg></div>
          <h3>{hasActiveJobs ? 'Your idea is taking shape' : 'Your next favorite image starts here'}</h3>
          <p>{hasActiveJobs ? 'Live previews appear while your image runs. Completed results stay in History.' : 'Describe an image and choose Generate. Your latest result appears here.'}</p>
        </div>}
        {hasActiveJobs && liveProgress && <progress className="create-live-progress" value={liveProgress.value} max={liveProgress.max} aria-label="Generation progress" />}
        {selectedOutput && !liveImage && <div className="create-preview-actions"><span>{selectedOutput.final === false ? 'Workflow stage result' : 'Completed image'}{imported ? ' · Saved to library' : ''}</span>
          <button type="button" className="create-primary" disabled={!destination || saving !== null || creatingDirectory || !!imported} onClick={() => saveOutput(selectedOutput)}>
            {saving === selectedOutput.id ? 'Saving…' : imported ? 'Saved to library' : 'Save to library'}
          </button>
        </div>}
      </div>}
      {!simpleView && historyContent}
    </div>
  </section>

  return (
    <div className="create-studio-host" hidden={!opened} ref={host}>
      <section className="create-studio-dialog" role="dialog" aria-modal="true" aria-labelledby="create-studio-title">
        <header className="create-studio-header">
          <div className="create-studio-brand"><div className="create-brand-mark" aria-hidden="true"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z" /></svg></div>
            <div><h1 id="create-studio-title">Create images</h1><span>Donut workflow v5 <span aria-hidden="true">·</span> Your creative workspace</span></div></div>
          <div className="create-actions">
            {session && <div className="create-view-switch"><button type="button" aria-pressed={simpleView} onClick={showSimpleStudio}>Simple studio</button>
              <button type="button" aria-pressed={!showSetup && studioView === 'advanced'} onClick={showAdvancedStudio}>Advanced editor</button></div>}
            <button type="button" aria-pressed={showSetup} onClick={() => { stopInstant(); setShowSetup(true) }}>Setup</button>
            <button type="button" ref={closeButton} onClick={closeStudio}>Close studio</button>
          </div>
        </header>
        <p className="create-sr-status" role="status" aria-live="polite">{announcement}</p>
        {error && <p className="create-message error" role="alert">{error}</p>}
        {bridgeError && snapshot?.ready && <p className="create-message error" role="alert">{bridgeError}</p>}
        {switchingServer && <p className="create-message" role="status">Connecting to the selected server…</p>}
        {workflowPending && <p className="create-message" role="status">{bridgeBusy === 'load-workflow' ? 'Loading the saved workflow…' : 'The saved workflow is waiting for the studio to be ready.'}</p>}
        {session && simpleView && <div className="create-mobile-workspace-nav" role="group" aria-label="Workspace panel">
          {[['controls', 'Prompt'], ['results', 'Preview'], ['effects', 'Effects'], ['models', 'Models'], ['history', 'History']].map(([pane, label]) =>
            <button type="button" key={pane} aria-pressed={mobilePane === pane} onClick={() => setMobilePane(pane)}>{label}</button>)}
        </div>}
        <div className={'create-studio-body create-view-' + (showSetup ? 'setup' : studioView) + ' create-mobile-' + mobilePane}>
          {session && <div className="create-controls-slot" hidden={!simpleView}>
            <SimpleCreateControls key={session.id + ':' + controlsRevision} snapshot={snapshot} activeTab={controlTab} onTabChange={changeControlTab}
              connected={connected} connecting={bridgeConnecting} busy={workflowPending ? bridgeBusy || 'load-workflow' : bridgeBusy} error={bridgeError} mobilePane={mobilePane}
              onPatch={patchStudio} onGenerate={generateImage} onUpload={uploadReference} onUploadMask={uploadSubjectMask} onRetry={retryConnection} onAdvanced={showAdvancedStudio}
              runInstant={runInstant} onRunInstantChange={changeRunInstant} onDraftChange={reportControlDrafts} referenceUrls={referenceUrls} onReferenceTools={openReferenceTools} historyContent={simpleView ? historyContent : null}>
              {simpleView ? resultsContent : null}
            </SimpleCreateControls>
          </div>}
          <div className="create-studio-main" hidden={!!session && simpleView}>
            {/* The same mounted graph backs simple controls and the advanced editor. */}
            {session && <iframe key={frameRevision} ref={studioFrame} title="DonutUI creation studio" src={frameUrl} className="create-studio-frame" hidden={showSetup || studioView !== 'advanced'} referrerPolicy="no-referrer" onLoad={frameLoaded} />}
            <div className="create-studio-setup" hidden={!showSetup && !!session}>
              {showSetup && !switchingServer && <CreateSettings key={serverRevision} onStatusChange={handleStatus} />}
              <button type="button" className="create-primary create-launch" disabled={switchingServer || opening || (!session && !status?.backend?.ready)} onClick={() => session ? showSimpleStudio() : launchStudio()}>
                {opening ? 'Connecting studio…' : session ? 'Return to studio' : 'Open studio'}
              </button>
              {!showSetup && !session && <p>Connecting to the creator…</p>}
            </div>
          </div>
          {session && !simpleView && resultsContent}
        </div>
      </section>
    </div>
  )
}
