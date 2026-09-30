import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { adjustmentLocator, imageFileHash, imageIdentityKey } from '../utils/imageAdjustments.js'
import { createErrorMessage, getImageWorkflow, openImageWorkflow } from '../services/donutCreate'
import { toast } from '../components/Toast'

/** Sidecar reads belong to the displayed image and the selected DMC server. */
export function useImageWorkflow(image, enabled) {
  const locator = useMemo(() => {
    if (!image || image.is_local_direct_file) return null
    try {
      const value = adjustmentLocator(image)
      return Number.isSafeInteger(value.imageId) && value.imageId > 0
        && Number.isSafeInteger(value.directoryId) && value.directoryId > 0 ? { ...value, fileHash: imageFileHash(image) } : null
    } catch { return null }
  }, [image])
  const identity = locator ? `${imageIdentityKey(locator)}:${locator.fileHash || ''}` : ''
  const currentIdentity = useRef(identity)
  currentIdentity.current = identity
  const mounted = useRef(false)
  const generation = useRef(0)
  const switching = useRef(false)
  const summaryRequest = useRef(null)
  const loadRequest = useRef(null)
  const loadLock = useRef(false)
  const [availability, setAvailability] = useState({ identity: '', available: false })
  const [loadingIdentity, setLoading] = useState('')
  const [serverRevision, setServerRevision] = useState(0)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      generation.current += 1
      summaryRequest.current?.abort()
      loadRequest.current?.abort()
    }
  }, [])

  useEffect(() => {
    return () => {
      generation.current += 1
      summaryRequest.current?.abort()
      loadRequest.current?.abort()
      loadLock.current = false
    }
  }, [identity])

  useEffect(() => {
    const changing = () => {
      switching.current = true
      generation.current += 1
      summaryRequest.current?.abort()
      loadRequest.current?.abort()
      loadLock.current = false
      setLoading('')
      setAvailability({ identity: '', available: false })
    }
    const changed = () => {
      switching.current = false
      setServerRevision(previous => previous + 1)
    }
    window.addEventListener('donut-create-server-changing', changing)
    window.addEventListener('donut-create-server-changed', changed)
    return () => {
      window.removeEventListener('donut-create-server-changing', changing)
      window.removeEventListener('donut-create-server-changed', changed)
    }
  }, [])

  useEffect(() => {
    if (!enabled || !locator || switching.current) return
    const controller = new AbortController()
    summaryRequest.current = controller
    const requestGeneration = generation.current
    getImageWorkflow(locator, true, controller.signal)
      .then(result => {
        if (mounted.current && !controller.signal.aborted && generation.current === requestGeneration && currentIdentity.current === identity) {
          setAvailability({ identity, available: result.available === true })
        }
      })
      .catch(() => { /* Availability is optional; explicit Load reports failures. */ })
    return () => {
      controller.abort()
      setAvailability(previous => previous.identity === identity ? { identity: '', available: false } : previous)
    }
  }, [enabled, identity, locator, serverRevision])

  const loadWorkflow = useCallback(async () => {
    if (!locator || loadLock.current || switching.current) return false
    loadLock.current = true
    const controller = new AbortController()
    loadRequest.current = controller
    const requestGeneration = generation.current
    setLoading(identity)
    try {
      const result = await getImageWorkflow(locator, false, controller.signal)
      if (!mounted.current || controller.signal.aborted || generation.current !== requestGeneration || currentIdentity.current !== identity) return false
      if (!result.available) throw new Error('This image no longer has a saved workflow.')
      openImageWorkflow(result.workflow, locator, image.original_filename || image.filename || 'Saved image')
      return true
    } catch (error) {
      if (mounted.current && !controller.signal.aborted && generation.current === requestGeneration && currentIdentity.current === identity) {
        toast.error(`Could not load saved workflow: ${createErrorMessage(error)}`)
      }
      return false
    } finally {
      if (generation.current === requestGeneration) {
        loadLock.current = false
        if (mounted.current) setLoading('')
      }
    }
  }, [identity, locator, image])

  return { available: enabled && availability.identity === identity && availability.available, loading: loadingIdentity === identity && !!identity, loadWorkflow }
}
