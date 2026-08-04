import { useEffect, useMemo, useReducer, useRef } from 'react'

import { nativeVideoAPI } from '../../../tauriAPI'
import { nativeViewportForLightbox, selectPlaybackTarget } from '../nativePlaybackPolicy.js'
import { captureWebPlaybackHandoff, initialNativeVideoView, nativeNavigationAction, nativeViewFlags, reduceNativeVideoView, snapshotMatchesTarget } from './nativeVideoLifecycle.js'

const unavailableCapabilities = {
  desktop_tauri: false,
  native_renderer_available: false,
  safe_mode: false,
  desktop_player_mode: 'react',
}

export function useNativeVideo({
  image,
  isVideoFile,
  isCasting,
  castProtocol,
  mobileWebView,
  resumePosition = 0,
  mediaRef,
  onNavigate,
  onClose,
}) {
  const [capabilities, setCapabilities] = useReducer((_, value) => value, unavailableCapabilities)
  const [capabilitiesReady, setCapabilitiesReady] = useReducer(() => true, false)
  const [view, dispatch] = useReducer(reduceNativeVideoView, initialNativeVideoView)
  const containerRef = useRef(null)
  const mountedRef = useRef(true)
  const latestItemIdRef = useRef(image?.id)
  const resumePositionRef = useRef(resumePosition)
  const generationRef = useRef(view.generation)
  const onNavigateRef = useRef(onNavigate)
  const onCloseRef = useRef(onClose)
  const requestRef = useRef(0)
  const targetRef = useRef('web-fallback')
  latestItemIdRef.current = image?.id
  resumePositionRef.current = resumePosition
  generationRef.current = view.generation
  onNavigateRef.current = onNavigate
  onCloseRef.current = onClose

  useEffect(() => {
    mountedRef.current = true
    nativeVideoAPI.capabilities()
      .then(value => { if (mountedRef.current) setCapabilities(value) })
      .catch(() => {})
      .finally(() => { if (mountedRef.current) setCapabilitiesReady() })
    return () => { mountedRef.current = false }
  }, [])

  useEffect(() => {
    let unlisten = () => {}
    let disposed = false
    nativeVideoAPI.subscribe({
      onSnapshot: snapshot => {
        if (!snapshotMatchesTarget(snapshot, targetRef.current)) return
        if (snapshot?.generation >= generationRef.current) generationRef.current = snapshot.generation
        dispatch({ type: 'snapshot', snapshot })
      },
      onEvent: event => {
        const action = nativeNavigationAction(event, generationRef.current)
        if (action === 'close') onCloseRef.current?.()
        else if (action) onNavigateRef.current?.(action)
      },
      // Generation-filtered snapshots are authoritative for fallback. Bare
      // runtime diagnostics must not tear down a newer native generation.
      onError: () => {},
      onExit: () => {},
    }).then(cleanup => {
      if (disposed) cleanup()
      else unlisten = cleanup
    })
    return () => {
      disposed = true
      unlisten()
    }
  }, [])

  const target = useMemo(() => capabilitiesReady ? selectPlaybackTarget({
    desktopTauri: capabilities.desktop_tauri,
    nativeRendererAvailable: capabilities.native_renderer_available && Boolean(image?.file_path),
    safeMode: capabilities.safe_mode,
    desktopPlayerMode: capabilities.desktop_player_mode,
    isVideo: isVideoFile,
    isCasting,
    castProtocol,
    mobileWebView,
    localFileAvailable: Boolean(image?.file_path),
  }) : 'pending', [capabilitiesReady, capabilities, image?.file_path, isVideoFile, isCasting, castProtocol, mobileWebView])
  targetRef.current = target

  useEffect(() => {
    if (!image?.direct_file) return
    import('@tauri-apps/api/core').then(({ invoke }) => invoke('report_direct_file_stage', {
      stage: `target=${target} desktop=${capabilities.desktop_tauri} native=${capabilities.native_renderer_available} safe=${capabilities.safe_mode}`,
    })).catch(() => {})
  }, [capabilities.desktop_tauri, capabilities.native_renderer_available, capabilities.safe_mode, image?.direct_file, target])

  useEffect(() => {
    if (!image?.id) return
    if (target === 'pending') return
    const request = ++requestRef.current
    if (target === 'native-local') {
      const handoff = captureWebPlaybackHandoff(mediaRef?.current, resumePositionRef.current)
      dispatch({ type: 'target-native' })
      nativeVideoAPI.open(image.id, image.file_path, handoff?.position ?? resumePositionRef.current)
        .then(async snapshot => {
          if (!snapshot || request !== requestRef.current) return
          generationRef.current = snapshot.generation
          dispatch({ type: 'snapshot', snapshot })
          if (handoff) {
            await Promise.all([
              nativeVideoAPI.setPaused(handoff.paused),
              nativeVideoAPI.setVolume(handoff.volume),
              nativeVideoAPI.setMuted(handoff.muted),
              nativeVideoAPI.setSpeed(handoff.speed),
            ])
          } else if (image?.direct_file && image?.muted) {
            await nativeVideoAPI.setMuted(true)
          }
          if (capabilities.desktop_player_mode === 'native_svp' || (image?.direct_file && image?.svp)) {
            // SVP is an enhancement to native playback, not a reason to tear
            // down a healthy ordinary native session when the external runtime
            // is missing or rejects the request.
            await nativeVideoAPI.setInterpolation('svp', 'balanced', 60).catch(() => false)
          }
        })
        .catch(error => {
          if (request === requestRef.current) {
            dispatch({ type: 'runtime-error', error: error?.message })
          }
        })
    } else {
      dispatch({ type: 'target-web', isVideo: isVideoFile })
      nativeVideoAPI.showImage(image.id)
        .then(snapshot => {
          if (snapshot && request === requestRef.current) dispatch({ type: 'snapshot', snapshot })
        })
        .catch(() => {})
    }
  }, [capabilities.desktop_player_mode, image?.id, image?.file_path, image?.direct_file, image?.muted, image?.svp, isVideoFile, mediaRef, target])

  const flags = nativeViewFlags(view)

  useEffect(() => {
    const element = containerRef.current
    if (!element || !flags.useNative) return
    let frame = 0
    let previous = ''
    const update = () => {
      const bounds = element.getBoundingClientRect()
      const viewport = nativeViewportForLightbox(bounds, flags.visible)
      const signature = `${viewport.x}:${viewport.y}:${viewport.width}:${viewport.height}:${viewport.visible}`
      if (signature !== previous) {
        previous = signature
        nativeVideoAPI.setViewport(viewport).catch(() => {})
      }
      frame = window.requestAnimationFrame(update)
    }
    frame = window.requestAnimationFrame(update)
    return () => window.cancelAnimationFrame(frame)
  }, [flags.useNative, flags.visible, image?.id])

  useEffect(() => () => {
    if (latestItemIdRef.current) nativeVideoAPI.showImage(latestItemIdRef.current).catch(() => {})
  }, [])

  return {
    ...flags,
    target,
    error: view.error,
    resolving: target === 'pending',
    containerRef,
  }
}
