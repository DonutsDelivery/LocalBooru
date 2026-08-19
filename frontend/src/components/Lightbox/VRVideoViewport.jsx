import { useCallback, useEffect, useRef, useState } from 'react'
import {
  DEFAULT_VR_CAMERA,
  DEFAULT_VR_FOV,
  detectVRProjection,
  detectVRStereo,
  normalizeYaw,
  updateVRCamera,
  updateVRFov,
} from './utils/vrVideo.js'

const VERTEX_SHADER = `
  attribute vec2 aPosition;
  varying vec2 vUv;

  void main() {
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`

const FRAGMENT_SHADER = `
  precision mediump float;

  varying vec2 vUv;
  uniform sampler2D uVideo;
  uniform float uAspect;
  uniform float uFov;
  uniform mat4 uRotationMatrix;
  uniform int uProjection;
  uniform int uStereo;
  uniform int uEye;

  const float PI = 3.14159265358979323846;

  void main() {
    float focalScale = tan(radians(uFov) * 0.5);
    vec2 screen = vec2((vUv.x * 2.0 - 1.0) * uAspect, vUv.y * 2.0 - 1.0);
    vec3 direction = normalize(vec3(screen * focalScale, 1.0));

    // Apply rotation matrix (computed on CPU)
    direction = (uRotationMatrix * vec4(direction, 0.0)).xyz;

    float longitude = atan(direction.x, direction.z);
    float latitude = asin(clamp(direction.y, -1.0, 1.0));
    vec2 sourceUv;

    if (uProjection == 1) {
      if (abs(longitude) > PI * 0.5 || abs(latitude) > PI * 0.5) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
      }
      sourceUv = vec2(longitude / PI + 0.5, 0.5 - latitude / PI);
    } else {
      sourceUv = vec2(fract(longitude / (2.0 * PI) + 0.5), 0.5 - latitude / PI);
    }

    float eye = float(uEye);
    if (uStereo == 1) {
      sourceUv.x = sourceUv.x * 0.5 + eye * 0.5;
    } else if (uStereo == 2) {
      sourceUv.y = sourceUv.y * 0.5 + eye * 0.5;
    }

    gl_FragColor = texture2D(uVideo, sourceUv);
  }
`

function compileShader(gl, type, source) {
  const shader = gl.createShader(type)
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'Unknown WebGL shader error'
    gl.deleteShader(shader)
    throw new Error(message)
  }
  return shader
}

function createProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER)
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER)
  const program = gl.createProgram()
  gl.attachShader(program, vertex)
  gl.attachShader(program, fragment)
  gl.linkProgram(program)
  gl.deleteShader(vertex)
  gl.deleteShader(fragment)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || 'Unknown WebGL link error'
    gl.deleteProgram(program)
    throw new Error(message)
  }
  return program
}

function pointerDistance(points) {
  const [first, second] = Array.from(points.values())
  if (!first || !second) return null
  return Math.hypot(second.x - first.x, second.y - first.y)
}

export default function VRVideoViewport({
  active,
  videoRef,
  mediaKey,
  filename,
  onUnavailable,
  onTap,
  onContextMenu,
  onInteraction,
}) {
  const canvasRef = useRef(null)
  const cameraRef = useRef({ ...DEFAULT_VR_CAMERA })
  const configRef = useRef({
    projection: detectVRProjection(filename) || '360',
    stereo: detectVRStereo(filename),
    eye: 'left',
    fov: DEFAULT_VR_FOV,
  })
  const pointerPositionsRef = useRef(new Map())
  const mousePointerIdRef = useRef(null)
  const gestureRef = useRef({ moved: false, distance: null })
  const [camera, setCamera] = useState(cameraRef.current)
  const [config, setConfig] = useState(configRef.current)
  const [dragging, setDragging] = useState(false)
  const [subtitleText, setSubtitleText] = useState('')

  // Diagnostic modes — use refs so render loop sees current values without re-creating effect
  const freezeTextureRef = useRef(false)
  const forceDpr1Ref = useRef(false)
  const [showPerfStats, setShowPerfStats] = useState(false)
  // Perf state for reactive overlay updates
  const [perfStats, setPerfStats] = useState({
    uploadCount: 0,
    drawCount: 0,
    avgUploadMs: 0,
    avgDrawMs: 0,
  })
  const perfAccumRef = useRef({
    uploadCount: 0,
    drawCount: 0,
    uploadTotalMs: 0,
    drawTotalMs: 0,
  })

  const applyCamera = useCallback((updater) => {
    const next = typeof updater === 'function' ? updater(cameraRef.current) : updater
    cameraRef.current = next
    setCamera(next)
  }, [])

  const applyConfig = useCallback((updater) => {
    const next = typeof updater === 'function' ? updater(configRef.current) : updater
    configRef.current = next
    setConfig(next)
  }, [])

  const resetView = useCallback(() => {
    applyCamera({ ...DEFAULT_VR_CAMERA })
    applyConfig(current => ({ ...current, fov: DEFAULT_VR_FOV }))
  }, [applyCamera, applyConfig])

  // Diagnostic mode toggles — update refs immediately, state for UI
  const toggleFreezeTexture = useCallback(() => {
    const next = !freezeTextureRef.current
    freezeTextureRef.current = next
    // Reset perf on toggle
    perfAccumRef.current = { uploadCount: 0, drawCount: 0, uploadTotalMs: 0, drawTotalMs: 0 }
    setPerfStats({ uploadCount: 0, drawCount: 0, avgUploadMs: 0, avgDrawMs: 0 })
    setFreezeTextureUI(next)
  }, [])

  const toggleForceDpr1 = useCallback(() => {
    const next = !forceDpr1Ref.current
    forceDpr1Ref.current = next
    setForceDpr1UI(next)
    // Force immediate resize on DPR change
    canvasRef.current?.dispatchEvent(new Event('vrviewchange'))
  }, [])

  const togglePerfStats = useCallback(() => {
    setShowPerfStats(prev => !prev)
  }, [])

  // UI-only state for checkboxes (refs are source of truth for render loop)
  const [freezeTextureUI, setFreezeTextureUI] = useState(false)
  const [forceDpr1UI, setForceDpr1UI] = useState(false)

  // Sync UI checkboxes with refs on mount (in case of HMR)
  useEffect(() => {
    setFreezeTextureUI(freezeTextureRef.current)
    setForceDpr1UI(forceDpr1Ref.current)
  }, [])

  useEffect(() => {
    const nextConfig = {
      projection: detectVRProjection(filename) || '360',
      stereo: detectVRStereo(filename),
      eye: 'left',
      fov: DEFAULT_VR_FOV,
    }
    pointerPositionsRef.current.clear()
    gestureRef.current = { moved: false, distance: null }
    setDragging(false)
    cameraRef.current = { ...DEFAULT_VR_CAMERA }
    configRef.current = nextConfig
    setCamera(cameraRef.current)
    setConfig(nextConfig)
  }, [mediaKey, filename])

  useEffect(() => {
    if (!active) {
      setSubtitleText('')
      return undefined
    }
    const video = videoRef.current
    if (!video?.textTracks) return undefined

    const boundTracks = new Set()
    const updateSubtitle = () => {
      const text = []
      for (let index = 0; index < video.textTracks.length; index += 1) {
        const track = video.textTracks[index]
        if (track.mode === 'disabled' || !track.activeCues) continue
        for (let cueIndex = 0; cueIndex < track.activeCues.length; cueIndex += 1) {
          const cueText = track.activeCues[cueIndex]?.text?.trim()
          if (cueText) text.push(cueText)
        }
      }
      setSubtitleText(text.join('\n'))
    }
    const bindTracks = () => {
      for (let index = 0; index < video.textTracks.length; index += 1) {
        const track = video.textTracks[index]
        if (boundTracks.has(track)) continue
        track.addEventListener?.('cuechange', updateSubtitle)
        boundTracks.add(track)
      }
      updateSubtitle()
    }

    bindTracks()
    video.textTracks.addEventListener?.('addtrack', bindTracks)
    video.addEventListener('timeupdate', updateSubtitle)
    return () => {
      video.textTracks.removeEventListener?.('addtrack', bindTracks)
      video.removeEventListener('timeupdate', updateSubtitle)
      boundTracks.forEach(track => track.removeEventListener?.('cuechange', updateSubtitle))
    }
  }, [active, videoRef, mediaKey])

  useEffect(() => {
    if (!active) return undefined
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return undefined

    let gl
    let program
    let texture
    let buffer
    let animationFrame = null
    let videoFrameCallback = null
    let cancelled = false
    let frameDirty = true
    let viewDirty = true
    let lastVideoTime = -1
    let hasUploadedFirstFrame = false

    const fail = (error) => {
      if (cancelled) return
      cancelled = true
      const message = error instanceof Error ? error.message : String(error)
      onUnavailable?.(`VR view unavailable: ${message}`)
    }

    try {
      gl = canvas.getContext('webgl', {
        alpha: false,
        antialias: false,
        powerPreference: 'high-performance',
      })
      if (!gl) throw new Error('WebGL is not supported by this device')
      program = createProgram(gl)
      buffer = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
        gl.STATIC_DRAW,
      )
      texture = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        1,
        1,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        new Uint8Array([0, 0, 0, 255]),
      )
      gl.useProgram(program)
      const position = gl.getAttribLocation(program, 'aPosition')
      gl.enableVertexAttribArray(position)
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
      gl.uniform1i(gl.getUniformLocation(program, 'uVideo'), 0)
    } catch (error) {
      fail(error)
      return undefined
    }

    const resize = () => {
      const ratio = forceDpr1Ref.current ? 1 : Math.min(window.devicePixelRatio || 1, 2)
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio))
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio))
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        gl.viewport(0, 0, width, height)
        viewDirty = true
      }
    }

    const locations = {
      aspect: gl.getUniformLocation(program, 'uAspect'),
      fov: gl.getUniformLocation(program, 'uFov'),
      rotationMatrix: gl.getUniformLocation(program, 'uRotationMatrix'),
      projection: gl.getUniformLocation(program, 'uProjection'),
      stereo: gl.getUniformLocation(program, 'uStereo'),
      eye: gl.getUniformLocation(program, 'uEye'),
    }

    let textureWidth = 0
    let textureHeight = 0
    let textureInitialized = false

    const ensureTextureStorage = (w, h) => {
      if (!textureInitialized || textureWidth !== w || textureHeight !== h) {
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, texture)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
        textureWidth = w
        textureHeight = h
        textureInitialized = true
      }
    }

    // Perf stats updater (2 Hz)
    let perfUpdateTimer = null
    const updatePerfStats = () => {
      const { uploadCount, drawCount, uploadTotalMs, drawTotalMs } = perfAccumRef.current
      setPerfStats({
        uploadCount,
        drawCount,
        avgUploadMs: uploadCount > 0 ? uploadTotalMs / uploadCount : 0,
        avgDrawMs: drawCount > 0 ? drawTotalMs / drawCount : 0,
      })
    }

    const render = () => {
      if (cancelled) return

      const shouldUpload = !freezeTextureRef.current && video.readyState >= 2 && (
        frameDirty
        || (typeof video.requestVideoFrameCallback !== 'function' && video.currentTime !== lastVideoTime)
      )

      if (shouldUpload) {
        const uploadStart = performance.now()
        try {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            ensureTextureStorage(video.videoWidth, video.videoHeight)
            gl.activeTexture(gl.TEXTURE0)
            gl.bindTexture(gl.TEXTURE_2D, texture)
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, video)
          }
          lastVideoTime = video.currentTime
          frameDirty = false
          viewDirty = true
          hasUploadedFirstFrame = true
          perfAccumRef.current.uploadCount++
          perfAccumRef.current.uploadTotalMs += performance.now() - uploadStart
        } catch (error) {
          fail(error)
          return
        }
      }

      const shouldDraw = viewDirty && video.readyState >= 2 && (textureInitialized || (freezeTextureRef.current && hasUploadedFirstFrame))

      if (shouldDraw) {
        const drawStart = performance.now()
        const currentCamera = cameraRef.current
        const currentConfig = configRef.current
        gl.useProgram(program)
        gl.uniform1f(locations.aspect, canvas.width / canvas.height)
        gl.uniform1f(locations.fov, currentConfig.fov)

        // Compute rotation matrix on CPU once per frame
        const yawRad = currentCamera.yaw * Math.PI / 180
        const pitchRad = currentCamera.pitch * Math.PI / 180
        const rollRad = currentCamera.roll * Math.PI / 180

        const cy = Math.cos(yawRad), sy = Math.sin(yawRad)
        const cp = Math.cos(pitchRad), sp = Math.sin(pitchRad)
        const cr = Math.cos(rollRad), sr = Math.sin(rollRad)

        // R = Rz(roll) * Ry(pitch) * Rz(yaw) — column-major for WebGL
        const rotMatrix = new Float32Array([
          cy * cp,  cp * sy,  -sp,  0,
          cr * sy + cy * sp * sr,  cr * cy - sr * sp * sy,  cp * sr,  0,
          sr * sy - cr * cy * sp,  cr * sp * sy + cy * sr,  cp * cr,  0,
          0,  0,  0,  1,
        ])
        gl.uniformMatrix4fv(locations.rotationMatrix, false, rotMatrix)

        gl.uniform1i(locations.projection, currentConfig.projection === '180' ? 1 : 0)
        gl.uniform1i(locations.stereo, currentConfig.stereo === 'sbs' ? 1 : currentConfig.stereo === 'tb' ? 2 : 0)
        gl.uniform1i(locations.eye, currentConfig.eye === 'right' ? 1 : 0)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
        viewDirty = false
        perfAccumRef.current.drawCount++
        perfAccumRef.current.drawTotalMs += performance.now() - drawStart
      }

      // Continue rAF only if: video frames still coming, view is dirty, or camera changed
      // In freeze mode: stop after first frame uploaded and drawn
      const shouldContinue = frameDirty || viewDirty || !freezeTextureRef.current
      if (shouldContinue) {
        animationFrame = requestAnimationFrame(render)
      }
    }

    const markFrameDirty = () => {
      if (cancelled) return
      frameDirty = true
      videoFrameCallback = video.requestVideoFrameCallback(markFrameDirty)
    }
    if (typeof video.requestVideoFrameCallback === 'function') {
      videoFrameCallback = video.requestVideoFrameCallback(markFrameDirty)
    }

    const resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(resize) : null
    resizeObserver?.observe(canvas)
    // NO window.addEventListener('resize', resize) — ResizeObserver handles it
    const markViewDirty = () => { viewDirty = true }
    canvas.addEventListener('vrviewchange', markViewDirty)
    resize()
    animationFrame = requestAnimationFrame(render)

    // Start perf stats timer
    perfUpdateTimer = setInterval(updatePerfStats, 500)

    const handleContextLost = (event) => {
      event.preventDefault()
      fail(new Error('WebGL context was lost'))
    }
    canvas.addEventListener('webglcontextlost', handleContextLost)

    return () => {
      cancelled = true
      if (animationFrame !== null) cancelAnimationFrame(animationFrame)
      if (videoFrameCallback !== null && typeof video.cancelVideoFrameCallback === 'function') {
        video.cancelVideoFrameCallback(videoFrameCallback)
      }
      if (perfUpdateTimer !== null) clearInterval(perfUpdateTimer)
      resizeObserver?.disconnect()
      canvas.removeEventListener('vrviewchange', markViewDirty)
      canvas.removeEventListener('webglcontextlost', handleContextLost)
      if (document.pointerLockElement === canvas) document.exitPointerLock?.()
      if (gl) {
        if (texture) gl.deleteTexture(texture)
        if (buffer) gl.deleteBuffer(buffer)
        if (program) gl.deleteProgram(program)
      }
    }
  }, [active, videoRef, mediaKey, onUnavailable])

  useEffect(() => {
    if (!active) return
    canvasRef.current?.dispatchEvent(new Event('vrviewchange'))
  }, [active, camera, config])

  useEffect(() => {
    if (!active) return undefined
    const handleLockedMouseMove = (event) => {
      if (document.pointerLockElement !== canvasRef.current
          || mousePointerIdRef.current === null
          || !pointerPositionsRef.current.has(mousePointerIdRef.current)) return
      if (Math.abs(event.movementX) + Math.abs(event.movementY) < 0.01) return
      event.preventDefault()
      if (Math.abs(event.movementX) + Math.abs(event.movementY) > 2) {
        gestureRef.current.moved = true
      }
      onInteraction?.()
      applyCamera(current => updateVRCamera(
        current,
        event.movementX,
        event.movementY,
        configRef.current.projection,
      ))
    }
    const handleLockedMouseUp = () => {
      const pointerId = mousePointerIdRef.current
      if (pointerId === null || !pointerPositionsRef.current.has(pointerId)) return
      pointerPositionsRef.current.delete(pointerId)
      mousePointerIdRef.current = null
      setDragging(false)
      if (document.pointerLockElement === canvasRef.current) document.exitPointerLock?.()
    }
    document.addEventListener('mousemove', handleLockedMouseMove)
    document.addEventListener('mouseup', handleLockedMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleLockedMouseMove)
      document.removeEventListener('mouseup', handleLockedMouseUp)
    }
  }, [active, applyCamera, onInteraction])

  const handlePointerDown = useCallback((event) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    onInteraction?.()
    const canvas = canvasRef.current
    canvas?.focus({ preventScroll: true })
    canvas?.setPointerCapture?.(event.pointerId)
    pointerPositionsRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (event.pointerType === 'mouse') mousePointerIdRef.current = event.pointerId
    if (pointerPositionsRef.current.size === 1) {
      gestureRef.current = { moved: false, distance: null }
    } else {
      gestureRef.current.moved = true
      gestureRef.current.distance = pointerDistance(pointerPositionsRef.current)
    }
    setDragging(true)
    if (event.pointerType === 'mouse' && canvas?.requestPointerLock) {
      try {
        const request = canvas.requestPointerLock()
        request?.catch?.(() => {})
      } catch {
        // Pointer capture remains the bounded fallback on WebViews without pointer lock.
      }
    }
  }, [onInteraction])

  const handlePointerMove = useCallback((event) => {
    const points = pointerPositionsRef.current
    if (!points.has(event.pointerId)) return
    event.preventDefault()
    event.stopPropagation()
    onInteraction?.()

    const previous = points.get(event.pointerId)
    points.set(event.pointerId, { x: event.clientX, y: event.clientY })

    if (points.size >= 2) {
      const distance = pointerDistance(points)
      const previousDistance = gestureRef.current.distance
      if (distance !== null && previousDistance !== null) {
        const delta = previousDistance - distance
        if (Math.abs(delta) > 0.5) {
          gestureRef.current.moved = true
          applyConfig(current => ({ ...current, fov: updateVRFov(current.fov, delta * 0.12) }))
        }
      }
      gestureRef.current.distance = distance
      return
    }

    if (document.pointerLockElement === canvasRef.current) return
    const deltaX = event.clientX - previous.x
    const deltaY = event.clientY - previous.y
    if (Math.abs(deltaX) + Math.abs(deltaY) < 0.01) return
    if (Math.abs(deltaX) + Math.abs(deltaY) > 2) gestureRef.current.moved = true
    applyCamera(current => updateVRCamera(current, deltaX, deltaY, configRef.current.projection))
  }, [applyCamera, applyConfig, onInteraction])

  const finishPointer = useCallback((event, cancelled = false) => {
    if (!pointerPositionsRef.current.has(event.pointerId)) return
    event.preventDefault()
    event.stopPropagation()
    const wasTap = !cancelled
      && pointerPositionsRef.current.size === 1
      && !gestureRef.current.moved
    pointerPositionsRef.current.delete(event.pointerId)
    if (mousePointerIdRef.current === event.pointerId) mousePointerIdRef.current = null
    gestureRef.current.distance = pointerDistance(pointerPositionsRef.current)
    if (pointerPositionsRef.current.size === 0) {
      setDragging(false)
      if (document.pointerLockElement === canvasRef.current) document.exitPointerLock?.()
      if (wasTap) onTap?.(event)
    }
  }, [onTap])

  const handleWheel = useCallback((event) => {
    event.preventDefault()
    event.stopPropagation()
    onInteraction?.()
    const direction = event.deltaY < 0 ? -1 : 1
    applyConfig(current => ({ ...current, fov: updateVRFov(current.fov, direction * 5) }))
  }, [applyConfig, onInteraction])

  const handleKeyDown = useCallback((event) => {
    let deltaX = 0
    let deltaY = 0
    if (event.key === 'ArrowLeft') deltaX = 5
    else if (event.key === 'ArrowRight') deltaX = -5
    else if (event.key === 'ArrowUp') deltaY = 5
    else if (event.key === 'ArrowDown') deltaY = -5
    else if (event.key === '0') {
      event.preventDefault()
      event.stopPropagation()
      resetView()
      return
    } else if (event.key === '+' || event.key === '=') {
      event.preventDefault()
      event.stopPropagation()
      applyConfig(current => ({ ...current, fov: updateVRFov(current.fov, -5) }))
      return
    } else if (event.key === '-' || event.key === '_') {
      event.preventDefault()
      event.stopPropagation()
      applyConfig(current => ({ ...current, fov: updateVRFov(current.fov, 5) }))
      return
    } else {
      return
    }
    event.preventDefault()
    event.stopPropagation()
    applyCamera(current => updateVRCamera(current, deltaX, deltaY, configRef.current.projection, 1))
  }, [applyCamera, applyConfig, resetView])

  if (!active) return null

  const setProjection = (projection) => {
    applyConfig(current => ({ ...current, projection }))
    if (projection === '180') {
      applyCamera(current => ({ ...current, yaw: Math.max(-90, Math.min(90, current.yaw)) }))
    } else {
      applyCamera(current => ({ ...current, yaw: normalizeYaw(current.yaw) }))
    }
  }

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`vr-video-canvas ${dragging ? 'dragging' : ''}`}
        tabIndex="0"
        role="img"
        aria-label={`${config.projection} degree VR video view. Drag to look around.`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={(event) => finishPointer(event)}
        onPointerCancel={(event) => finishPointer(event, true)}
        onWheel={handleWheel}
        onKeyDown={handleKeyDown}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
        }}
        onContextMenu={(event) => {
          event.stopPropagation()
          onContextMenu?.(event)
        }}
        onTouchStart={(event) => event.stopPropagation()}
        onTouchMove={(event) => event.stopPropagation()}
        onTouchEnd={(event) => event.stopPropagation()}
      />
      <div
        className="vr-video-controls"
        data-curation-gesture-block
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
        onTouchStart={(event) => event.stopPropagation()}
      >
        <div className="vr-projection-toggle" aria-label="VR projection">
          <button
            className={config.projection === '180' ? 'active' : ''}
            onClick={() => setProjection('180')}
            title="180° equirectangular input"
          >180°</button>
          <button
            className={config.projection === '360' ? 'active' : ''}
            onClick={() => setProjection('360')}
            title="360° equirectangular input"
          >360°</button>
        </div>
        <select
          value={config.stereo}
          onChange={(event) => applyConfig(current => ({ ...current, stereo: event.target.value }))}
          aria-label="VR input stereo layout"
          title="Input stereo layout"
        >
          <option value="mono">Mono</option>
          <option value="sbs">Side by side</option>
          <option value="tb">Top / bottom</option>
        </select>
        {config.stereo !== 'mono' && (
          <button
            className="vr-eye-toggle"
            onClick={() => applyConfig(current => ({ ...current, eye: current.eye === 'left' ? 'right' : 'left' }))}
            title="Switch the displayed eye"
          >Eye: {config.eye === 'left' ? 'L' : 'R'}</button>
        )}
        <span className="vr-fov-readout" title="Field of view">{Math.round(config.fov)}°</span>
        <button className="vr-reset-view" onClick={resetView} title="Recenter VR view (0)" aria-label="Recenter VR view">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12a9 9 0 1 0 3-6.7" />
            <path d="M3 4v6h6" />
          </svg>
        </button>
      </div>
      <div className={`vr-drag-hint ${dragging ? 'active' : ''}`}>
        {dragging ? `${Math.round(camera.yaw)}° / ${Math.round(camera.pitch)}°` : 'Drag to look • wheel or pinch to zoom'}
      </div>
      {subtitleText && <div className="vr-video-subtitles">{subtitleText}</div>}
      {/* Diagnostic mode toggles */}
      <div className="vr-diagnostic-controls" style={{
        position: 'absolute', zIndex: 18, right: 16, top: 60,
        display: 'flex', flexDirection: 'column', gap: 6,
        background: 'rgba(10, 12, 16, 0.78)', padding: 8,
        borderRadius: 8, border: '1px solid rgba(255,255,255,0.16)',
        backdropFilter: 'blur(12px)', fontSize: '0.7rem'
      }}>
        <label style={{display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', color: '#fff'}}>
          <input type="checkbox" checked={freezeTextureUI} onChange={toggleFreezeTexture} /> Freeze texture
        </label>
        <label style={{display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', color: '#fff'}}>
          <input type="checkbox" checked={forceDpr1UI} onChange={toggleForceDpr1} /> Force DPR=1
        </label>
        <label style={{display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', color: '#fff'}}>
          <input type="checkbox" checked={showPerfStats} onChange={togglePerfStats} /> Perf stats
        </label>
      </div>
      {showPerfStats && perfStats.uploadCount > 0 && (
        <div className="vr-perf-stats" style={{
          position: 'absolute', zIndex: 18, right: 16, bottom: 180,
          background: 'rgba(0, 0, 0, 0.8)', padding: 10, borderRadius: 8,
          fontFamily: 'monospace', fontSize: '0.7rem', color: '#0f0',
          border: '1px solid rgba(255,255,255,0.16)', minWidth: 200
        }}>
          <div>Uploads: {perfStats.uploadCount}</div>
          <div>Avg upload: {perfStats.avgUploadMs.toFixed(2)}ms</div>
          <div>Draws: {perfStats.drawCount}</div>
          <div>Avg draw: {perfStats.avgDrawMs.toFixed(2)}ms</div>
          <div>Canvas: {canvasRef.current?.width}×{canvasRef.current?.height}</div>
          <div>Video: {videoRef.current?.videoWidth}×{videoRef.current?.videoHeight}</div>
        </div>
      )}
    </>
  )
}