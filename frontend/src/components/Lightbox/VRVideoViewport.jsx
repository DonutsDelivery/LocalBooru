import { useCallback, useEffect, useRef, useState } from 'react'
import {
  DEFAULT_VR_CAMERA,
  DEFAULT_VR_FOV,
  detectVRInputProjection,
  detectVRProjection,
  detectVRStereo,
  normalizeYaw,
  uploadVRVideoFrame,
  updateVRCamera,
  updateVRFov,
  vrPointerDelta,
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
  uniform float uYaw;
  uniform float uPitch;
  uniform float uRoll;
  uniform int uInputProjection;
  uniform int uProjection;
  uniform int uStereo;
  uniform int uEye;

  const float PI = 3.14159265358979323846;

  void main() {
    float focalScale = tan(radians(uFov) * 0.5);
    vec2 screen = vec2((vUv.x * 2.0 - 1.0) * uAspect, vUv.y * 2.0 - 1.0);
    vec3 direction = normalize(vec3(screen * focalScale, 1.0));

    float rollCos = cos(uRoll);
    float rollSin = sin(uRoll);
    direction.xy = vec2(
      direction.x * rollCos - direction.y * rollSin,
      direction.x * rollSin + direction.y * rollCos
    );

    float pitchCos = cos(uPitch);
    float pitchSin = sin(uPitch);
    direction.yz = vec2(
      direction.y * pitchCos + direction.z * pitchSin,
      -direction.y * pitchSin + direction.z * pitchCos
    );

    float yawCos = cos(uYaw);
    float yawSin = sin(uYaw);
    direction.xz = vec2(
      direction.x * yawCos + direction.z * yawSin,
      -direction.x * yawSin + direction.z * yawCos
    );

    vec2 sourceUv;

    if (uInputProjection == 1) {
      float theta = acos(clamp(direction.z, -1.0, 1.0));
      if (theta > PI * 0.5) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
      }
      float sourceAngle = atan(direction.y, direction.x);
      float sourceRadius = theta / PI;
      sourceUv = vec2(
        0.5 + cos(sourceAngle) * sourceRadius,
        0.5 - sin(sourceAngle) * sourceRadius
      );
    } else {
      float longitude = atan(direction.x, direction.z);
      float latitude = asin(clamp(direction.y, -1.0, 1.0));
      if (uProjection == 1) {
        if (abs(longitude) > PI * 0.5 || abs(latitude) > PI * 0.5) {
          gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
          return;
        }
        sourceUv = vec2(longitude / PI + 0.5, 0.5 - latitude / PI);
      } else {
        sourceUv = vec2(fract(longitude / (2.0 * PI) + 0.5), 0.5 - latitude / PI);
      }
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
    inputProjection: detectVRInputProjection(filename),
    projection: detectVRProjection(filename) || '360',
    stereo: detectVRStereo(filename),
    eye: 'left',
    fov: DEFAULT_VR_FOV,
  })
  const pointerPositionsRef = useRef(new Map())
  const gestureRef = useRef({ moved: false, distance: null })
  const [camera, setCamera] = useState(cameraRef.current)
  const [config, setConfig] = useState(configRef.current)
  const [dragging, setDragging] = useState(false)
  const [subtitleText, setSubtitleText] = useState('')

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

  useEffect(() => {
    const nextConfig = {
      inputProjection: detectVRInputProjection(filename),
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
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
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
      yaw: gl.getUniformLocation(program, 'uYaw'),
      pitch: gl.getUniformLocation(program, 'uPitch'),
      roll: gl.getUniformLocation(program, 'uRoll'),
      inputProjection: gl.getUniformLocation(program, 'uInputProjection'),
      projection: gl.getUniformLocation(program, 'uProjection'),
      stereo: gl.getUniformLocation(program, 'uStereo'),
      eye: gl.getUniformLocation(program, 'uEye'),
    }

    let textureInitialized = false

    const hasVideoFrameCallback = typeof video.requestVideoFrameCallback === 'function'

    const requestRender = () => {
      if (cancelled || animationFrame !== null) return
      animationFrame = requestAnimationFrame(() => {
        animationFrame = null
        render()
      })
    }

    const render = () => {
      if (cancelled) return

      const shouldUpload = video.readyState >= 2 && (
        frameDirty
        || (!hasVideoFrameCallback && video.currentTime !== lastVideoTime)
      )

      if (shouldUpload) {
        try {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            gl.activeTexture(gl.TEXTURE0)
            gl.bindTexture(gl.TEXTURE_2D, texture)
            uploadVRVideoFrame(gl, video)
            textureInitialized = true
          }
          lastVideoTime = video.currentTime
          frameDirty = false
          viewDirty = true
        } catch (error) {
          fail(error)
          return
        }
      }

      const shouldDraw = viewDirty && video.readyState >= 2 && textureInitialized

      if (shouldDraw) {
        const currentCamera = cameraRef.current
        const currentConfig = configRef.current
        gl.useProgram(program)
        gl.uniform1f(locations.aspect, canvas.width / canvas.height)
        gl.uniform1f(locations.fov, currentConfig.fov)
        gl.uniform1f(locations.yaw, currentCamera.yaw * Math.PI / 180)
        gl.uniform1f(locations.pitch, currentCamera.pitch * Math.PI / 180)
        gl.uniform1f(locations.roll, currentCamera.roll * Math.PI / 180)
        gl.uniform1i(locations.inputProjection, currentConfig.inputProjection === 'fisheye' ? 1 : 0)
        gl.uniform1i(locations.projection, currentConfig.projection === '180' ? 1 : 0)
        gl.uniform1i(locations.stereo, currentConfig.stereo === 'sbs' ? 1 : currentConfig.stereo === 'tb' ? 2 : 0)
        gl.uniform1i(locations.eye, currentConfig.eye === 'right' ? 1 : 0)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
        viewDirty = false
      }

      // requestVideoFrameCallback schedules the next real decoded frame. Older
      // WebViews keep polling only while playback can advance.
      if (frameDirty || viewDirty || (!hasVideoFrameCallback && !video.paused)) requestRender()
    }

    const markFrameDirty = () => {
      if (cancelled) return
      frameDirty = true
      requestRender()
      videoFrameCallback = video.requestVideoFrameCallback(markFrameDirty)
    }
    if (hasVideoFrameCallback) {
      videoFrameCallback = video.requestVideoFrameCallback(markFrameDirty)
    }

    const resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
      resize()
      requestRender()
    }) : null
    resizeObserver?.observe(canvas)
    // NO window.addEventListener('resize', resize) — ResizeObserver handles it
    const markViewDirty = () => {
      viewDirty = true
      requestRender()
    }
    canvas.addEventListener('vrviewchange', markViewDirty)
    resize()
    requestRender()

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
      resizeObserver?.disconnect()
      canvas.removeEventListener('vrviewchange', markViewDirty)
      canvas.removeEventListener('webglcontextlost', handleContextLost)
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

  const handlePointerDown = useCallback((event) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    onInteraction?.()
    const canvas = canvasRef.current
    canvas?.focus({ preventScroll: true })
    canvas?.setPointerCapture?.(event.pointerId)
    pointerPositionsRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY })
    if (pointerPositionsRef.current.size === 1) {
      gestureRef.current = { moved: false, distance: null }
    } else {
      gestureRef.current.moved = true
      gestureRef.current.distance = pointerDistance(pointerPositionsRef.current)
    }
    setDragging(true)
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

    const { deltaX, deltaY } = vrPointerDelta(previous, event.clientX, event.clientY)
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
    gestureRef.current.distance = pointerDistance(pointerPositionsRef.current)
    if (pointerPositionsRef.current.size === 0) {
      setDragging(false)
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

  const setInputProjection = (inputProjection) => {
    applyConfig(current => ({
      ...current,
      inputProjection,
      projection: inputProjection === 'fisheye' ? '180' : current.projection,
    }))
    if (inputProjection === 'fisheye') {
      applyCamera(current => ({ ...current, yaw: Math.max(-90, Math.min(90, current.yaw)) }))
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
        <select
          value={config.inputProjection}
          onChange={(event) => setInputProjection(event.target.value)}
          aria-label="VR input projection"
          title="Input projection"
        >
          <option value="equirectangular">Equirectangular (rectangular)</option>
          <option value="fisheye">Fisheye</option>
        </select>
        <div className="vr-projection-toggle" aria-label="VR projection">
          <button
            className={config.projection === '180' ? 'active' : ''}
            onClick={() => setProjection('180')}
            title="180° coverage"
          >180°</button>
          <button
            className={config.projection === '360' ? 'active' : ''}
            onClick={() => setProjection('360')}
            disabled={config.inputProjection === 'fisheye'}
            title={config.inputProjection === 'fisheye' ? 'Fisheye input is 180°' : '360° coverage'}
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
    </>
  )
}