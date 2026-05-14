import { useCallback, useRef } from 'react'
import { getAudioGain } from '../../../api'

/**
 * Applies audio peak normalization to a video element via Web Audio API.
 *
 * Used for direct-play (original quality, no FFmpeg stream), since stream paths
 * (SVP / transcode) already apply normalization in FFmpeg with -af volume=XdB.
 *
 * The AudioContext + source node are created once and reused — browsers only
 * allow createMediaElementSource to be called once per element.
 */
export function useAudioNormalization(mediaRef) {
  const ctxRef = useRef(null)
  const gainNodeRef = useRef(null)

  const _ensureChain = useCallback(() => {
    if (ctxRef.current) return true
    const video = mediaRef.current
    if (!video) return false
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)()
      const source = ctx.createMediaElementSource(video)
      const gain = ctx.createGain()
      source.connect(gain)
      gain.connect(ctx.destination)
      ctxRef.current = ctx
      gainNodeRef.current = gain
      return true
    } catch (e) {
      // createMediaElementSource may fail on Android WebView or cross-origin src
      console.warn('[AudioNorm] Failed to create Web Audio chain:', e?.message)
      return false
    }
  }, [mediaRef])

  // Apply normalization for direct play — queries gain from backend and sets it.
  const applyNormalization = useCallback(async (filePath) => {
    if (!_ensureChain()) return
    try {
      const { gain_db } = await getAudioGain(filePath)
      if (gain_db != null && Math.abs(gain_db) > 0.5) {
        const linear = Math.pow(10, gain_db / 20)
        gainNodeRef.current.gain.value = linear
        console.log(`[AudioNorm] Direct play: ${gain_db > 0 ? '+' : ''}${gain_db.toFixed(1)} dB (×${linear.toFixed(1)})`)
      } else {
        gainNodeRef.current.gain.value = 1.0
      }
      if (ctxRef.current.state === 'suspended') {
        ctxRef.current.resume().catch(() => {})
      }
    } catch (e) {
      console.warn('[AudioNorm] gain fetch failed:', e?.message)
    }
  }, [_ensureChain])

  // Reset to unity gain — call when switching to a stream that already normalizes.
  const resetGain = useCallback(() => {
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = 1.0
    }
  }, [])

  return { applyNormalization, resetGain }
}
