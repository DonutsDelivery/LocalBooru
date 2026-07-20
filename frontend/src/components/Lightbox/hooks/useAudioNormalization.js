import { useCallback, useRef } from 'react'
import { getAudioGain } from '../../../api'

/**
 * Applies audio peak attenuation to a video element.
 *
 * Used for direct-play (original quality, no FFmpeg stream), since stream paths
 * (SVP / transcode) already apply peak attenuation in FFmpeg with -af volume=XdB.
 */
export function useAudioNormalization(mediaRef) {
  const attenuationRef = useRef(1)
  const outputVolumeRef = useRef(1)
  const outputMutedRef = useRef(false)
  const requestSeqRef = useRef(0)

  const applyElementVolume = useCallback(() => {
    const video = mediaRef.current
    if (!video) return
    const effectiveVolume = outputMutedRef.current
      ? 0
      : outputVolumeRef.current * attenuationRef.current
    video.volume = Math.max(0, Math.min(1, effectiveVolume))
    video.muted = outputMutedRef.current || outputVolumeRef.current === 0
  }, [mediaRef])

  // Apply attenuation for direct play. Positive gain is ignored intentionally.
  const applyNormalization = useCallback(async (filePath) => {
    const requestSeq = ++requestSeqRef.current
    attenuationRef.current = 1
    applyElementVolume()
    try {
      const { gain_db } = await getAudioGain(filePath)
      if (requestSeq !== requestSeqRef.current) return
      if (gain_db != null && gain_db < -0.5) {
        const cappedGainDb = Math.max(gain_db, -12)
        attenuationRef.current = Math.pow(10, cappedGainDb / 20)
        console.log(`[AudioNorm] Direct play attenuation: ${cappedGainDb.toFixed(1)} dB`)
      } else {
        attenuationRef.current = 1
      }
      applyElementVolume()
    } catch (e) {
      console.warn('[AudioNorm] gain fetch failed:', e?.message)
    }
  }, [applyElementVolume])

  // Reset to unity gain when switching to a stream that handles attenuation server-side.
  const resetGain = useCallback(() => {
    requestSeqRef.current += 1
    attenuationRef.current = 1
    applyElementVolume()
  }, [applyElementVolume])

  const setOutputVolume = useCallback((volume) => {
    outputVolumeRef.current = Math.max(0, Math.min(1, volume))
    if (volume > 0) outputMutedRef.current = false
    applyElementVolume()
    return true
  }, [applyElementVolume])

  const setOutputMuted = useCallback((muted, volume = 1) => {
    outputMutedRef.current = muted
    outputVolumeRef.current = Math.max(0, Math.min(1, volume))
    applyElementVolume()
    return true
  }, [applyElementVolume])

  return { applyNormalization, resetGain, setOutputVolume, setOutputMuted }
}
