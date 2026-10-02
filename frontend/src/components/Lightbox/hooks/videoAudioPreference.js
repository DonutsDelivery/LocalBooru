const storageKey = 'video_audio_preference'

function validatedPreference(value) {
  return {
    volume: typeof value?.volume === 'number' && Number.isFinite(value.volume)
      && value.volume >= 0 && value.volume <= 1 ? value.volume : 1,
    muted: typeof value?.muted === 'boolean' ? value.muted : false,
  }
}

export function readVideoAudioPreference() {
  try {
    return validatedPreference(JSON.parse(localStorage.getItem(storageKey)))
  } catch {
    return validatedPreference(null)
  }
}

export function writeVideoAudioPreference(value) {
  const preference = validatedPreference(value)
  try {
    localStorage.setItem(storageKey, JSON.stringify(preference))
  } catch {
    // Playback controls still work when device storage is unavailable.
  }
  return preference
}
