export const ADDON_SETTINGS_CATALOG = Object.freeze([
  { id: 'auto-tagger', label: 'Auto Tagger' },
  { id: 'age-detector', label: 'Age Detector' },
  { id: 'whisper-subtitles', label: 'Whisper Subtitles' },
  { id: 'cast', label: 'Chromecast & DLNA' },
  { id: 'svp', label: 'SVP' },
])

export function getInstalledConfigurableAddons(addons) {
  const installedIds = new Set(
    addons.filter((addon) => addon.installed).map((addon) => addon.id),
  )

  return ADDON_SETTINGS_CATALOG.filter((addon) => installedIds.has(addon.id))
}
