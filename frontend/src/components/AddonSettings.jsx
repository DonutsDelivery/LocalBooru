import { useEffect, useMemo, useState } from 'react'
import { getAddons } from '../api'
import AgeDetectionSettings from './AgeDetectionSettings'
import AutoTaggerSettings from './AutoTaggerSettings'
import CastSettings from './CastSettings'
import SVPSettings from './SVPSettings'
import WhisperSubtitleSettings from './WhisperSubtitleSettings'
import { getInstalledConfigurableAddons } from './addonSettingsCatalog'
import './AddonSettings.css'

const SETTINGS_COMPONENTS = {
  'auto-tagger': AutoTaggerSettings,
  'age-detector': AgeDetectionSettings,
  'whisper-subtitles': WhisperSubtitleSettings,
  cast: CastSettings,
  svp: SVPSettings,
}

export default function AddonSettings() {
  const [addons, setAddons] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    getAddons()
      .then((data) => {
        if (active) setAddons(data.addons || [])
      })
      .catch((error) => console.error('Failed to load add-ons for settings:', error))
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => { active = false }
  }, [])

  const configurableAddons = useMemo(
    () => getInstalledConfigurableAddons(addons),
    [addons],
  )

  if (loading) {
    return <section className="optical-flow-settings"><h2>Add-on Settings</h2><p className="settings-description">Loading...</p></section>
  }

  if (configurableAddons.length === 0) {
    return (
      <section className="optical-flow-settings">
        <h2>Add-on Settings</h2>
        <p className="settings-description">Install a configurable add-on from Addons to manage its settings here.</p>
      </section>
    )
  }

  const activeId = configurableAddons.some((addon) => addon.id === selectedId)
    ? selectedId
    : configurableAddons[0].id
  const SettingsComponent = SETTINGS_COMPONENTS[activeId]

  return (
    <div className="addon-settings">
      <nav className="addon-settings-nav" aria-label="Add-on settings">
        {configurableAddons.map((addon) => (
          <button
            key={addon.id}
            type="button"
            className={addon.id === activeId ? 'active' : ''}
            aria-pressed={addon.id === activeId}
            onClick={() => setSelectedId(addon.id)}
          >
            {addon.label}
          </button>
        ))}
      </nav>
      {SettingsComponent && <SettingsComponent />}
    </div>
  )
}
