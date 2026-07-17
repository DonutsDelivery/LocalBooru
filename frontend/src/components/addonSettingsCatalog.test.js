import assert from 'node:assert/strict'
import test from 'node:test'

import {
  ADDON_SETTINGS_CATALOG,
  getInstalledConfigurableAddons,
} from './addonSettingsCatalog.js'

// AC: @addon-settings ac-1
test('lists configurable add-ons in a stable order', () => {
  assert.deepEqual(
    ADDON_SETTINGS_CATALOG.map(({ id }) => id),
    ['auto-tagger', 'age-detector', 'whisper-subtitles', 'cast', 'svp', 'curation-game'],
  )
})

// AC: @addon-settings ac-1
test('excludes uninstalled and non-configurable add-ons from settings', () => {
  const addons = [
    { id: 'svp', installed: true },
    { id: 'auto-tagger', installed: true },
    { id: 'cast', installed: false },
  ]

  assert.deepEqual(
    getInstalledConfigurableAddons(addons).map(({ id }) => id),
    ['auto-tagger', 'svp'],
  )
})
