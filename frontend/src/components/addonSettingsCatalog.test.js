import assert from 'node:assert/strict'
import test from 'node:test'

import {
  ADDON_SETTINGS_CATALOG,
  getInstalledConfigurableAddons,
} from './addonSettingsCatalog.js'

// AC: @addon-settings ac-1
// AC: @wd14-addon-settings ac-installed-visibility
test('lists configurable add-ons in a stable order', () => {
  assert.deepEqual(
    ADDON_SETTINGS_CATALOG.map(({ id }) => id),
    ['auto-tagger', 'age-detector', 'whisper-subtitles', 'cast', 'svp', 'curation-game', 'wd14-sidecar'],
  )
})

// AC: @addon-settings ac-1
// AC: @wd14-addon-settings ac-installed-visibility
test('excludes uninstalled and non-configurable add-ons from settings', () => {
  const addons = [
    { id: 'svp', installed: true },
    { id: 'auto-tagger', installed: true },
    { id: 'cast', installed: false },
    { id: 'wd14-sidecar', installed: false },
  ]

  assert.deepEqual(
    getInstalledConfigurableAddons(addons).map(({ id }) => id),
    ['auto-tagger', 'svp'],
  )

  addons.at(-1).installed = true
  assert.deepEqual(
    getInstalledConfigurableAddons(addons).map(({ id }) => id),
    ['auto-tagger', 'svp', 'wd14-sidecar'],
  )
})
