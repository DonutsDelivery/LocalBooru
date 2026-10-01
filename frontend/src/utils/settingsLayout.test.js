import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const appCss = readFileSync(new URL('../App.css', import.meta.url), 'utf8')
const migrationCss = readFileSync(new URL('../components/MigrationSettings.css', import.meta.url), 'utf8')
const historyCss = readFileSync(new URL('../components/ContinueWatching.css', import.meta.url), 'utf8')
function rule(css, selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return css.match(new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]+)\\}`))?.[1] || ''
}

test('Settings owns scrolling and directory lists cannot collapse to a zero flex basis', () => {
  assert.match(rule(appCss, '.content > .page'), /overflow-y:\s*auto/)
  assert.doesNotMatch(rule(appCss, '.directories-page'), /display:\s*flex|overflow:\s*hidden/)
  assert.match(rule(appCss, '.directories-page .directory-list'), /max-height:\s*none/)
  assert.match(rule(appCss, '.directories-page .directory-list'), /overflow:\s*visible/)
})

test('migration directory styles stay inside their selector and history actions stay beside the heading', () => {
  assert.doesNotMatch(migrationCss, /(?:^|\n)\.directory-(?:list|item|info|name|path|stats)\b/)
  assert.match(rule(historyCss, '.continue-watching-header'), /justify-content:\s*flex-start/)
  assert.match(rule(historyCss, '.continue-watching-header'), /flex-wrap:\s*wrap/)
})
