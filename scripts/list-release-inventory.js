#!/usr/bin/env node

const fs = require('fs')
const path = require('path')

function walk(root, current = root, entries = []) {
  for (const item of fs.readdirSync(current, { withFileTypes: true })) {
    const absolute = path.join(current, item.name)
    const relative = path.relative(root, absolute).split(path.sep).join('/')
    entries.push(relative)
    if (item.isDirectory()) {
      walk(root, absolute, entries)
    } else if (item.isFile() && item.name === 'app.asar') {
      const asar = require('@electron/asar')
      for (const member of asar.listPackage(absolute)) {
        entries.push(`${relative}/${member.replace(/^[/\\]+/, '')}`)
      }
    }
  }
  return entries
}

const roots = process.argv.slice(2)
if (roots.length === 0) {
  console.error('usage: list-release-inventory.js <unpacked-root> [...]')
  process.exit(2)
}
for (const root of roots) {
  if (!fs.statSync(root).isDirectory()) {
    throw new Error(`release inventory root is not a directory: ${root}`)
  }
  for (const entry of walk(root)) {
    process.stdout.write(`${path.basename(root)}/${entry}\n`)
  }
}
