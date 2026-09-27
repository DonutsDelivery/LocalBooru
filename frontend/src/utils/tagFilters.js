export function splitTagFilters(tags = '', excludeTags = '') {
  const included = []
  const excluded = []

  for (const raw of String(tags || '').split(',')) {
    const tag = raw.trim()
    if (!tag) continue
    if (tag.startsWith('-')) {
      const name = tag.slice(1).trim()
      if (name) excluded.push(name)
    } else {
      included.push(tag)
    }
  }

  for (const raw of String(excludeTags || '').split(',')) {
    const tag = raw.trim()
    if (tag) excluded.push(tag)
  }

  return { included: included.join(','), excluded: [...new Set(excluded)].join(',') }
}

export function toggleTagFilter(tags, tagName) {
  const selected = String(tagName || '').trim()
  const active = String(tags || '').split(',').map(tag => tag.trim()).filter(Boolean)
  if (!selected) return active.join(',')
  if (active.includes(selected)) return active.filter(tag => tag !== selected).join(',')

  const opposite = selected.startsWith('-') ? selected.slice(1) : `-${selected}`
  return [...active.filter(tag => tag !== opposite), selected].join(',')
}
