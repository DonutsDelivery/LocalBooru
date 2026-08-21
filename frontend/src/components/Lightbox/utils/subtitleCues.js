function parseTimestamp(value) {
  const parts = value.trim().replace(',', '.').split(':').map(Number)
  if (parts.some(part => !Number.isFinite(part))) return null
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  if (parts.length === 2) return parts[0] * 60 + parts[1]
  return null
}

export function parseWebVttCues(source) {
  if (!source) return []
  return source.replace(/^\uFEFF/, '').split(/\r?\n\r?\n+/).flatMap((block) => {
    const lines = block.split(/\r?\n/).map(line => line.trimEnd())
    const timingIndex = lines.findIndex(line => line.includes('-->'))
    if (timingIndex < 0) return []
    const [rawStart, rawEnd] = lines[timingIndex].split('-->')
    const start = parseTimestamp(rawStart)
    const end = parseTimestamp(rawEnd.trim().split(/\s+/)[0])
    const text = lines.slice(timingIndex + 1).join('\n').trim()
    return start === null || end === null || end <= start || !text
      ? []
      : [{ start, end, text }]
  })
}

export function activeSubtitleText(cues, currentTime) {
  if (!Array.isArray(cues) || !Number.isFinite(currentTime)) return ''
  return cues
    .filter(cue => cue.start <= currentTime && currentTime < cue.end)
    .map(cue => cue.text?.trim())
    .filter(Boolean)
    .join('\n')
}
