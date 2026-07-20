export const DEFAULT_CURATION_GOAL = Object.freeze({ enabled: false, cadence: 'daily', target: 50 })
const GOAL_KEY = 'localbooru_curation_goal'
const PROGRESS_KEY = 'localbooru_curation_progress'

export function localDayKey(date = new Date()) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

export function loadCurationGoal(storage = localStorage) {
  try {
    const value = JSON.parse(storage.getItem(GOAL_KEY) || 'null')
    if (!value) return { ...DEFAULT_CURATION_GOAL }
    return {
      enabled: Boolean(value.enabled),
      cadence: value.cadence === 'weekly' ? 'weekly' : 'daily',
      target: Math.max(1, Math.floor(Number(value.target) || 50)),
    }
  } catch {
    return { ...DEFAULT_CURATION_GOAL }
  }
}

export function saveCurationGoal(goal, storage = localStorage) {
  const normalized = {
    enabled: Boolean(goal.enabled),
    cadence: goal.cadence === 'weekly' ? 'weekly' : 'daily',
    target: Math.max(1, Math.floor(Number(goal.target) || 50)),
  }
  storage.setItem(GOAL_KEY, JSON.stringify(normalized))
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('localbooru-curation-goal-changed'))
  }
  return normalized
}

function loadCounts(storage) {
  try { return JSON.parse(storage.getItem(PROGRESS_KEY) || '{}') || {} } catch { return {} }
}

export function recordCurated(now = new Date(), storage = localStorage) {
  const key = localDayKey(now)
  const counts = loadCounts(storage)
  counts[key] = (counts[key] || 0) + 1
  storage.setItem(PROGRESS_KEY, JSON.stringify(counts))
  return key
}

export function undoRecordedCurated(dayKey, storage = localStorage) {
  const counts = loadCounts(storage)
  counts[dayKey] = Math.max(0, (counts[dayKey] || 0) - 1)
  storage.setItem(PROGRESS_KEY, JSON.stringify(counts))
}

export function getCurationProgress(goal, now = new Date(), storage = localStorage) {
  const counts = loadCounts(storage)
  if (goal.cadence !== 'weekly') return counts[localDayKey(now)] || 0
  const cursor = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const day = cursor.getDay() || 7
  cursor.setDate(cursor.getDate() - day + 1)
  let total = 0
  for (let index = 0; index < 7; index += 1) {
    total += counts[localDayKey(cursor)] || 0
    cursor.setDate(cursor.getDate() + 1)
  }
  return total
}
