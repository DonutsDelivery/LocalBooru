import { useState } from 'react'
import { getCurationProgress, loadCurationGoal, saveCurationGoal } from '../utils/curationProgress'

export default function CurationGameSettings() {
  const [goal, setGoal] = useState(() => loadCurationGoal())
  const [saved, setSaved] = useState(false)
  const progress = getCurationProgress(goal)

  const save = () => {
    setGoal(saveCurationGoal(goal))
    setSaved(true)
    setTimeout(() => setSaved(false), 1500)
  }

  return (
    <section className="optical-flow-settings">
      <h2>Curation Game</h2>
      <p className="settings-description">Set an optional target for the number of items you want to review.</p>
      <label className="setting-row">
        <span>Track a curation goal</span>
        <input type="checkbox" checked={goal.enabled} onChange={event => setGoal(value => ({ ...value, enabled: event.target.checked }))} />
      </label>
      <label className="setting-row">
        <span>Period</span>
        <select value={goal.cadence} disabled={!goal.enabled} onChange={event => setGoal(value => ({ ...value, cadence: event.target.value }))}>
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
        </select>
      </label>
      <label className="setting-row">
        <span>Target items</span>
        <input type="number" min="1" value={goal.target} disabled={!goal.enabled} onChange={event => setGoal(value => ({ ...value, target: event.target.value }))} />
      </label>
      {goal.enabled && <p className="settings-description">Current progress: {progress} / {goal.target}</p>}
      <button className="save-btn" type="button" onClick={save}>{saved ? 'Saved' : 'Save'}</button>
    </section>
  )
}
