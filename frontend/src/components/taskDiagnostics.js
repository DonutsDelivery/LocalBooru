import React from 'react'

const MAX_ATTEMPTS = 3

function utcTimestamp(value) {
  if (!value) return null
  const normalized = value.endsWith('Z') ? value : `${value}Z`
  const date = new Date(normalized)
  if (Number.isNaN(date.getTime())) return value
  return date.toISOString().replace('T', ' ').replace('.000Z', ' UTC')
}

export function taskDiagnosticModel(task) {
  const payload = task.payload || {}
  const retryScheduled = task.status === 'pending' && Boolean(task.next_attempt_at)
  const target = [
    payload.library_id && `Library ${payload.library_id}`,
    payload.directory_id != null && `Directory ${payload.directory_id}`,
    payload.image_id != null && `Image ${payload.image_id}`,
    payload.image_path || payload.directory_path,
  ].filter(Boolean)

  return {
    id: task.id,
    detailsId: `task-details-${task.id}`,
    canExpand: Boolean(task.error_message || retryScheduled || task.status === 'failed'),
    statusLabel: retryScheduled ? 'Retry scheduled' : task.status,
    target: target.length ? target.join(' · ') : 'No target details',
    attempts: `${task.attempts || 0} of ${MAX_ATTEMPTS}`,
    error: task.error_message || null,
    createdAt: utcTimestamp(task.created_at),
    startedAt: utcTimestamp(task.started_at),
    completedAt: utcTimestamp(task.completed_at),
    nextAttemptAt: utcTimestamp(task.next_attempt_at),
  }
}

export function TaskDetailsButton({ task, expanded, onToggle }) {
  const details = taskDiagnosticModel(task)
  if (!details.canExpand) return null
  return React.createElement(
    'button',
    {
      type: 'button',
      className: 'task-details-btn',
      'aria-expanded': expanded,
      'aria-controls': details.detailsId,
      'aria-label': `${expanded ? 'Hide details' : 'Details'} for ${task.task_type || 'task'} ${task.id}`,
      onClick: onToggle,
    },
    expanded ? 'Hide details' : 'Details',
  )
}

export function TaskDetailsPanel({ task }) {
  const details = taskDiagnosticModel(task)
  const fields = [
    ['Target', details.target],
    ['Attempts', details.attempts],
    ['Created', details.createdAt],
    ['Started', details.startedAt],
    ['Completed', details.completedAt],
    ['Next retry', details.nextAttemptAt],
  ].filter(([, value]) => value)

  return React.createElement(
    'div',
    { id: details.detailsId, className: 'task-details-panel' },
    React.createElement(
      'dl',
      null,
      fields.flatMap(([label, value]) => [
        React.createElement('dt', { key: `${label}-label` }, label),
        React.createElement('dd', { key: `${label}-value` }, value),
      ]),
    ),
    details.error
      ? React.createElement(
          'div',
          { className: 'task-details-error' },
          React.createElement('strong', null, 'Latest error'),
          React.createElement('pre', null, details.error),
        )
      : null,
  )
}
