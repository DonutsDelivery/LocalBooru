import assert from 'node:assert/strict'
import test from 'node:test'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  TaskDetailsButton,
  TaskDetailsPanel,
  taskDiagnosticModel,
} from './taskDiagnostics.js'

const failedTask = {
  id: 41,
  task_type: 'tag',
  status: 'failed',
  attempts: 3,
  error_message: 'FOREIGN KEY constraint failed while saving predicted tags',
  payload: { library_id: 'library-a', directory_id: 7, image_id: 12 },
  created_at: '2026-07-22 10:00:00',
  started_at: '2026-07-22 10:01:00',
  completed_at: '2026-07-22 10:01:02',
  next_attempt_at: null,
}

// AC: @actionable-task-diagnostics ac-failed-details
test('failed task details expose target, attempts, error, and timestamps', () => {
  const model = taskDiagnosticModel(failedTask)
  assert.equal(model.target, 'Library library-a · Directory 7 · Image 12')
  assert.equal(model.attempts, '3 of 3')
  assert.match(model.completedAt, /2026-07-22 10:01:02 UTC/)

  const html = renderToStaticMarkup(React.createElement(TaskDetailsPanel, { task: failedTask }))
  assert.match(html, /FOREIGN KEY constraint failed/)
  assert.match(html, /Directory 7/)
  assert.match(html, /3 of 3/)
})

// AC: @actionable-task-diagnostics ac-retry-details
test('pending delayed task is identified as a scheduled retry', () => {
  const task = {
    ...failedTask,
    status: 'pending',
    attempts: 1,
    completed_at: null,
    next_attempt_at: '2026-07-22 10:05:00',
  }
  const model = taskDiagnosticModel(task)
  assert.equal(model.statusLabel, 'Retry scheduled')
  assert.match(model.nextAttemptAt, /2026-07-22 10:05:00 UTC/)
})

// AC: @actionable-task-diagnostics ac-accessible-details
test('details control is keyboard-native and linked to wrapped diagnostic content', () => {
  const button = renderToStaticMarkup(React.createElement(TaskDetailsButton, {
    task: failedTask,
    expanded: false,
    onToggle: () => {},
  }))
  assert.match(button, /<button/)
  assert.match(button, /aria-expanded="false"/)
  assert.match(button, /aria-controls="task-details-41"/)
  assert.match(button, /aria-label="Details for tag 41"/)

  const longTask = { ...failedTask, error_message: 'x'.repeat(4096) }
  const panel = renderToStaticMarkup(React.createElement(TaskDetailsPanel, { task: longTask }))
  assert.match(panel, /<pre>x{100}/)
  assert.match(panel, /id="task-details-41"/)
})

// AC: @actionable-task-diagnostics ac-current-only
test('ordinary successful task has no diagnostic expansion', () => {
  const task = { ...failedTask, status: 'completed', error_message: null, next_attempt_at: null }
  assert.equal(taskDiagnosticModel(task).canExpand, false)
  assert.equal(renderToStaticMarkup(React.createElement(TaskDetailsButton, {
    task,
    expanded: false,
    onToggle: () => {},
  })), '')
})
