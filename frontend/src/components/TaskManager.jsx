import { useState, useEffect, useCallback, useRef } from 'react'
import { getQueueTasks, getQueueStatus, cancelTask, clearCompletedTasks, retryFailedTasks, pauseQueue, resumeQueue, getQueuePaused, resetQueue, subscribeToLibraryEvents } from '../api'
import './TaskManager.css'

const TASK_TYPE_LABELS = {
  scan_directory: 'Scan Directory',
  complete_directory_imports: 'Generate Thumbnails',
  tag: 'Auto-Tag',
  age_detect: 'Age Detection',
  extract_metadata: 'Extract Metadata',
  verify_files: 'Verify Files',
  upload: 'Upload',
}

const STATUS_CLASSES = {
  pending: 'status-pending',
  processing: 'status-processing',
  completed: 'status-completed',
  failed: 'status-failed',
  cancelled: 'status-cancelled',
}

function formatDuration(startedAt, completedAt) {
  if (!startedAt) return '-'
  const start = new Date(startedAt + 'Z')
  const end = completedAt ? new Date(completedAt + 'Z') : new Date()
  const ms = end - start
  if (ms < 1000) return '<1s'
  if (ms < 60000) return `${Math.round(ms / 1000)}s`
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`
  return `${Math.floor(ms / 3600000)}h ${Math.round((ms % 3600000) / 60000)}m`
}

function formatTimeAgo(dateStr) {
  if (!dateStr) return '-'
  const date = new Date(dateStr + 'Z')
  const now = new Date()
  const ms = now - date
  if (ms < 60000) return 'just now'
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ago`
  if (ms < 86400000) return `${Math.floor(ms / 3600000)}h ago`
  return `${Math.floor(ms / 86400000)}d ago`
}

function getDirName(task) {
  const p = task.payload
  if (p.directory_path) {
    const parts = p.directory_path.split('/')
    return parts[parts.length - 1] || p.directory_path
  }
  if (p.directory_id) return `Directory #${p.directory_id}`
  return null
}

/** Group tasks by (task_type, directory) for active statuses, keep individual rows for failed */
function groupTasks(tasks) {
  const groups = []
  const groupMap = new Map() // key -> group index

  for (const task of tasks) {
    // Failed/cancelled tasks show individually (useful to see errors)
    if (task.status === 'failed' || task.status === 'cancelled') {
      groups.push({ type: 'single', task })
      continue
    }

    const dirName = getDirName(task) || '-'
    const key = `${task.task_type}::${task.payload?.directory_id ?? 'none'}`

    if (groupMap.has(key)) {
      const group = groups[groupMap.get(key)]
      group.count++
      if (task.status === 'processing') group.processing++
      if (task.status === 'pending') group.pending++
      if (task.status === 'completed') group.completed++
      // Track earliest started_at for duration
      if (task.started_at && (!group.earliestStarted || task.started_at < group.earliestStarted)) {
        group.earliestStarted = task.started_at
      }
      group.taskIds.push(task.id)
    } else {
      groupMap.set(key, groups.length)
      groups.push({
        type: 'group',
        taskType: task.task_type,
        dirName,
        directoryId: task.payload?.directory_id,
        count: 1,
        processing: task.status === 'processing' ? 1 : 0,
        pending: task.status === 'pending' ? 1 : 0,
        completed: task.status === 'completed' ? 1 : 0,
        earliestStarted: task.started_at,
        taskIds: [task.id],
      })
    }
  }

  return groups
}

export default function TaskManager() {
  const [tasks, setTasks] = useState([])
  const [total, setTotal] = useState(0)
  const [stats, setStats] = useState(null)
  const [paused, setPaused] = useState(false)
  const [statusFilter, setStatusFilter] = useState('')
  const [typeFilter, setTypeFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState({})
  // Track progress for long-running tasks: { "task_type::directory_id": { processed, total } }
  const [progress, setProgress] = useState({})
  const mounted = useRef(true)

  const loadTasks = useCallback(async () => {
    try {
      const data = await getQueueTasks(statusFilter || null, typeFilter || null, 200, 0)
      if (mounted.current) {
        setTasks(data.tasks)
        setTotal(data.total)
      }
    } catch (e) {
      console.error('Failed to load tasks:', e)
    }
  }, [statusFilter, typeFilter])

  const loadStats = useCallback(async () => {
    try {
      const data = await getQueueStatus()
      if (mounted.current) setStats(data)
    } catch (e) {
      console.error('Failed to load queue stats:', e)
    }
  }, [])

  const loadPaused = useCallback(async () => {
    try {
      const data = await getQueuePaused()
      if (mounted.current) setPaused(data.paused)
    } catch (e) {
      console.error('Failed to load pause status:', e)
    }
  }, [])

  // Initial load
  useEffect(() => {
    mounted.current = true
    setLoading(true)
    Promise.all([loadTasks(), loadStats(), loadPaused()]).finally(() => {
      if (mounted.current) setLoading(false)
    })
    return () => { mounted.current = false }
  }, [loadTasks, loadStats, loadPaused])

  // SSE auto-refresh on task events
  useEffect(() => {
    const unsubscribe = subscribeToLibraryEvents((event) => {
      if (event.type === 'task_started' || event.type === 'task_completed') {
        loadTasks()
        loadStats()
        // Clear progress for completed tasks
        if (event.type === 'task_completed' && event.data) {
          const key = `${event.data.task_type}::${event.data.directory_id ?? 'none'}`
          setProgress(prev => {
            const next = { ...prev }
            delete next[key]
            return next
          })
        }
      }
      if (event.type === 'task_progress' && event.data) {
        const d = event.data
        const key = `${d.task_type}::${d.directory_id ?? 'none'}`
        setProgress(prev => ({ ...prev, [key]: { processed: d.processed, total: d.total } }))
      }
    })
    return unsubscribe
  }, [loadTasks, loadStats])

  // Poll while there's active work
  useEffect(() => {
    const hasActive = stats?.by_status?.processing > 0 || stats?.by_status?.pending > 0
    if (!hasActive) return
    const interval = setInterval(() => {
      loadTasks()
      loadStats()
    }, 5000)
    return () => clearInterval(interval)
  }, [stats?.by_status?.processing, stats?.by_status?.pending, loadTasks, loadStats])

  const handlePauseToggle = async () => {
    if (paused) {
      await resumeQueue()
      setPaused(false)
    } else {
      await pauseQueue()
      setPaused(true)
    }
  }

  const handleCancel = async (taskId) => {
    setActionLoading(prev => ({ ...prev, [taskId]: true }))
    try {
      await cancelTask(taskId)
      await loadTasks()
      await loadStats()
    } catch (e) {
      console.error('Failed to cancel task:', e)
    }
    setActionLoading(prev => ({ ...prev, [taskId]: false }))
  }

  const handleCancelGroup = async (taskIds) => {
    try {
      await Promise.all(taskIds.map(id => cancelTask(id)))
      await loadTasks()
      await loadStats()
    } catch (e) {
      console.error('Failed to cancel tasks:', e)
    }
  }

  const handleClearCompleted = async () => {
    try {
      await clearCompletedTasks(0)
      await loadTasks()
      await loadStats()
    } catch (e) {
      console.error('Failed to clear completed:', e)
    }
  }

  const handleRetryFailed = async () => {
    try {
      await retryFailedTasks()
      await loadTasks()
      await loadStats()
    } catch (e) {
      console.error('Failed to retry:', e)
    }
  }

  const pendingCount = stats?.by_status?.pending || 0
  const processingCount = stats?.by_status?.processing || 0
  const completedCount = stats?.by_status?.completed || 0
  const failedCount = stats?.by_status?.failed || 0
  const cancelledCount = stats?.by_status?.cancelled || 0

  const grouped = groupTasks(tasks)

  return (
    <div className="task-manager">
      <section>
        <h2>Task Queue</h2>

        {/* Stats summary */}
        <div className="task-stats-bar">
          <span className="task-stat">
            <span className="task-stat-dot pending" /> {pendingCount} pending
          </span>
          <span className="task-stat">
            <span className="task-stat-dot processing" /> {processingCount} processing
          </span>
          <span className="task-stat">
            <span className="task-stat-dot completed" /> {completedCount} completed
          </span>
          {failedCount > 0 && (
            <span className="task-stat">
              <span className="task-stat-dot failed" /> {failedCount} failed
            </span>
          )}
          {cancelledCount > 0 && (
            <span className="task-stat">
              <span className="task-stat-dot cancelled" /> {cancelledCount} cancelled
            </span>
          )}
        </div>

        {/* Action buttons */}
        <div className="task-actions-bar">
          <button
            className={`task-action-btn ${paused ? 'paused' : ''}`}
            onClick={handlePauseToggle}
          >
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
          <button className="task-action-btn" onClick={handleClearCompleted}>
            Clear Completed
          </button>
          {failedCount > 0 && (
            <button className="task-action-btn" onClick={handleRetryFailed}>
              Retry Failed
            </button>
          )}
          <button
            className="task-action-btn danger"
            onClick={async () => {
              if (!confirm('Delete ALL tasks? The tag guardian will re-queue what\'s needed.')) return
              await resetQueue()
              await loadTasks()
              await loadStats()
            }}
          >
            Reset Queue
          </button>
        </div>

        {/* Filters */}
        <div className="task-filters">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All Statuses</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
          >
            <option value="">All Types</option>
            {Object.entries(TASK_TYPE_LABELS).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </div>

        {/* Task list */}
        {loading ? (
          <div className="task-loading">Loading tasks...</div>
        ) : grouped.length === 0 ? (
          <div className="task-empty">No tasks found</div>
        ) : (
          <div className="task-list">
            <div className="task-list-header">
              <span className="task-col-type">Type</span>
              <span className="task-col-target">Directory</span>
              <span className="task-col-status">Status</span>
              <span className="task-col-time">Time</span>
              <span className="task-col-actions">Actions</span>
            </div>
            {grouped.map((item, i) => {
              if (item.type === 'single') {
                const task = item.task
                const singleProgKey = `${task.task_type}::${task.payload?.directory_id ?? 'none'}`
                const singleProg = task.status === 'processing' ? progress[singleProgKey] : null
                return (
                  <div key={task.id} className={`task-row ${task.status}`}>
                    <span className="task-col-type" title={task.task_type}>
                      {TASK_TYPE_LABELS[task.task_type] || task.task_type}
                    </span>
                    <span className="task-col-target" title={task.payload?.directory_path || ''}>
                      {getDirName(task) || '-'}
                    </span>
                    <span className="task-col-status">
                      {singleProg && singleProg.total > 0 ? (
                        <span className="task-progress-info">
                          <span className="task-spinner" />
                          <span>{singleProg.processed}/{singleProg.total}</span>
                          <span className="task-progress-bar-wrap">
                            <span className="task-progress-bar-fill" style={{ width: `${Math.round(singleProg.processed / singleProg.total * 100)}%` }} />
                          </span>
                        </span>
                      ) : (
                        <>
                          <span className={`task-status-badge ${STATUS_CLASSES[task.status] || ''}`}>
                            {task.status === 'processing' && <span className="task-spinner" />}
                            {task.status}
                          </span>
                          {task.error_message && (
                            <span className="task-error-hint" title={task.error_message}>!</span>
                          )}
                        </>
                      )}
                    </span>
                    <span className="task-col-time">
                      {task.status === 'processing' || task.status === 'completed' || task.status === 'failed'
                        ? formatDuration(task.started_at, task.completed_at)
                        : formatTimeAgo(task.created_at)
                      }
                    </span>
                    <span className="task-col-actions">
                      {(task.status === 'pending' || task.status === 'processing') && (
                        <button
                          className="task-cancel-btn"
                          onClick={() => handleCancel(task.id)}
                          disabled={actionLoading[task.id]}
                          title="Cancel task"
                        >
                          ✕
                        </button>
                      )}
                    </span>
                  </div>
                )
              }

              // Grouped row
              const g = item
              const isActive = g.processing > 0
              const statusClass = isActive ? 'processing' : g.pending > 0 ? '' : 'completed'
              const progressKey = `${g.taskType}::${g.directoryId ?? 'none'}`
              const prog = progress[progressKey]
              return (
                <div key={`group-${i}`} className={`task-row ${statusClass}`}>
                  <span className="task-col-type" title={g.taskType}>
                    {TASK_TYPE_LABELS[g.taskType] || g.taskType}
                  </span>
                  <span className="task-col-target">
                    {g.dirName}
                  </span>
                  <span className="task-col-status">
                    {prog && prog.total > 0 ? (
                      <span className="task-progress-info">
                        <span className="task-spinner" />
                        <span>{prog.processed}/{prog.total}</span>
                        <span className="task-progress-bar-wrap">
                          <span className="task-progress-bar-fill" style={{ width: `${Math.round(prog.processed / prog.total * 100)}%` }} />
                        </span>
                      </span>
                    ) : (
                      <>
                        <span className={`task-status-badge ${isActive ? 'status-processing' : 'status-pending'}`}>
                          {isActive && <span className="task-spinner" />}
                          {g.processing > 0 && `${g.processing} active`}
                          {g.processing > 0 && g.pending > 0 && ', '}
                          {g.pending > 0 && `${g.pending} queued`}
                          {g.completed > 0 && g.processing === 0 && g.pending === 0 && `${g.completed} done`}
                        </span>
                        <span className="task-group-count">{g.count} total</span>
                      </>
                    )}
                  </span>
                  <span className="task-col-time">
                    {g.earliestStarted ? formatDuration(g.earliestStarted, null) : '-'}
                  </span>
                  <span className="task-col-actions">
                    {(g.pending > 0 || g.processing > 0) && (
                      <button
                        className="task-cancel-btn"
                        onClick={() => handleCancelGroup(g.taskIds)}
                        title={`Cancel all ${g.count} tasks`}
                      >
                        ✕
                      </button>
                    )}
                  </span>
                </div>
              )
            })}
            {total > tasks.length && (
              <div className="task-more">
                Showing {grouped.length} groups from {total} tasks
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  )
}
