"""
Task queue management endpoints
"""
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...models import TaskQueue, TaskStatus, TaskType

router = APIRouter()


@router.get("/queue")
async def queue_status(db: AsyncSession = Depends(get_db)):
    """Get background task queue status"""
    # Count by status
    status_query = (
        select(TaskQueue.status, func.count(TaskQueue.id))
        .group_by(TaskQueue.status)
    )
    status_result = await db.execute(status_query)
    by_status = {row[0].value: row[1] for row in status_result}

    # Count by type
    type_query = (
        select(TaskQueue.task_type, func.count(TaskQueue.id))
        .where(TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing]))
        .group_by(TaskQueue.task_type)
    )
    type_result = await db.execute(type_query)
    by_type = {row[0].value: row[1] for row in type_result}

    # Recent failed tasks
    failed_query = (
        select(TaskQueue)
        .where(TaskQueue.status == TaskStatus.failed)
        .order_by(TaskQueue.completed_at.desc())
        .limit(10)
    )
    failed_result = await db.execute(failed_query)
    failed_tasks = [
        {
            "id": t.id,
            "type": t.task_type.value,
            "error": t.error_message,
            "attempts": t.attempts
        }
        for t in failed_result.scalars().all()
    ]

    return {
        "by_status": by_status,
        "pending_by_type": by_type,
        "recent_failures": failed_tasks
    }


@router.post("/queue/retry-failed")
async def retry_failed_tasks(db: AsyncSession = Depends(get_db)):
    """Retry all failed tasks"""
    result = await db.execute(
        update(TaskQueue)
        .where(TaskQueue.status == TaskStatus.failed)
        .values(status=TaskStatus.pending, attempts=0, error_message=None)
    )
    await db.commit()

    return {"retried": result.rowcount}


@router.get("/queue/paused")
async def get_queue_paused():
    """Check if the task queue is paused"""
    from ...services.task_queue import task_queue, is_gpu_busy
    return {
        "paused": task_queue.paused,
        "auto_paused": task_queue.auto_paused,
        "gpu_busy": is_gpu_busy()
    }


@router.post("/queue/pause")
async def pause_queue():
    """Pause the task queue"""
    from ...services.task_queue import task_queue
    task_queue.pause()
    return {"paused": True}


@router.post("/queue/resume")
async def resume_queue():
    """Resume the task queue"""
    from ...services.task_queue import task_queue
    task_queue.resume()
    return {"paused": False}


@router.delete("/queue/pending")
async def clear_pending_tasks(db: AsyncSession = Depends(get_db)):
    """Clear all pending tasks from the queue"""
    result = await db.execute(
        delete(TaskQueue).where(TaskQueue.status == TaskStatus.pending)
    )
    await db.commit()

    return {"cleared": result.rowcount}


@router.delete("/queue/pending/directory/{directory_id}")
async def clear_directory_pending_tasks(
    directory_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Clear pending tag tasks for a specific directory"""
    # Get all pending tag tasks
    pending_query = select(TaskQueue).where(
        TaskQueue.task_type == TaskType.tag,
        TaskQueue.status == TaskStatus.pending
    )
    result = await db.execute(pending_query)
    pending_tasks = result.scalars().all()

    # Filter by directory_id in payload and delete
    cleared = 0
    for task in pending_tasks:
        try:
            payload = json.loads(task.payload)
            if payload.get('directory_id') == directory_id:
                await db.delete(task)
                cleared += 1
        except:
            pass

    await db.commit()
    return {"cleared": cleared}


@router.post("/clear-duplicate-tasks")
async def clear_duplicate_tasks_endpoint(db: AsyncSession = Depends(get_db)):
    """Remove duplicate pending tasks from the queue.

    Keeps the oldest task for each image_id and marks duplicates as failed.
    """
    from ...services.task_queue import clear_duplicate_tasks
    removed = await clear_duplicate_tasks(db)
    return {"duplicates_removed": removed}


@router.delete("/clear-pending-tasks")
async def clear_all_pending_tasks(
    task_type: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Clear all pending tasks, optionally filtered by type.

    WARNING: This will cancel all queued work!
    """
    query = (
        update(TaskQueue)
        .where(TaskQueue.status == TaskStatus.pending)
    )

    if task_type:
        try:
            tt = TaskType(task_type)
            query = query.where(TaskQueue.task_type == tt)
        except ValueError:
            raise HTTPException(400, f"Invalid task type: {task_type}")

    query = query.values(status=TaskStatus.failed, error_message="Cancelled by user")

    result = await db.execute(query)
    await db.commit()

    return {"cancelled": result.rowcount}
