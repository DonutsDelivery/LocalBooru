"""
Background task queue for LocalBooru - handles tagging, file verification, etc.
"""
import asyncio
import json
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal
from ..models import TaskQueue, TaskStatus, TaskType, Image, ImageFile
from ..config import get_settings

settings = get_settings()


class BackgroundTaskQueue:
    """In-process background task processor"""

    def __init__(self):
        self.running = False
        self.worker_task = None
        self.concurrency = settings.task_queue_concurrency

    async def start(self):
        """Start the background worker"""
        # Reset any tasks stuck in 'processing' from previous crash
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                update(TaskQueue)
                .where(TaskQueue.status == TaskStatus.processing)
                .values(status=TaskStatus.pending)
            )
            if result.rowcount > 0:
                await db.commit()
                print(f"[TaskQueue] Reset {result.rowcount} stuck tasks from previous session")

        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        print(f"Task queue started with concurrency={self.concurrency}")

    async def stop(self):
        """Stop the background worker"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print("Task queue stopped")

    async def _worker_loop(self):
        """Main worker loop - processes tasks from the queue"""
        print("[TaskQueue] Worker loop started", flush=True)
        loop_count = 0
        while self.running:
            loop_count += 1
            try:
                # Get pending task IDs (using temporary session)
                async with AsyncSessionLocal() as db:
                    tasks = await self._get_pending_tasks(db, limit=self.concurrency)
                    task_ids = [t.id for t in tasks]

                if loop_count % 10 == 0:
                    print(f"[TaskQueue] Heartbeat: loop={loop_count}, pending tasks found={len(task_ids)}", flush=True)

                if not task_ids:
                    await asyncio.sleep(1)
                    continue

                print(f"[TaskQueue] Processing {len(task_ids)} tasks...", flush=True)

                # Process tasks concurrently - each with its own session
                results = await asyncio.gather(*[
                    self._process_task_by_id(task_id) for task_id in task_ids
                ], return_exceptions=True)

                # Log any exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"[TaskQueue] Task {i} failed: {result}")

                # Small delay between batches to let DB breathe
                await asyncio.sleep(0.5)

            except Exception as e:
                import traceback
                print(f"Task queue error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _get_pending_tasks(self, db: AsyncSession, limit: int = 2):
        """Get pending tasks, ordered by priority (high first) and creation time"""
        query = (
            select(TaskQueue)
            .where(TaskQueue.status == TaskStatus.pending)
            .order_by(TaskQueue.priority.desc(), TaskQueue.created_at)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    async def _process_task_by_id(self, task_id: int):
        """Process a single task by ID, creating its own database session"""
        async with AsyncSessionLocal() as db:
            # Fetch the task
            result = await db.execute(
                select(TaskQueue).where(TaskQueue.id == task_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                return

            # Skip if no longer pending (another worker may have picked it up)
            if task.status != TaskStatus.pending:
                return

            await self._process_task(task, db)

    async def _process_task(self, task: TaskQueue, db: AsyncSession):
        """Process a single task"""
        print(f"[TaskQueue] Starting task {task.id} ({task.task_type.value})")

        # Mark as processing
        task.status = TaskStatus.processing
        task.started_at = datetime.now()
        task.attempts += 1
        await db.commit()

        try:
            if task.task_type == TaskType.tag:
                await self._process_tag_task(task, db)
            elif task.task_type == TaskType.scan_directory:
                await self._process_scan_task(task, db)
            elif task.task_type == TaskType.verify_files:
                await self._process_verify_task(task, db)
            elif task.task_type == TaskType.upload:
                await self._process_upload_task(task, db)
            elif task.task_type == TaskType.age_detect:
                await self._process_age_detect_task(task, db)
            elif task.task_type == TaskType.extract_metadata:
                await self._process_metadata_task(task, db)

            task.status = TaskStatus.completed
            task.completed_at = datetime.now()
            print(f"[TaskQueue] Task {task.id} completed")

        except Exception as e:
            import traceback
            print(f"[TaskQueue] Task {task.id} error: {e}")
            traceback.print_exc()
            task.error_message = str(e)
            # Retry up to 3 times
            if task.attempts >= 3:
                task.status = TaskStatus.failed
            else:
                task.status = TaskStatus.pending

        await db.commit()

    async def _process_tag_task(self, task: TaskQueue, db: AsyncSession):
        """Run tagger on an image"""
        payload = json.loads(task.payload)
        image_id = payload['image_id']
        image_path = payload['image_path']

        # Import tagger here to avoid circular imports
        from .tagger import tag_image
        await tag_image(image_path, db, image_id)

    async def _process_scan_task(self, task: TaskQueue, db: AsyncSession):
        """Scan a directory for images"""
        payload = json.loads(task.payload)
        directory_id = payload['directory_id']
        directory_path = payload['directory_path']

        from .file_tracker import scan_directory
        await scan_directory(directory_id, directory_path, db)

    async def _process_verify_task(self, task: TaskQueue, db: AsyncSession):
        """Verify file locations still exist"""
        from .file_tracker import verify_file_locations
        await verify_file_locations(db)

    async def _process_upload_task(self, task: TaskQueue, db: AsyncSession):
        """Upload image to external booru"""
        payload = json.loads(task.payload)
        image_id = payload['image_id']
        booru_id = payload['booru_id']

        from .external_upload import upload_to_booru
        await upload_to_booru(image_id, booru_id, db)

    async def _process_age_detect_task(self, task: TaskQueue, db: AsyncSession):
        """Run age detection on an image"""
        payload = json.loads(task.payload)
        image_id = payload['image_id']
        image_path = payload['image_path']

        from .age_detector import detect_ages
        from ..models import Image

        result = await detect_ages(image_path)
        if result is None:
            return

        # Update image with age detection results
        image_result = await db.execute(
            select(Image).where(Image.id == image_id)
        )
        image = image_result.scalar_one_or_none()
        if image:
            image.num_faces = result.num_faces
            if result.min_age is not None:
                image.min_detected_age = result.min_age
                image.max_detected_age = result.max_age
                image.detected_ages = json.dumps(result.ages)
                image.age_detection_data = json.dumps(result.to_dict())
            await db.commit()

    async def _process_metadata_task(self, task: TaskQueue, db: AsyncSession):
        """Extract AI generation metadata from an image"""
        payload = json.loads(task.payload)
        image_id = payload['image_id']
        image_path = payload['image_path']
        comfyui_prompt_node_ids = payload.get('comfyui_prompt_node_ids', [])
        comfyui_negative_node_ids = payload.get('comfyui_negative_node_ids', [])
        format_hint = payload.get('format_hint', 'auto')
        extract_tags = payload.get('extract_tags', True)

        from .metadata_extractor import extract_and_save_metadata_with_tags

        result, added_tags = await extract_and_save_metadata_with_tags(
            image_path,
            image_id,
            db,
            comfyui_prompt_node_ids,
            comfyui_negative_node_ids,
            format_hint,
            extract_tags
        )

        # Log result for debugging
        if result.status == 'success':
            tag_info = f" (added {len(added_tags)} tags)" if added_tags else ""
            print(f"[TaskQueue] Metadata extracted for image {image_id}{tag_info}")
        elif result.status == 'config_mismatch':
            print(f"[TaskQueue] ComfyUI config mismatch for image {image_id}: {result.message}")
        elif result.status == 'no_metadata':
            pass  # Silent - most images won't have metadata


# Global task queue instance
task_queue = BackgroundTaskQueue()


async def enqueue_task(
    task_type: TaskType,
    payload: dict,
    priority: int = 0,
    db: AsyncSession = None
) -> TaskQueue:
    """Add a task to the queue"""
    task = TaskQueue(
        task_type=task_type,
        payload=json.dumps(payload),
        priority=priority,
        status=TaskStatus.pending
    )

    if db:
        db.add(task)
        await db.commit()
        await db.refresh(task)
    else:
        async with AsyncSessionLocal() as session:
            session.add(task)
            await session.commit()
            await session.refresh(task)

    return task
