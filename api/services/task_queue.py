"""
Background task queue for LocalBooru - handles tagging, file verification, etc.
"""
import asyncio
import json
from datetime import datetime, timedelta
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal
from ..models import TaskQueue, TaskStatus, TaskType, Image, ImageFile
from ..config import get_settings

settings = get_settings()


class BackgroundTaskQueue:
    """In-process background task processor with Tag Guardian"""

    def __init__(self):
        self.running = False
        self.paused = False
        self.worker_task = None
        self.guardian_task = None
        self.concurrency = settings.task_queue_concurrency

    def pause(self):
        """Pause task processing"""
        self.paused = True
        print("[TaskQueue] Paused")

    def resume(self):
        """Resume task processing"""
        self.paused = False
        print("[TaskQueue] Resumed")

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
        self.guardian_task = asyncio.create_task(self._tag_guardian_loop())
        print(f"Task queue started with concurrency={self.concurrency}")
        print(f"Tag Guardian started (interval={settings.tag_guardian_interval}s)")

    async def stop(self):
        """Stop the background worker and tag guardian"""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        if self.guardian_task:
            self.guardian_task.cancel()
            try:
                await self.guardian_task
            except asyncio.CancelledError:
                pass
        print("Task queue and Tag Guardian stopped")

    async def _worker_loop(self):
        """Main worker loop - processes tasks from the queue"""
        print("[TaskQueue] Worker loop started", flush=True)
        loop_count = 0
        while self.running:
            loop_count += 1

            # Check if paused
            if self.paused:
                await asyncio.sleep(1)
                continue

            try:
                # Get pending task IDs (using temporary session)
                async with AsyncSessionLocal() as db:
                    tasks = await self._get_pending_tasks(db, limit=self.concurrency)
                    task_ids = [t.id for t in tasks]

                if loop_count % 10 == 0:
                    print(f"[TaskQueue] Heartbeat: loop={loop_count}, pending tasks found={len(task_ids)}, paused={self.paused}", flush=True)

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

    async def _tag_guardian_loop(self):
        """Tag Guardian - periodic loop that catches untagged images and retries failed tasks.

        This ensures no image stays untagged for more than ~10 minutes, even if:
        - Bulk imports set auto_tag=False
        - DB locks caused task creation to fail
        - Tasks failed after 3 retries
        """
        # Initial delay to let system settle after startup
        await asyncio.sleep(60)
        print("[TagGuardian] Started monitoring for untagged images")

        while self.running:
            # Skip guardian work when paused
            if self.paused:
                await asyncio.sleep(settings.tag_guardian_interval)
                continue

            try:
                async with AsyncSessionLocal() as db:
                    queued, retried = await self._run_tag_guardian(db)
                    if queued > 0 or retried > 0:
                        print(f"[TagGuardian] Queued {queued} untagged images, retried {retried} failed tasks")
            except Exception as e:
                print(f"[TagGuardian] Error: {e}")

            await asyncio.sleep(settings.tag_guardian_interval)

    async def _run_tag_guardian(self, db: AsyncSession) -> tuple[int, int]:
        """Find untagged images and retry old failures.

        Returns (queued_count, retried_count)
        """
        from ..models import image_tags

        # 1. Find untagged images (no entries in image_tags, file exists)
        tagged_subq = select(image_tags.c.image_id).distinct()
        untagged_query = (
            select(Image.id, ImageFile.original_path)
            .join(ImageFile, ImageFile.image_id == Image.id)
            .where(
                Image.id.not_in(tagged_subq),
                ImageFile.file_exists == True
            )
            .limit(settings.tag_guardian_batch_size)
        )

        # 2. Get already-queued image IDs to avoid duplicates
        pending_query = select(TaskQueue.payload).where(
            TaskQueue.task_type == TaskType.tag,
            TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing])
        )
        pending_result = await db.execute(pending_query)
        already_queued = set()
        for row in pending_result.scalars():
            try:
                already_queued.add(json.loads(row).get('image_id'))
            except:
                pass

        # 3. Queue untagged images
        untagged_result = await db.execute(untagged_query)
        queued = 0
        for image_id, image_path in untagged_result:
            if image_id not in already_queued:
                task = TaskQueue(
                    task_type=TaskType.tag,
                    payload=json.dumps({'image_id': image_id, 'image_path': image_path}),
                    priority=0,  # Low priority - don't interrupt new imports
                    status=TaskStatus.pending
                )
                db.add(task)
                queued += 1

        # 4. Retry old failed tasks (older than configured age)
        retry_cutoff = datetime.now() - timedelta(seconds=settings.tag_guardian_retry_age)
        retry_result = await db.execute(
            update(TaskQueue)
            .where(
                TaskQueue.task_type == TaskType.tag,
                TaskQueue.status == TaskStatus.failed,
                TaskQueue.completed_at < retry_cutoff
            )
            .values(status=TaskStatus.pending, attempts=0, error_message=None)
        )
        retried = retry_result.rowcount

        if queued > 0 or retried > 0:
            await db.commit()

        return queued, retried

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
        directory_id = payload.get('directory_id')  # New: directory-specific images

        # Import tagger here to avoid circular imports
        from .tagger import tag_image
        await tag_image(image_path, db, image_id, directory_id=directory_id)

    async def _process_scan_task(self, task: TaskQueue, db: AsyncSession):
        """Scan a directory for images"""
        from ..models import WatchDirectory

        payload = json.loads(task.payload)
        directory_id = payload['directory_id']
        directory_path = payload['directory_path']
        clean_deleted = payload.get('clean_deleted', False)

        # Get recursive setting from directory config
        directory = await db.get(WatchDirectory, directory_id)
        recursive = directory.recursive if directory else True

        from .file_tracker import scan_directory
        await scan_directory(directory_id, directory_path, db, recursive=recursive, clean_deleted=clean_deleted)

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
        directory_id = payload.get('directory_id')  # New: directory-specific images

        from .age_detector import detect_ages
        from ..models import Image, DirectoryImage
        from ..database import directory_db_manager

        result = await detect_ages(image_path)
        if result is None:
            return

        if directory_id:
            # Update image in directory database
            dir_db = await directory_db_manager.get_session(directory_id)
            try:
                image_result = await dir_db.execute(
                    select(DirectoryImage).where(DirectoryImage.id == image_id)
                )
                image = image_result.scalar_one_or_none()
                if image:
                    image.num_faces = result.num_faces
                    if result.min_age is not None:
                        image.min_detected_age = result.min_age
                        image.max_detected_age = result.max_age
                        image.detected_ages = json.dumps(result.ages)
                        image.age_detection_data = json.dumps(result.to_dict())
                    await dir_db.commit()
            finally:
                await dir_db.close()
        else:
            # Legacy: Update image in main database
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
        """Extract AI generation metadata from an image.

        Also handles 'complete_import' flag for fast-imported images that need
        dimensions, thumbnails, etc. calculated in background.
        """
        payload = json.loads(task.payload)
        image_id = payload['image_id']
        image_path = payload['image_path']
        directory_id = payload.get('directory_id')
        complete_import = payload.get('complete_import', False)
        auto_tag = payload.get('auto_tag', True)

        # Handle fast-import completion first
        if complete_import and directory_id:
            await self._complete_fast_import(image_id, image_path, directory_id, auto_tag, db)
            return

        # Regular metadata extraction
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
            extract_tags,
            directory_id=directory_id
        )

        if result.status == 'success':
            tag_info = f" (added {len(added_tags)} tags)" if added_tags else ""
            print(f"[TaskQueue] Metadata extracted for image {image_id}{tag_info}")
        elif result.status == 'config_mismatch':
            print(f"[TaskQueue] ComfyUI config mismatch for image {image_id}: {result.message}")

    async def _complete_fast_import(self, image_id: int, image_path: str, directory_id: int, auto_tag: bool, db: AsyncSession):
        """Complete a fast-imported image: calculate dimensions, generate thumbnail."""
        from pathlib import Path
        from ..database import directory_db_manager
        from ..models import DirectoryImage
        from ..config import get_settings
        from .importer import (
            is_video_file, get_image_dimensions_async, calculate_perceptual_hash_async,
            generate_thumbnail_async, generate_video_thumbnail_async
        )
        from .video_preview import get_video_duration_async

        settings = get_settings()

        if not Path(image_path).exists():
            return

        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            # Get the image record
            result = await dir_db.execute(
                select(DirectoryImage).where(DirectoryImage.id == image_id)
            )
            image = result.scalar_one_or_none()
            if not image:
                return

            is_video = is_video_file(image_path)

            # Calculate dimensions and perceptual hash
            if is_video:
                duration = await get_video_duration_async(image_path)
                image.duration = duration
            else:
                dimensions, phash = await asyncio.gather(
                    get_image_dimensions_async(image_path),
                    calculate_perceptual_hash_async(image_path)
                )
                if dimensions:
                    image.width, image.height = dimensions
                image.perceptual_hash = phash

            await dir_db.commit()

            # Generate thumbnail
            thumbnails_dir = Path(settings.thumbnails_dir)
            thumbnails_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_path = thumbnails_dir / f"{image.file_hash[:16]}.webp"

            if is_video:
                await generate_video_thumbnail_async(image_path, str(thumbnail_path))
            else:
                await generate_thumbnail_async(image_path, str(thumbnail_path))

            # Queue tagging if enabled
            if auto_tag:
                await enqueue_task(
                    TaskType.tag,
                    {'image_id': image_id, 'directory_id': directory_id, 'image_path': image_path},
                    priority=1,
                    db=db
                )

        finally:
            await dir_db.close()


# Global task queue instance
task_queue = BackgroundTaskQueue()


async def enqueue_task(
    task_type: TaskType,
    payload: dict,
    priority: int = 0,
    db: AsyncSession = None,
    dedupe_key: str = None
) -> TaskQueue | None:
    """Add a task to the queue.

    Args:
        task_type: Type of task
        payload: Task payload dict
        priority: Task priority (higher = processed first)
        db: Database session (optional)
        dedupe_key: If provided, skip if a pending/processing task with same
                    task_type and dedupe_key already exists. For image tasks,
                    this should be the image_id.

    Returns:
        TaskQueue object if created, None if skipped due to deduplication
    """
    async def _do_enqueue(session: AsyncSession) -> TaskQueue | None:
        # Check for duplicate pending/processing task
        if dedupe_key is not None:
            # Use proper JSON boundary matching to avoid false positives
            # e.g., image_id: 1 should not match image_id: 10 or 100
            from sqlalchemy import or_
            existing = await session.execute(
                select(TaskQueue.id)
                .where(
                    TaskQueue.task_type == task_type,
                    TaskQueue.status.in_([TaskStatus.pending, TaskStatus.processing]),
                    or_(
                        TaskQueue.payload.like(f'%"image_id": {dedupe_key},%'),  # Middle of JSON
                        TaskQueue.payload.like(f'%"image_id": {dedupe_key}}}%')  # End of JSON
                    )
                )
                .limit(1)
            )
            if existing.scalar_one_or_none():
                return None  # Already queued

        task = TaskQueue(
            task_type=task_type,
            payload=json.dumps(payload),
            priority=priority,
            status=TaskStatus.pending
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)
        return task

    if db:
        return await _do_enqueue(db)
    else:
        async with AsyncSessionLocal() as session:
            return await _do_enqueue(session)


async def clear_duplicate_tasks(db: AsyncSession = None) -> int:
    """Remove duplicate pending tasks, keeping only the oldest one per image_id.

    Returns number of duplicates removed.
    """
    async def _do_clear(session: AsyncSession) -> int:
        # Get all pending tag tasks
        result = await session.execute(
            select(TaskQueue)
            .where(
                TaskQueue.task_type == TaskType.tag,
                TaskQueue.status == TaskStatus.pending
            )
            .order_by(TaskQueue.created_at)
        )
        tasks = result.scalars().all()

        seen_images = set()
        duplicates = []

        for task in tasks:
            try:
                payload = json.loads(task.payload)
                image_id = payload.get('image_id')
                if image_id in seen_images:
                    duplicates.append(task.id)
                else:
                    seen_images.add(image_id)
            except:
                pass

        if duplicates:
            await session.execute(
                update(TaskQueue)
                .where(TaskQueue.id.in_(duplicates))
                .values(status=TaskStatus.failed, error_message="Duplicate task removed")
            )
            await session.commit()

        return len(duplicates)

    if db:
        return await _do_clear(db)
    else:
        async with AsyncSessionLocal() as session:
            return await _do_clear(session)
