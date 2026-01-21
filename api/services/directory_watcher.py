"""
Filesystem watcher service - monitors watch directories for new files
"""
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import WatchDirectory
from ..database import AsyncSessionLocal


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov'}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


class DirectoryEventHandler(FileSystemEventHandler):
    """Handle filesystem events for a watched directory"""

    def __init__(self, directory_id: int, auto_tag: bool, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.directory_id = directory_id
        self.auto_tag = auto_tag
        self.loop = loop
        self._pending_files: Set[str] = set()
        self._debounce_delay = 1.0  # Wait 1 second for file to finish writing

    def _is_supported_file(self, path: str) -> bool:
        """Check if file has a supported extension"""
        ext = Path(path).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    def _schedule_import(self, file_path: str):
        """Schedule file import with debouncing"""
        if file_path in self._pending_files:
            return

        self._pending_files.add(file_path)
        asyncio.run_coroutine_threadsafe(
            self._debounced_import(file_path),
            self.loop
        )

    async def _debounced_import(self, file_path: str):
        """Wait for file to finish writing, then import"""
        await asyncio.sleep(self._debounce_delay)

        # Check file still exists and is stable
        try:
            if not os.path.exists(file_path):
                self._pending_files.discard(file_path)
                return

            # Check file size is stable (not still being written)
            size1 = os.path.getsize(file_path)
            await asyncio.sleep(0.5)
            size2 = os.path.getsize(file_path)

            if size1 != size2:
                # File still being written, reschedule
                self._pending_files.discard(file_path)
                self._schedule_import(file_path)
                return

            # Import the file
            await self._import_file(file_path)

        except Exception as e:
            print(f"[Watcher] Error importing {file_path}: {e}")
        finally:
            self._pending_files.discard(file_path)

    async def _import_file(self, file_path: str):
        """Import a file into the library"""
        from .importer import import_image

        async with AsyncSessionLocal() as db:
            try:
                result = await import_image(
                    file_path,
                    db,
                    watch_directory_id=self.directory_id,
                    auto_tag=self.auto_tag
                )
                if result['status'] == 'imported':
                    print(f"[Watcher] Imported: {Path(file_path).name}")
                elif result['status'] == 'duplicate':
                    print(f"[Watcher] Duplicate: {Path(file_path).name}")
            except Exception as e:
                print(f"[Watcher] Failed to import {file_path}: {e}")

    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        if self._is_supported_file(event.src_path):
            self._schedule_import(event.src_path)

    def on_moved(self, event):
        """Handle file move/rename events (file moved INTO directory)"""
        if event.is_directory:
            return
        if self._is_supported_file(event.dest_path):
            self._schedule_import(event.dest_path)


class DirectoryWatcher:
    """Manages filesystem watchers for all watch directories"""

    def __init__(self):
        self.observer: Observer = None
        self.watches: Dict[int, object] = {}  # directory_id -> watch handle
        self.handlers: Dict[int, DirectoryEventHandler] = {}
        self.loop: asyncio.AbstractEventLoop = None
        self._running = False

    async def start(self):
        """Start watching all enabled directories"""
        self.loop = asyncio.get_event_loop()
        self.observer = Observer()
        self.observer.start()
        self._running = True

        # Load and watch all enabled directories
        await self._refresh_watches()
        print("[Watcher] Directory watcher started")

        # Scan for files added while app was closed
        asyncio.create_task(self._startup_scan())

    async def stop(self):
        """Stop all watchers"""
        self._running = False
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        self.watches.clear()
        self.handlers.clear()
        print("[Watcher] Directory watcher stopped")

    async def _refresh_watches(self):
        """Refresh watches from database"""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(WatchDirectory).where(WatchDirectory.enabled == True)
            )
            directories = result.scalars().all()

            current_ids = set(self.watches.keys())
            new_ids = {d.id for d in directories}

            # Remove watches for disabled/deleted directories
            for dir_id in current_ids - new_ids:
                self._remove_watch(dir_id)

            # Add watches for new directories
            for directory in directories:
                if directory.id not in self.watches:
                    self._add_watch(directory)

    async def _startup_scan(self):
        """Scan watched directories for files added while app was closed"""
        from .importer import import_image

        print("[Watcher] Starting startup scan for new files...")

        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(WatchDirectory).where(WatchDirectory.enabled == True)
                )
                directories = result.scalars().all()

                total_imported = 0
                total_duplicates = 0

                for directory in directories:
                    path = Path(directory.path)
                    if not path.exists():
                        print(f"[Watcher] Skipping non-existent path: {directory.path}")
                        continue

                    # Get last scan time (or use epoch if never scanned)
                    last_scan = directory.last_scanned_at
                    if last_scan and last_scan.tzinfo is None:
                        last_scan = last_scan.replace(tzinfo=timezone.utc)

                    print(f"[Watcher] Checking {directory.name or directory.path}, last scanned: {last_scan}")

                    # Find files to scan
                    files_to_check = []
                    if directory.recursive:
                        pattern = '**/*'
                    else:
                        pattern = '*'

                    for file_path in path.glob(pattern):
                        if not file_path.is_file():
                            continue
                        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            continue

                        # Check if file was modified after last scan
                        try:
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                            if last_scan is None or mtime > last_scan:
                                files_to_check.append(file_path)
                        except OSError:
                            continue

                    if files_to_check:
                        print(f"[Watcher] Found {len(files_to_check)} new files in {directory.name or directory.path}")

                        for file_path in files_to_check:
                            # Use a separate session for each import to avoid cascade failures
                            try:
                                async with AsyncSessionLocal() as import_db:
                                    import_result = await import_image(
                                        str(file_path),
                                        import_db,
                                        watch_directory_id=directory.id,
                                        auto_tag=directory.auto_tag
                                    )
                                    if import_result['status'] == 'imported':
                                        total_imported += 1
                                    elif import_result['status'] == 'duplicate':
                                        total_duplicates += 1
                            except Exception as e:
                                print(f"[Watcher] Error importing {file_path.name}: {e}")

                    # Update last_scanned_at
                    directory.last_scanned_at = datetime.now(timezone.utc)

                await db.commit()

            if total_imported > 0 or total_duplicates > 0:
                print(f"[Watcher] Startup scan complete: {total_imported} imported, {total_duplicates} duplicates")
            else:
                print("[Watcher] Startup scan complete: no new files found")

        except Exception as e:
            print(f"[Watcher] Startup scan FAILED: {e}")
            import traceback
            traceback.print_exc()

    def _add_watch(self, directory: WatchDirectory):
        """Add a watch for a directory"""
        path = Path(directory.path)
        if not path.exists():
            print(f"[Watcher] Skipping non-existent path: {directory.path}")
            return

        handler = DirectoryEventHandler(
            directory_id=directory.id,
            auto_tag=directory.auto_tag,
            loop=self.loop
        )

        try:
            watch = self.observer.schedule(
                handler,
                str(path),
                recursive=directory.recursive
            )
            self.watches[directory.id] = watch
            self.handlers[directory.id] = handler
            print(f"[Watcher] Watching: {directory.path} (recursive={directory.recursive})")
        except Exception as e:
            print(f"[Watcher] Failed to watch {directory.path}: {e}")

    def _remove_watch(self, directory_id: int):
        """Remove a watch for a directory"""
        if directory_id in self.watches:
            try:
                self.observer.unschedule(self.watches[directory_id])
            except Exception:
                pass
            del self.watches[directory_id]
            del self.handlers[directory_id]
            print(f"[Watcher] Stopped watching directory {directory_id}")

    async def add_directory(self, directory: WatchDirectory):
        """Add a new directory to watch"""
        if not self._running:
            return
        self._add_watch(directory)

    async def remove_directory(self, directory_id: int):
        """Stop watching a directory"""
        self._remove_watch(directory_id)

    async def refresh(self):
        """Refresh watches from database (call after directory changes)"""
        if self._running:
            await self._refresh_watches()


# Global watcher instance
directory_watcher = DirectoryWatcher()
