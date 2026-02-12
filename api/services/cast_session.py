"""
Cast session orchestration — one active session at a time.

Manages Chromecast and DLNA protocol backends, media server lifecycle,
and format compatibility checking.
"""
import asyncio
import logging
import time
import uuid as uuid_mod
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Protocol
# =============================================================================

@runtime_checkable
class CastBackend(Protocol):
    """Interface for cast protocol backends (Chromecast, DLNA)."""

    async def connect(self, **kwargs) -> None:
        """Connect to the cast device."""
        ...

    async def play(
        self,
        url: str,
        content_type: str,
        title: Optional[str] = None,
        subtitle_url: Optional[str] = None,
    ) -> None:
        """Start playing media on the device."""
        ...

    async def pause(self) -> None:
        """Pause playback."""
        ...

    async def resume(self) -> None:
        """Resume playback."""
        ...

    async def seek(self, position: float) -> None:
        """Seek to a position in seconds."""
        ...

    async def stop(self) -> None:
        """Stop playback."""
        ...

    async def set_volume(self, level: float) -> None:
        """Set volume level (0.0 - 1.0)."""
        ...

    async def get_status(self) -> dict:
        """Get current playback status."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the device."""
        ...


# =============================================================================
# Chromecast Backend
# =============================================================================

class ChromecastBackend:
    """Cast backend using pychromecast for Google Cast devices."""

    def __init__(self):
        self._cast = None
        self._mc = None  # MediaController
        self._status_listener = None
        self._connected = False

    async def connect(self, host: str, port: int, **kwargs) -> None:
        """Connect to a Chromecast device by host and port.

        Uses asyncio.to_thread because pychromecast is blocking.
        """
        import pychromecast

        def _blocking_connect():
            # pychromecast.get_chromecast_from_host expects a tuple:
            # (host, port, uuid, model_name, friendly_name)
            friendly_name = kwargs.get("friendly_name", "")
            device_uuid = kwargs.get("device_uuid")
            if isinstance(device_uuid, str):
                device_uuid = UUID(device_uuid)
            cast = pychromecast.get_chromecast_from_host(
                (host, port, device_uuid, None, friendly_name),
                tries=2,
                timeout=10,
            )
            cast.wait(timeout=10)
            return cast

        self._cast = await asyncio.to_thread(_blocking_connect)
        self._mc = self._cast.media_controller
        self._connected = True

        # Install status listener for live updates
        # Capture the event loop now (we're on the main async thread)
        loop = asyncio.get_running_loop()
        self._status_listener = _ChromecastStatusListener(self, loop)
        self._mc.register_status_listener(self._status_listener)

        logger.info(f"[Cast/Chromecast] Connected to {host}:{port}")

    async def play(
        self,
        url: str,
        content_type: str,
        title: Optional[str] = None,
        subtitle_url: Optional[str] = None,
    ) -> None:
        if not self._mc:
            raise RuntimeError("Not connected to Chromecast")

        def _blocking_play():
            # BUFFERED = seekable file with known duration
            # LIVE = live stream (default in pychromecast, but wrong for local files)
            self._mc.play_media(
                url,
                content_type,
                title=title or "LocalBooru Cast",
                subtitles=subtitle_url,
                subtitles_lang="en-US",
                subtitles_mime="text/vtt",
                autoplay=True,
                stream_type="BUFFERED",
            )
            self._mc.block_until_active(timeout=15)
            # Subtitles must be explicitly enabled after media is active
            if subtitle_url:
                self._mc.enable_subtitle(1)

        await asyncio.to_thread(_blocking_play)
        logger.info(f"[Cast/Chromecast] Playing: {title or url}")

    async def pause(self) -> None:
        if not self._mc:
            return
        await asyncio.to_thread(self._mc.pause)

    async def resume(self) -> None:
        if not self._mc:
            return
        await asyncio.to_thread(self._mc.play)

    async def seek(self, position: float) -> None:
        if not self._mc:
            return
        await asyncio.to_thread(self._mc.seek, position)

    async def stop(self) -> None:
        if not self._mc:
            return
        await asyncio.to_thread(self._mc.stop)

    async def set_volume(self, level: float) -> None:
        if not self._cast:
            return
        clamped = max(0.0, min(1.0, level))
        await asyncio.to_thread(self._cast.set_volume, clamped)

    async def get_status(self) -> dict:
        if not self._mc:
            return {"state": "idle", "current_time": 0, "duration": 0, "volume": 0, "title": None}

        ms = self._mc.status
        cast_status = self._cast.status if self._cast else None

        # Map pychromecast player_state to our simplified states
        state_map = {
            "PLAYING": "playing",
            "PAUSED": "paused",
            "IDLE": "idle",
            "BUFFERING": "buffering",
        }
        player_state = ms.player_state if ms else "IDLE"
        state = state_map.get(player_state, "idle")

        return {
            "state": state,
            "current_time": ms.current_time if ms and ms.current_time else 0,
            "duration": ms.duration if ms and ms.duration else 0,
            "volume": cast_status.volume_level if cast_status else 0,
            "title": ms.title if ms else None,
        }

    async def disconnect(self) -> None:
        if self._cast:
            try:
                await asyncio.to_thread(self._cast.disconnect, timeout=5)
            except Exception as e:
                logger.warning(f"[Cast/Chromecast] Error during disconnect: {e}")
            self._cast = None
            self._mc = None
        self._connected = False
        logger.info("[Cast/Chromecast] Disconnected")


class _ChromecastStatusListener:
    """Listener that broadcasts Chromecast media status changes via SSE.

    pychromecast fires callbacks from a background thread, so we use
    call_soon_threadsafe() to schedule async broadcasts on the event loop.
    """

    def __init__(self, backend: ChromecastBackend, loop: asyncio.AbstractEventLoop):
        self._backend = backend
        self._loop = loop

    def new_media_status(self, status):
        """Called by pychromecast when media status changes."""
        try:
            from .events import cast_events, CastEventType

            state_map = {
                "PLAYING": "playing",
                "PAUSED": "paused",
                "IDLE": "idle",
                "BUFFERING": "buffering",
            }
            player_state = status.player_state if status else "IDLE"
            state = state_map.get(player_state, "idle")

            data = {
                "state": state,
                "current_time": status.current_time if status and status.current_time else 0,
                "duration": status.duration if status and status.duration else 0,
                "volume": 0,
                "title": status.title if status else None,
            }

            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                cast_events.broadcast(CastEventType.STATUS, data),
            )
        except Exception as e:
            logger.debug(f"[Cast/Chromecast] Status listener error: {e}")

    def load_media_failed(self, queue_item_id, error_code):
        """Called by pychromecast when media loading fails."""
        logger.error(f"[Cast/Chromecast] Media load failed: item={queue_item_id}, error={error_code}")
        try:
            from .events import cast_events, CastEventType
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                cast_events.broadcast(CastEventType.ERROR, {
                    "error": f"Media load failed (code {error_code})",
                }),
            )
        except Exception as e:
            logger.debug(f"[Cast/Chromecast] Error broadcasting load failure: {e}")


# =============================================================================
# DLNA Backend
# =============================================================================

class DLNABackend:
    """Cast backend using async_upnp_client for DLNA/UPnP renderers."""

    def __init__(self):
        self._device = None  # DmrDevice
        self._factory = None  # UpnpFactory
        self._poll_task: Optional[asyncio.Task] = None
        self._connected = False

    async def connect(self, device_url: str, **kwargs) -> None:
        """Connect to a DLNA renderer by its description URL."""
        from async_upnp_client.aiohttp import AiohttpRequester
        from async_upnp_client.client_factory import UpnpFactory
        from async_upnp_client.profiles.dlna import DmrDevice

        requester = AiohttpRequester()
        self._factory = UpnpFactory(requester)
        device = await self._factory.async_create_device(device_url)
        self._device = DmrDevice(device, event_handler=None)
        self._connected = True

        # Start polling for status updates
        self._poll_task = asyncio.create_task(self._poll_status_loop())

        logger.info(f"[Cast/DLNA] Connected to {device_url}")

    async def play(
        self,
        url: str,
        content_type: str,
        title: Optional[str] = None,
        subtitle_url: Optional[str] = None,
    ) -> None:
        if not self._device:
            raise RuntimeError("Not connected to DLNA device")

        # Build DIDL-Lite metadata for the media item
        didl_title = title or "LocalBooru Cast"
        metadata = (
            '<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            '<item id="0" parentID="-1" restricted="1">'
            f'<dc:title>{_xml_escape(didl_title)}</dc:title>'
            '<upnp:class>object.item.videoItem</upnp:class>'
            f'<res protocolInfo="http-get:*:{content_type}:*">{_xml_escape(url)}</res>'
        )
        if subtitle_url:
            metadata += (
                f'<res protocolInfo="http-get:*:text/vtt:*">'
                f'{_xml_escape(subtitle_url)}</res>'
            )
        metadata += '</item></DIDL-Lite>'

        await self._device.async_set_transport_uri(url, didl_title, metadata)
        await self._device.async_play()

        logger.info(f"[Cast/DLNA] Playing: {didl_title}")

    async def pause(self) -> None:
        if not self._device:
            return
        await self._device.async_pause()

    async def resume(self) -> None:
        if not self._device:
            return
        await self._device.async_play()

    async def seek(self, position: float) -> None:
        if not self._device:
            return
        # Format as HH:MM:SS for UPnP seek target
        hours = int(position // 3600)
        minutes = int((position % 3600) // 60)
        seconds = int(position % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        await self._device.async_seek_rel_time(time_str)

    async def stop(self) -> None:
        if not self._device:
            return
        await self._device.async_stop()

    async def set_volume(self, level: float) -> None:
        if not self._device:
            return
        clamped = max(0.0, min(1.0, level))
        await self._device.async_set_volume_level(clamped)

    async def get_status(self) -> dict:
        if not self._device:
            return {"state": "idle", "current_time": 0, "duration": 0, "volume": 0, "title": None}

        # Map DLNA transport states to our states
        transport_state = (self._device.transport_state or "").upper()
        state_map = {
            "PLAYING": "playing",
            "PAUSED_PLAYBACK": "paused",
            "STOPPED": "idle",
            "TRANSITIONING": "buffering",
            "NO_MEDIA_PRESENT": "idle",
        }
        state = state_map.get(transport_state, "idle")

        current_time = 0
        if self._device.media_position is not None:
            current_time = self._device.media_position

        duration = 0
        if self._device.media_duration is not None:
            duration = self._device.media_duration

        volume = 0
        if self._device.volume_level is not None:
            volume = self._device.volume_level

        title = self._device.media_title

        return {
            "state": state,
            "current_time": current_time,
            "duration": duration,
            "volume": volume,
            "title": title,
        }

    async def disconnect(self) -> None:
        # Cancel polling task
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        self._device = None
        self._factory = None
        self._connected = False
        logger.info("[Cast/DLNA] Disconnected")

    async def _poll_status_loop(self) -> None:
        """Poll DLNA device for status updates and broadcast changes."""
        from .events import cast_events, CastEventType

        last_state = None
        while self._connected and self._device:
            try:
                await self._device.async_update()
                status = await self.get_status()

                # Only broadcast on state changes or periodically for position updates
                if status != last_state:
                    await cast_events.broadcast(CastEventType.STATUS, status)
                    last_state = status

                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Cast/DLNA] Poll error: {e}")
                await asyncio.sleep(2.0)


def _xml_escape(text: str) -> str:
    """Escape special XML characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# =============================================================================
# Cast Session Orchestrator
# =============================================================================

# Module-level singleton
_active_session: Optional["CastSession"] = None


class CastSession:
    """Orchestrates a single cast session: media server, backend, and playback."""

    def __init__(self):
        self.backend: Optional[CastBackend] = None
        self.device_id: Optional[str] = None
        self.device_info: Optional[dict] = None
        self.file_path: Optional[str] = None
        self.media_url: Optional[str] = None
        self.subtitle_url: Optional[str] = None
        self.content_type: Optional[str] = None
        self._transcode_stream = None
        self._media_id: Optional[str] = None  # ID registered with cast media server
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    async def start(
        self,
        device_id: str,
        file_path: str,
        image_id: Optional[int] = None,
        directory_id: Optional[int] = None,
    ) -> dict:
        """Start casting a file to a device.

        Args:
            device_id: ID of the target cast device (from discovery).
            file_path: Absolute path to the media file.
            image_id: Optional database image ID (for metadata lookup).
            directory_id: Optional directory ID for context.

        Returns:
            Dict with session info or raises on error.
        """
        from .events import cast_events, CastEventType
        from .cast_media_server import (
            start_server,
            register_media,
            get_media_url,
            is_chromecast_compatible,
            is_running as media_server_running,
        )
        from .cast_discovery import get_devices
        from .transcode_stream import TranscodeStream
        from ..routers.settings.models import get_cast_settings

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")

        # 1. Start cast media server if not already running
        if not media_server_running():
            settings = get_cast_settings()
            port = settings.get("cast_media_port", 8792)
            await start_server(port=port)
            logger.info(f"[CastSession] Started cast media server on port {port}")

        # 2. Check format compatibility and find subtitles
        compatible = is_chromecast_compatible(file_path)
        content_type = _guess_content_type(file_path)
        subtitle_vtt = self._find_subtitle_file(path)
        hls_dir = None

        if not compatible:
            # Start transcode stream for incompatible formats
            self._transcode_stream = TranscodeStream(
                video_path=file_path,
                target_bitrate="8M",
                target_resolution=(1920, 1080),
            )
            started = await self._transcode_stream.start()
            if not started:
                error = self._transcode_stream.error or "Failed to start transcode"
                raise RuntimeError(f"Transcode failed: {error}")

            # Wait for HLS playlist to be ready
            for _ in range(300):  # Up to 30 seconds
                if self._transcode_stream.playlist_ready:
                    break
                await asyncio.sleep(0.1)

            if not self._transcode_stream.playlist_ready:
                self._transcode_stream.stop()
                self._transcode_stream = None
                raise RuntimeError("Transcode stream failed to produce playlist in time")

            hls_dir = self._transcode_stream.hls_dir

        # 3. Register media with cast media server (single registration)
        media_id = str(uuid_mod.uuid4())[:8]
        register_media(
            media_id=media_id,
            file_path=file_path,
            hls_dir=hls_dir,
            subtitle_paths=[subtitle_vtt] if subtitle_vtt else None,
        )
        self._media_id = media_id

        if compatible:
            self.media_url = get_media_url(media_id, "file")
            self.content_type = content_type
            logger.info(f"[CastSession] Direct cast (compatible): {path.name}")
        else:
            self.media_url = get_media_url(media_id, "hls")
            self.content_type = "application/x-mpegURL"
            logger.info(f"[CastSession] Transcoded cast (HLS): {path.name}")

        if subtitle_vtt:
            self.subtitle_url = get_media_url(media_id, "subs") + subtitle_vtt.name
            logger.info(f"[CastSession] Found subtitle: {subtitle_vtt.name}")

        # 4. Look up device from discovery and create appropriate backend
        devices = await get_devices()
        device = None
        for d in devices:
            if d.id == device_id:
                device = d
                break

        if not device:
            raise ValueError(f"Device not found: {device_id}")

        self.device_id = device_id
        self.device_info = device.to_dict()
        self.file_path = file_path

        device_type = device.type.lower()
        if device_type == "chromecast":
            # Extract UUID from device id (format: "chromecast-{uuid}")
            device_uuid = device.id.replace("chromecast-", "", 1) if device.id.startswith("chromecast-") else None
            self.backend = ChromecastBackend()
            await self.backend.connect(
                host=device.ip,
                port=device.port or 8009,
                friendly_name=device.name,
                device_uuid=device_uuid,
            )
        elif device_type == "dlna":
            self.backend = DLNABackend()
            await self.backend.connect(
                device_url=device.location,
            )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

        # 5. Start playback
        title = path.stem
        if image_id is not None:
            title = f"LocalBooru #{image_id} - {path.stem}"

        await self.backend.play(
            url=self.media_url,
            content_type=self.content_type,
            title=title,
            subtitle_url=self.subtitle_url,
        )

        self._active = True

        # 6. Broadcast connected event
        await cast_events.broadcast(CastEventType.CONNECTED, {
            "device_id": device_id,
            "device_name": device.name,
            "device_type": device_type,
            "file": path.name,
            "title": title,
            "transcoding": self._transcode_stream is not None,
        })

        logger.info(f"[CastSession] Casting {path.name} to {device.name}")

        return {
            "device_id": device_id,
            "device_name": device.name,
            "file": path.name,
            "media_url": self.media_url,
            "subtitle_url": self.subtitle_url,
            "transcoding": self._transcode_stream is not None,
        }

    async def control(self, action: str, value=None) -> dict:
        """Dispatch a playback control action to the backend.

        Args:
            action: One of "pause", "resume", "seek", "volume".
            value: Required for "seek" (float seconds) and "volume" (float 0-1).

        Returns:
            Current status dict after the action.
        """
        if not self.backend or not self._active:
            raise RuntimeError("No active cast session")

        if action == "pause":
            await self.backend.pause()
        elif action == "resume":
            await self.backend.resume()
        elif action == "seek":
            if value is None:
                raise ValueError("Seek requires a position value")
            await self.backend.seek(float(value))
        elif action == "volume":
            if value is None:
                raise ValueError("Volume requires a level value")
            await self.backend.set_volume(float(value))
        else:
            raise ValueError(f"Unknown control action: {action}")

        # Return updated status
        return await self.backend.get_status()

    async def stop(self) -> None:
        """Stop the cast session and clean up all resources."""
        from .events import cast_events, CastEventType
        from .cast_media_server import unregister_media

        device_name = "Unknown"
        if self.device_info:
            device_name = self.device_info.get("name", "Unknown")

        # 1. Stop backend playback
        if self.backend:
            try:
                await self.backend.stop()
            except Exception as e:
                logger.warning(f"[CastSession] Error stopping playback: {e}")
            try:
                await self.backend.disconnect()
            except Exception as e:
                logger.warning(f"[CastSession] Error disconnecting backend: {e}")
            self.backend = None

        # 2. Unregister media from cast media server
        if self._media_id:
            try:
                unregister_media(self._media_id)
            except Exception as e:
                logger.warning(f"[CastSession] Error unregistering media {self._media_id}: {e}")
            self._media_id = None

        # 3. Stop transcode stream if active
        if self._transcode_stream:
            try:
                self._transcode_stream.stop()
            except Exception as e:
                logger.warning(f"[CastSession] Error stopping transcode: {e}")
            self._transcode_stream = None

        self._active = False
        self.media_url = None
        self.subtitle_url = None
        self.content_type = None
        self.device_id = None
        self.device_info = None
        self.file_path = None

        # 4. Broadcast disconnected event
        await cast_events.broadcast(CastEventType.DISCONNECTED, {
            "device_name": device_name,
        })

        logger.info(f"[CastSession] Stopped casting to {device_name}")

    async def get_status(self) -> dict:
        """Get current playback status from the backend.

        Returns:
            Dict with state, current_time, duration, volume, title,
            plus session-level metadata.
        """
        if not self.backend or not self._active:
            return {
                "active": False,
                "state": "idle",
                "current_time": 0,
                "duration": 0,
                "volume": 0,
                "title": None,
                "device_id": None,
                "device_name": None,
                "transcoding": False,
            }

        status = await self.backend.get_status()
        return {
            "active": True,
            **status,
            "device_id": self.device_id,
            "device_name": self.device_info.get("name", "Unknown") if self.device_info else None,
            "transcoding": self._transcode_stream is not None,
        }

    def _find_subtitle_file(self, video_path: Path) -> Optional[Path]:
        """Find a VTT subtitle file adjacent to the video file.

        Searches for *.vtt files in the same directory as the video,
        preferring files that share the video's stem (e.g., video.en.vtt).
        """
        parent = video_path.parent
        stem = video_path.stem

        # First look for files matching the video stem
        stem_matches = sorted(parent.glob(f"{stem}*.vtt"))
        if stem_matches:
            return stem_matches[0]

        # Fall back to any VTT file in the same directory
        any_vtt = sorted(parent.glob("*.vtt"))
        if any_vtt:
            return any_vtt[0]

        return None


def _guess_content_type(file_path: str) -> str:
    """Guess the MIME content type from a file extension."""
    ext = Path(file_path).suffix.lower()
    content_types = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".ts": "video/mp2t",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return content_types.get(ext, "application/octet-stream")


# =============================================================================
# Module-Level API
# =============================================================================

def get_session() -> Optional[CastSession]:
    """Get the current active cast session, or None."""
    global _active_session
    return _active_session


async def start_cast(
    device_id: str,
    file_path: str,
    image_id: Optional[int] = None,
    directory_id: Optional[int] = None,
) -> dict:
    """Start a new cast session, stopping any existing one first.

    Args:
        device_id: Target device ID from discovery.
        file_path: Absolute path to the media file to cast.
        image_id: Optional database image ID.
        directory_id: Optional directory ID.

    Returns:
        Session info dict from CastSession.start().
    """
    global _active_session

    # Stop any existing session first
    if _active_session and _active_session.active:
        logger.info("[CastSession] Stopping previous session before starting new one")
        await _active_session.stop()

    _active_session = CastSession()
    try:
        result = await _active_session.start(
            device_id=device_id,
            file_path=file_path,
            image_id=image_id,
            directory_id=directory_id,
        )
        return result
    except Exception:
        # Clean up on failure
        _active_session = None
        raise


async def stop_cast() -> None:
    """Stop the active cast session."""
    global _active_session
    if _active_session and _active_session.active:
        await _active_session.stop()
    _active_session = None


async def cast_control(action: str, value=None) -> dict:
    """Send a control action to the active cast session.

    Args:
        action: One of "pause", "resume", "seek", "volume".
        value: Required for "seek" and "volume".

    Returns:
        Current playback status dict.
    """
    if not _active_session or not _active_session.active:
        raise RuntimeError("No active cast session")
    return await _active_session.control(action, value)


async def get_cast_status() -> dict:
    """Get the current cast session status.

    Returns:
        Status dict with active flag, playback state, device info, etc.
    """
    if not _active_session:
        return {
            "active": False,
            "state": "idle",
            "current_time": 0,
            "duration": 0,
            "volume": 0,
            "title": None,
            "device_id": None,
            "device_name": None,
            "transcoding": False,
        }
    return await _active_session.get_status()
