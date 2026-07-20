"""
Cast Sidecar — Standalone FastAPI app.

Discovers and controls Chromecast and DLNA devices on the local network.
The Rust backend resolves media files to LAN-accessible URLs before calling /play,
so this sidecar only needs to talk to cast devices — no file system access needed.

Endpoints:
  GET  /health           → health check + backend availability
  GET  /devices          → list discovered devices
  POST /devices/refresh  → trigger re-discovery
  POST /play             → start casting to a device
  POST /control          → pause/resume/seek/volume
  POST /stop             → stop casting
  GET  /status           → current playback status
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("cast")

# ─── Backend detection ────────────────────────────────────────────────────────

HAS_PYCHROMECAST = False
HAS_UPNP = False

try:
    import pychromecast
    HAS_PYCHROMECAST = True
except ImportError:
    pass

try:
    from async_upnp_client.search import async_search
    HAS_UPNP = True
except ImportError:
    pass


# ─── Device model ─────────────────────────────────────────────────────────────

@dataclass
class CastDevice:
    id: str
    name: str
    type: str  # "chromecast" | "dlna"
    model: str
    ip: str
    port: int
    location: str = ""  # DLNA description URL

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Discovery ────────────────────────────────────────────────────────────────

_discovered_devices: List[CastDevice] = []
_last_scan_time: float = 0.0
_scanning: bool = False
_scan_event: Optional[asyncio.Event] = None
CACHE_TTL = 30


async def _discover_chromecasts() -> List[CastDevice]:
    if not HAS_PYCHROMECAST:
        return []

    devices = []
    try:
        chromecasts, browser = await asyncio.to_thread(
            pychromecast.get_chromecasts, timeout=10
        )
        for cc in chromecasts:
            ci = cc.cast_info
            devices.append(CastDevice(
                id=f"chromecast-{cc.uuid}",
                name=ci.friendly_name or cc.name,
                type="chromecast",
                model=ci.model_name or "Chromecast",
                ip=ci.host,
                port=ci.port,
            ))
        browser.stop_discovery()
        logger.info(f"Found {len(devices)} Chromecast(s)")
    except Exception as e:
        logger.warning(f"Chromecast scan failed: {e}")
    return devices


async def _discover_dlna() -> List[CastDevice]:
    if not HAS_UPNP:
        return []

    devices = []
    seen = set()

    try:
        from async_upnp_client.search import async_search

        async def _on_response(response: dict):
            try:
                usn = response.get("USN", "")
                location = response.get("LOCATION", "")
                server = response.get("SERVER", "")
                friendly_name = response.get("_FRIENDLY_NAME", "")

                if not location:
                    return

                parsed = urlparse(location)
                ip = parsed.hostname or ""
                port = parsed.port or 80

                device_id = usn.split("::")[0] if "::" in usn else usn
                if not device_id or device_id in seen:
                    return
                seen.add(device_id)

                name = friendly_name or server or f"DLNA Renderer ({ip})"

                model = ""
                if server:
                    for part in reversed(server.split()):
                        lower = part.lower()
                        if not lower.startswith("upnp/") and not lower.startswith("dlnadoc/"):
                            model = part
                            break
                model = model or "DLNA Renderer"

                devices.append(CastDevice(
                    id=f"dlna-{device_id}",
                    name=name,
                    type="dlna",
                    model=model,
                    ip=ip,
                    port=port,
                    location=location,
                ))
            except Exception as e:
                logger.debug(f"Error parsing SSDP response: {e}")

        await async_search(
            search_target="urn:schemas-upnp-org:device:MediaRenderer:1",
            timeout=5,
            async_callback=_on_response,
        )
        logger.info(f"Found {len(devices)} DLNA renderer(s)")
    except Exception as e:
        logger.warning(f"DLNA scan failed: {e}")
    return devices


async def _run_scan() -> List[CastDevice]:
    global _scanning, _scan_event
    _scanning = True
    _scan_event = asyncio.Event()
    try:
        cc_task = asyncio.create_task(_discover_chromecasts())
        dlna_task = asyncio.create_task(_discover_dlna())
        cc_result, dlna_result = await asyncio.gather(cc_task, dlna_task, return_exceptions=True)

        merged = []
        if isinstance(cc_result, list):
            merged.extend(cc_result)
        if isinstance(dlna_result, list):
            merged.extend(dlna_result)
        return merged
    finally:
        _scanning = False
        _scan_event.set()


async def get_devices() -> List[CastDevice]:
    global _discovered_devices, _last_scan_time
    if _scanning and _scan_event:
        try:
            await asyncio.wait_for(_scan_event.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass
    age = time.monotonic() - _last_scan_time
    if age > CACHE_TTL and not _scanning:
        _discovered_devices = await _run_scan()
        _last_scan_time = time.monotonic()
    return list(_discovered_devices)


async def refresh_devices() -> List[CastDevice]:
    global _discovered_devices, _last_scan_time
    _discovered_devices = await _run_scan()
    _last_scan_time = time.monotonic()
    return list(_discovered_devices)


# ─── Cast backends ────────────────────────────────────────────────────────────

class ChromecastBackend:
    def __init__(self):
        self._cast = None
        self._mc = None
        self._connected = False

    async def connect(self, host: str, port: int, **kwargs):
        from uuid import UUID

        def _blocking_connect():
            friendly_name = kwargs.get("friendly_name", "")
            device_uuid = kwargs.get("device_uuid")
            if isinstance(device_uuid, str):
                device_uuid = UUID(device_uuid)
            cast = pychromecast.get_chromecast_from_host(
                (host, port, device_uuid, None, friendly_name),
                tries=2, timeout=10,
            )
            cast.wait(timeout=10)
            return cast

        self._cast = await asyncio.to_thread(_blocking_connect)
        self._mc = self._cast.media_controller
        self._connected = True
        logger.info(f"[Chromecast] Connected to {host}:{port}")

    async def play(self, url: str, content_type: str, title: str = None, subtitle_url: str = None):
        if not self._mc:
            raise RuntimeError("Not connected")

        def _blocking_play():
            self._mc.play_media(
                url, content_type,
                title=title or "LocalBooru Cast",
                subtitles=subtitle_url,
                subtitles_lang="en-US",
                subtitles_mime="text/vtt",
                autoplay=True,
                stream_type="BUFFERED",
            )
            self._mc.block_until_active(timeout=30)
            if subtitle_url:
                self._mc.enable_subtitle(1)

        await asyncio.to_thread(_blocking_play)
        logger.info(f"[Chromecast] Playing: {title or url}")

    async def pause(self):
        if self._mc:
            await asyncio.to_thread(self._mc.pause)

    async def resume(self):
        if self._mc:
            await asyncio.to_thread(self._mc.play)

    async def seek(self, position: float):
        if self._mc:
            await asyncio.to_thread(self._mc.seek, position)

    async def stop(self):
        if self._mc:
            await asyncio.to_thread(self._mc.stop)

    async def set_volume(self, level: float):
        if self._cast:
            await asyncio.to_thread(self._cast.set_volume, max(0.0, min(1.0, level)))

    async def get_status(self) -> dict:
        if not self._mc:
            return {"state": "idle", "current_time": 0, "duration": 0, "volume": 0, "title": None}

        ms = self._mc.status
        cast_status = self._cast.status if self._cast else None
        state_map = {"PLAYING": "playing", "PAUSED": "paused", "IDLE": "idle", "BUFFERING": "buffering"}
        state = state_map.get(ms.player_state if ms else "IDLE", "idle")

        return {
            "state": state,
            "current_time": ms.current_time if ms and ms.current_time else 0,
            "duration": ms.duration if ms and ms.duration else 0,
            "volume": cast_status.volume_level if cast_status else 0,
            "title": ms.title if ms else None,
        }

    async def disconnect(self):
        if self._cast:
            try:
                await asyncio.to_thread(self._cast.disconnect, timeout=5)
            except Exception:
                pass
            self._cast = None
            self._mc = None
        self._connected = False


class DLNABackend:
    def __init__(self):
        self._device = None
        self._factory = None
        self._poll_task: Optional[asyncio.Task] = None
        self._connected = False

    async def connect(self, device_url: str, **kwargs):
        from async_upnp_client.aiohttp import AiohttpRequester
        from async_upnp_client.client_factory import UpnpFactory
        from async_upnp_client.profiles.dlna import DmrDevice

        requester = AiohttpRequester()
        self._factory = UpnpFactory(requester)
        device = await self._factory.async_create_device(device_url)
        self._device = DmrDevice(device, event_handler=None)
        self._connected = True
        logger.info(f"[DLNA] Connected to {device_url}")

    async def play(self, url: str, content_type: str, title: str = None, subtitle_url: str = None):
        if not self._device:
            raise RuntimeError("Not connected")

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
            metadata += f'<res protocolInfo="http-get:*:text/vtt:*">{_xml_escape(subtitle_url)}</res>'
        metadata += '</item></DIDL-Lite>'

        await self._device.async_set_transport_uri(url, didl_title, metadata)
        await self._device.async_play()
        logger.info(f"[DLNA] Playing: {didl_title}")

    async def pause(self):
        if self._device:
            await self._device.async_pause()

    async def resume(self):
        if self._device:
            await self._device.async_play()

    async def seek(self, position: float):
        if self._device:
            hours = int(position // 3600)
            minutes = int((position % 3600) // 60)
            seconds = int(position % 60)
            await self._device.async_seek_rel_time(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    async def stop(self):
        if self._device:
            await self._device.async_stop()

    async def set_volume(self, level: float):
        if self._device:
            await self._device.async_set_volume_level(max(0.0, min(1.0, level)))

    async def get_status(self) -> dict:
        if not self._device:
            return {"state": "idle", "current_time": 0, "duration": 0, "volume": 0, "title": None}

        try:
            await self._device.async_update()
        except Exception:
            pass

        transport_state = (self._device.transport_state or "").upper()
        state_map = {
            "PLAYING": "playing", "PAUSED_PLAYBACK": "paused",
            "STOPPED": "idle", "TRANSITIONING": "buffering",
            "NO_MEDIA_PRESENT": "idle",
        }
        return {
            "state": state_map.get(transport_state, "idle"),
            "current_time": self._device.media_position or 0,
            "duration": self._device.media_duration or 0,
            "volume": self._device.volume_level or 0,
            "title": self._device.media_title,
        }

    async def disconnect(self):
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._device = None
        self._factory = None
        self._connected = False


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
    )


# ─── Session orchestrator ────────────────────────────────────────────────────

_active_backend = None
_active_device_id: Optional[str] = None
_active_device_info: Optional[dict] = None


async def _stop_active():
    """Stop the active cast session if any."""
    global _active_backend, _active_device_id, _active_device_info
    if _active_backend:
        try:
            await _active_backend.stop()
        except Exception as e:
            logger.warning(f"Error stopping playback: {e}")
        try:
            await _active_backend.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")
        _active_backend = None
    _active_device_id = None
    _active_device_info = None


# ─── Content type guessing ───────────────────────────────────────────────────

_CONTENT_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".ts": "video/mp2t",
    ".mp3": "audio/mpeg", ".flac": "audio/flac", ".wav": "audio/wav",
    ".ogg": "audio/ogg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}


def _guess_content_type(url: str) -> str:
    # Try to extract extension from URL path
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    return _CONTENT_TYPES.get(ext, "video/mp4")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Cast Sidecar")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chromecast": HAS_PYCHROMECAST,
        "dlna": HAS_UPNP,
    }


@app.get("/devices")
async def list_devices():
    devices = await get_devices()
    return {"devices": [d.to_dict() for d in devices]}


@app.post("/devices/refresh")
async def refresh():
    devices = await refresh_devices()
    return {"devices": [d.to_dict() for d in devices]}


class PlayRequest(BaseModel):
    device_id: str
    media_url: str
    content_type: Optional[str] = None
    title: Optional[str] = None
    subtitle_url: Optional[str] = None
    image_id: Optional[int] = None


@app.post("/play")
async def play(req: PlayRequest):
    # Stop any active session
    await _stop_active()

    global _active_backend, _active_device_id, _active_device_info

    # Find the device
    devices = await get_devices()
    device = None
    for d in devices:
        if d.id == req.device_id:
            device = d
            break

    if not device:
        raise HTTPException(status_code=404, detail=f"Device not found: {req.device_id}")

    # Create appropriate backend
    if device.type == "chromecast":
        if not HAS_PYCHROMECAST:
            raise HTTPException(status_code=503, detail="pychromecast not installed")
        device_uuid = device.id.replace("chromecast-", "", 1) if device.id.startswith("chromecast-") else None
        backend = ChromecastBackend()
        await backend.connect(
            host=device.ip,
            port=device.port or 8009,
            friendly_name=device.name,
            device_uuid=device_uuid,
        )
    elif device.type == "dlna":
        if not HAS_UPNP:
            raise HTTPException(status_code=503, detail="async-upnp-client not installed")
        backend = DLNABackend()
        await backend.connect(device_url=device.location)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported device type: {device.type}")

    # Determine content type
    content_type = req.content_type or _guess_content_type(req.media_url)

    # Start playback
    title = req.title or f"LocalBooru #{req.image_id}" if req.image_id else "LocalBooru Cast"
    await backend.play(
        url=req.media_url,
        content_type=content_type,
        title=title,
        subtitle_url=req.subtitle_url,
    )

    _active_backend = backend
    _active_device_id = req.device_id
    _active_device_info = device.to_dict()

    return {
        "success": True,
        "device_id": device.id,
        "device_name": device.name,
        "device_type": device.type,
        "device_host": device.ip,
        "media_url": req.media_url,
    }


class ControlRequest(BaseModel):
    action: str
    value: Optional[float] = None


@app.post("/control")
async def control(req: ControlRequest):
    if not _active_backend:
        raise HTTPException(status_code=409, detail="No active cast session")

    if req.action == "pause":
        await _active_backend.pause()
    elif req.action in ("resume", "play"):
        await _active_backend.resume()
    elif req.action == "seek":
        if req.value is None:
            raise HTTPException(status_code=400, detail="Seek requires a value")
        await _active_backend.seek(req.value)
    elif req.action == "volume":
        if req.value is None:
            raise HTTPException(status_code=400, detail="Volume requires a value")
        await _active_backend.set_volume(req.value)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

    status = await _active_backend.get_status()
    return status


@app.post("/stop")
async def stop():
    await _stop_active()
    return {"success": True, "status": "idle"}


@app.get("/status")
async def status():
    if not _active_backend:
        return {
            "status": "idle",
            "active": False,
            "current_time": 0,
            "duration": 0,
            "volume": 0,
            "title": None,
            "device_id": None,
            "device_name": None,
        }

    backend_status = await _active_backend.get_status()
    return {
        "status": backend_status.get("state", "idle"),
        "active": True,
        "position": backend_status.get("current_time", 0),
        "current_time": backend_status.get("current_time", 0),
        "duration": backend_status.get("duration", 0),
        "volume": backend_status.get("volume", 0),
        "title": backend_status.get("title"),
        "device_id": _active_device_id,
        "device_name": _active_device_info.get("name") if _active_device_info else None,
    }
