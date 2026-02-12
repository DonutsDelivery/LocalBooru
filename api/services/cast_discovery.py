"""
Unified Chromecast + DLNA device discovery.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CastDevice:
    id: str
    name: str
    type: str  # "chromecast" | "dlna"
    model: str
    ip: str
    port: int
    location: str = ""  # DLNA device description URL

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_discovered_devices: List[CastDevice] = []
_discovery_task: Optional[asyncio.Task] = None
_last_scan_time: float = 0.0
_scanning: bool = False

CACHE_TTL = 30  # seconds


# ---------------------------------------------------------------------------
# Chromecast discovery (pychromecast)
# ---------------------------------------------------------------------------

async def _discover_chromecasts() -> List[CastDevice]:
    """Discover Chromecast devices on the local network.

    Uses pychromecast which is blocking, so we run it in a thread.
    Returns an empty list if pychromecast is not installed.
    """
    try:
        import pychromecast
    except ImportError:
        logger.debug("[CastDiscovery] pychromecast not installed, skipping Chromecast scan")
        return []

    devices: List[CastDevice] = []
    try:
        chromecasts, browser = await asyncio.to_thread(
            pychromecast.get_chromecasts, timeout=5
        )
        for cc in chromecasts:
            devices.append(CastDevice(
                id=f"chromecast-{cc.uuid}",
                name=cc.name,
                type="chromecast",
                model=cc.model_name or "Chromecast",
                ip=cc.host,
                port=cc.port,
            ))
        # Stop the browser's background listener so it doesn't leak
        browser.stop_discovery()
        logger.info(f"[CastDiscovery] Found {len(devices)} Chromecast(s)")
    except Exception as exc:
        logger.warning(f"[CastDiscovery] Chromecast scan failed: {exc}")

    return devices


# ---------------------------------------------------------------------------
# DLNA / UPnP discovery (async_upnp_client)
# ---------------------------------------------------------------------------

async def _discover_dlna() -> List[CastDevice]:
    """Discover DLNA MediaRenderer devices via SSDP.

    Uses async_upnp_client for the SSDP search.
    Returns an empty list if async_upnp_client is not installed.
    """
    try:
        from async_upnp_client.search import async_search
        from async_upnp_client.aiohttp import AiohttpRequester
    except ImportError:
        logger.debug("[CastDiscovery] async_upnp_client not installed, skipping DLNA scan")
        return []

    devices: List[CastDevice] = []
    seen_ids: set = set()

    try:
        async def _on_response(response: dict) -> None:
            """Callback invoked for each SSDP response."""
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

                # Build a stable ID from USN (typically uuid:XXXXX::urn:...)
                device_id = usn.split("::")[0] if "::" in usn else usn
                if not device_id or device_id in seen_ids:
                    return
                seen_ids.add(device_id)

                # Derive a name — prefer the friendly name from the response,
                # fall back to the server string, then a generic label.
                name = friendly_name or server or f"DLNA Renderer ({ip})"

                # Try to pull a model hint from the SERVER header
                # (e.g. "Linux/3.x UPnP/1.0 LG-Smart-TV/1.0")
                model = ""
                if server:
                    parts = server.split()
                    # Use the last token that isn't UPnP/* or DLNADOC/*
                    for part in reversed(parts):
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
            except Exception as inner_exc:
                logger.debug(f"[CastDiscovery] Error parsing SSDP response: {inner_exc}")

        await async_search(
            search_target="urn:schemas-upnp-org:device:MediaRenderer:1",
            timeout=5,
            async_callback=_on_response,
        )
        logger.info(f"[CastDiscovery] Found {len(devices)} DLNA renderer(s)")
    except Exception as exc:
        logger.warning(f"[CastDiscovery] DLNA scan failed: {exc}")

    return devices


# ---------------------------------------------------------------------------
# Combined scan
# ---------------------------------------------------------------------------

async def _run_scan() -> List[CastDevice]:
    """Run both Chromecast and DLNA scans concurrently and merge results."""
    global _scanning
    _scanning = True
    try:
        chromecast_task = asyncio.create_task(_discover_chromecasts())
        dlna_task = asyncio.create_task(_discover_dlna())
        chromecast_devices, dlna_devices = await asyncio.gather(
            chromecast_task, dlna_task, return_exceptions=True
        )

        merged: List[CastDevice] = []
        if isinstance(chromecast_devices, list):
            merged.extend(chromecast_devices)
        else:
            logger.warning(f"[CastDiscovery] Chromecast scan returned exception: {chromecast_devices}")

        if isinstance(dlna_devices, list):
            merged.extend(dlna_devices)
        else:
            logger.warning(f"[CastDiscovery] DLNA scan returned exception: {dlna_devices}")

        return merged
    finally:
        _scanning = False


# ---------------------------------------------------------------------------
# Background discovery loop
# ---------------------------------------------------------------------------

async def _discovery_loop() -> None:
    """Background loop that refreshes device list every CACHE_TTL seconds."""
    global _discovered_devices, _last_scan_time
    logger.info("[CastDiscovery] Background discovery loop started")
    try:
        while True:
            devices = await _run_scan()
            _discovered_devices = devices
            _last_scan_time = time.monotonic()
            logger.debug(f"[CastDiscovery] Cached {len(devices)} device(s)")
            await asyncio.sleep(CACHE_TTL)
    except asyncio.CancelledError:
        logger.info("[CastDiscovery] Background discovery loop stopped")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_discovery() -> None:
    """Start the background discovery loop (idempotent)."""
    global _discovery_task
    if _discovery_task is not None and not _discovery_task.done():
        logger.debug("[CastDiscovery] Discovery already running")
        return
    _discovery_task = asyncio.create_task(_discovery_loop())
    logger.info("[CastDiscovery] Discovery started")


async def stop_discovery() -> None:
    """Cancel the background discovery loop."""
    global _discovery_task
    if _discovery_task is not None and not _discovery_task.done():
        _discovery_task.cancel()
        try:
            await _discovery_task
        except asyncio.CancelledError:
            pass
    _discovery_task = None
    logger.info("[CastDiscovery] Discovery stopped")


async def get_devices() -> List[CastDevice]:
    """Return the current cached device list.

    If the cache is stale (older than CACHE_TTL) and no scan is running,
    triggers a one-off refresh before returning.
    """
    global _discovered_devices, _last_scan_time
    age = time.monotonic() - _last_scan_time
    if age > CACHE_TTL and not _scanning:
        _discovered_devices = await _run_scan()
        _last_scan_time = time.monotonic()
    return list(_discovered_devices)


async def refresh_devices() -> List[CastDevice]:
    """Force an immediate rescan, update cache, and return results."""
    global _discovered_devices, _last_scan_time
    _discovered_devices = await _run_scan()
    _last_scan_time = time.monotonic()
    return list(_discovered_devices)
