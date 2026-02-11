"""
Network configuration router - manage local network and public access settings.

Most endpoints in this router are localhost-only (enforced by middleware).
Exception: /verify-handshake is accessible from the local network for mobile app trust verification.
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, Literal

from .settings import get_network_settings, save_network_settings, get_default_local_port
from ..services.network import (
    get_local_ip,
    get_all_local_ips,
    test_port_local,
    upnp_manager
)
from ..services.auth import get_server_fingerprint, create_handshake_nonce, verify_handshake_nonce

router = APIRouter()


class NetworkConfigUpdate(BaseModel):
    """Request body for updating network configuration"""
    local_network_enabled: Optional[bool] = None
    public_network_enabled: Optional[bool] = None
    local_port: Optional[int] = None
    public_port: Optional[int] = None
    auth_required_level: Optional[Literal["none", "public", "local_network", "always"]] = None
    upnp_enabled: Optional[bool] = None
    allow_settings_local_network: Optional[bool] = None  # Allow settings access from LAN


class PortTestRequest(BaseModel):
    """Request body for port testing"""
    port: int


class UPnPPortRequest(BaseModel):
    """Request body for UPnP port operations"""
    external_port: int
    internal_port: Optional[int] = None  # Defaults to external_port
    protocol: Optional[Literal["TCP", "UDP"]] = "TCP"
    description: Optional[str] = "LocalBooru"


class HandshakeVerifyRequest(BaseModel):
    """Request body for verifying handshake nonce"""
    nonce: str


@router.get("")
async def get_network_config():
    """
    Get current network configuration and status.

    Returns network settings, local IP addresses, and UPnP status.
    """
    settings = get_network_settings()
    local_ip = get_local_ip()
    all_ips = get_all_local_ips()

    # Build access URLs
    local_url = None
    public_url = None

    if local_ip and settings.get("local_network_enabled"):
        local_port = settings.get("local_port", get_default_local_port())
        local_url = f"http://{local_ip}:{local_port}"

    # Check UPnP status if enabled
    upnp_status = None
    if settings.get("upnp_enabled"):
        external_ip = upnp_manager.get_external_ip()
        if external_ip and settings.get("public_network_enabled"):
            public_port = settings.get("public_port", 8791)
            public_url = f"http://{external_ip}:{public_port}"
        upnp_status = {
            "external_ip": external_ip,
            "gateway_found": upnp_manager._gateway_found
        }

    return {
        "settings": settings,
        "local_ip": local_ip,
        "all_local_ips": all_ips,
        "local_url": local_url,
        "public_url": public_url,
        "upnp_status": upnp_status
    }


@router.get("/qr-data")
async def get_qr_data(request: Request):
    """
    Get data for QR code to connect mobile app.

    Returns server info including local and public URLs for the mobile app to try.
    Version 2 includes cert_fingerprint for HTTPS certificate pinning.
    """
    settings = get_network_settings()
    local_ip = get_local_ip()

    # Detect actual protocol from the incoming request — certificate may exist on disk
    # but uvicorn might not be configured with SSL (e.g. dev mode)
    has_https = request.url.scheme == "https"
    protocol = "https" if has_https else "http"

    # Build local URL (always include if we have an IP)
    local_url = None
    if local_ip:
        local_port = settings.get("local_port", get_default_local_port())
        local_url = f"{protocol}://{local_ip}:{local_port}"

    # Build public URL if UPnP is enabled and has external IP
    public_url = None
    if settings.get("upnp_enabled"):
        external_ip = upnp_manager.get_external_ip()
        if external_ip and settings.get("public_network_enabled"):
            public_port = settings.get("public_port", 8791)
            public_url = f"{protocol}://{external_ip}:{public_port}"

    # Check if auth is required
    auth_level = settings.get("auth_required_level", "none")
    auth_required = auth_level in ["local_network", "always"]

    # Generate handshake nonce for verification
    nonce, nonce_expires = create_handshake_nonce()

    # Get TLS certificate fingerprint for certificate pinning (only when actually serving HTTPS)
    cert_fingerprint = None
    if has_https:
        try:
            from ..services.certificate import get_certificate_fingerprint
            cert_fingerprint = get_certificate_fingerprint()
        except ImportError:
            pass  # cryptography not available

    return {
        "type": "localbooru",
        "version": 2,  # Bumped from 1 to indicate HTTPS support
        "name": "LocalBooru",
        "local": local_url,
        "public": public_url,
        "auth": auth_required,
        "fingerprint": get_server_fingerprint(),
        "cert_fingerprint": cert_fingerprint,  # NEW: TLS certificate fingerprint for pinning
        "nonce": nonce,
        "nonce_expires": int(nonce_expires)
    }


@router.post("")
async def update_network_config(config: NetworkConfigUpdate):
    """
    Update network configuration.

    Note: Changes to ports or enabled status may require a restart to take effect.
    """
    current = get_network_settings()

    # Update only provided fields
    if config.local_network_enabled is not None:
        current["local_network_enabled"] = config.local_network_enabled
    if config.public_network_enabled is not None:
        current["public_network_enabled"] = config.public_network_enabled
    if config.local_port is not None:
        current["local_port"] = config.local_port
    if config.public_port is not None:
        current["public_port"] = config.public_port
    if config.auth_required_level is not None:
        current["auth_required_level"] = config.auth_required_level
    if config.upnp_enabled is not None:
        current["upnp_enabled"] = config.upnp_enabled
    if config.allow_settings_local_network is not None:
        current["allow_settings_local_network"] = config.allow_settings_local_network

    save_network_settings(current)

    return {
        "success": True,
        "settings": current,
        "restart_required": True  # Always true for network changes
    }


@router.post("/test-port")
async def test_port(request: PortTestRequest):
    """
    Test if a port is available for binding.

    Returns whether the port can be used by LocalBooru.
    """
    result = test_port_local(request.port)
    return {
        "port": request.port,
        "available": result["available"],
        "error": result["error"]
    }


@router.post("/upnp/discover")
async def discover_upnp():
    """
    Discover UPnP gateway on the network.

    Required before opening ports via UPnP.
    """
    result = upnp_manager.discover()
    return result


@router.post("/upnp/open-port")
async def open_upnp_port(request: UPnPPortRequest):
    """
    Open a port on the router via UPnP.

    Maps external_port on router to internal_port on this machine.
    """
    internal = request.internal_port or request.external_port

    result = upnp_manager.add_port_mapping(
        external_port=request.external_port,
        internal_port=internal,
        protocol=request.protocol,
        description=request.description
    )

    return result


@router.delete("/upnp/close-port/{external_port}")
async def close_upnp_port(external_port: int, protocol: str = "TCP"):
    """
    Close a port mapping on the router via UPnP.
    """
    result = upnp_manager.remove_port_mapping(
        external_port=external_port,
        protocol=protocol
    )

    return result


@router.get("/upnp/mappings")
async def get_upnp_mappings():
    """
    Get all current UPnP port mappings on the router.
    """
    mappings = upnp_manager.get_port_mappings()
    return {"mappings": mappings}


@router.get("/upnp/external-ip")
async def get_external_ip():
    """
    Get the external (public) IP address via UPnP.
    """
    ip = upnp_manager.get_external_ip()
    return {"external_ip": ip}


@router.post("/verify-handshake")
async def verify_handshake(request: HandshakeVerifyRequest):
    """
    Verify a handshake nonce from QR code scanning.

    This endpoint is accessible from the local network (not localhost-only)
    to allow mobile apps to verify they're connecting to the correct server.

    The nonce is single-use and expires after 5 minutes.

    Returns:
        - On success: { "valid": true, "fingerprint": "...", "server_name": "LocalBooru" }
        - On failure: { "valid": false, "error": "..." } with 401 status
    """
    if verify_handshake_nonce(request.nonce):
        return {
            "valid": True,
            "fingerprint": get_server_fingerprint(),
            "server_name": "LocalBooru"
        }
    else:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={
                "valid": False,
                "error": "Invalid, expired, or already-used nonce"
            }
        )
