"""
Access control middleware for LocalBooru network security.

Controls access based on client IP address:
- localhost: Full access (read + write)
- local_network: Read-only by default (if enabled)
- public: Read-only by default (if enabled)

Write operations (POST/PUT/PATCH/DELETE) are blocked for non-localhost
unless the user has explicit can_write permission.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Callable

from ..services.network import classify_ip
from ..routers.settings import get_network_settings


# Endpoints that are always localhost-only (sensitive settings)
LOCALHOST_ONLY_PREFIXES = [
    "/api/settings",
    "/api/network",
    "/api/users",
]

# Read-only endpoints safe for remote access
READ_ONLY_ENDPOINTS = [
    "GET",  # All GET requests are read-only
]

# Endpoints that require write access
WRITE_METHODS = ["POST", "PUT", "PATCH", "DELETE"]

# Endpoints exempt from access control (health checks, static files)
EXEMPT_PREFIXES = [
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/assets",  # Static frontend assets
    "/icon.png",
]


def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP from the request.

    Checks X-Forwarded-For header first (for reverse proxy setups),
    then falls back to the direct connection IP.
    """
    # Check X-Forwarded-For header (reverse proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2, ...
        # The first one is the original client
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct connection
    if request.client:
        return request.client.host

    return "127.0.0.1"  # Default to localhost if we can't determine


def cors_response(status_code: int, content: dict) -> JSONResponse:
    """Create a JSON response with CORS headers for error responses."""
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


class AccessControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce network access control based on client IP.

    Sets request.state.access_level to one of: 'localhost', 'local_network', 'public'
    Sets request.state.client_ip to the client's IP address

    Blocks:
    - Non-localhost access to sensitive endpoints (/settings, /network, /users)
    - Write operations from non-localhost unless explicitly allowed
    - Access levels that aren't enabled in settings
    """

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        method = request.method

        # Log ALL incoming requests for debugging
        client = request.client.host if request.client else "unknown"
        print(f"[AccessControl] INCOMING: {method} {path} from {client}")

        try:
            # Always allow OPTIONS requests (CORS preflight)
            if method == "OPTIONS":
                return await call_next(request)

            # Skip access control for exempt endpoints
            for prefix in EXEMPT_PREFIXES:
                if path.startswith(prefix):
                    return await call_next(request)

            # Get client IP and classify
            client_ip = get_client_ip(request)
            access_level = classify_ip(client_ip)
        except Exception as e:
            print(f"[AccessControl] EXCEPTION in middleware: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Store in request state for use by route handlers
        request.state.client_ip = client_ip
        request.state.access_level = access_level

        # Debug logging for non-localhost access
        if access_level != "localhost":
            print(f"[AccessControl] {method} {path} from {client_ip} ({access_level})")

        # Localhost always has full access
        if access_level == "localhost":
            return await call_next(request)

        # Get network settings
        network_settings = get_network_settings()
        print(f"[AccessControl] Network settings: local_network_enabled={network_settings.get('local_network_enabled')}")

        # Check if this access level is enabled
        if access_level == "local_network":
            if not network_settings.get("local_network_enabled", False):
                return cors_response(
                    403,
                    {
                        "error": "Local network access is disabled",
                        "detail": "Enable local network access in settings to connect from LAN"
                    }
                )
        elif access_level == "public":
            if not network_settings.get("public_network_enabled", False):
                return cors_response(
                    403,
                    {
                        "error": "Public network access is disabled",
                        "detail": "Enable public network access in settings to connect from internet"
                    }
                )

        # Block localhost-only endpoints for non-localhost
        for prefix in LOCALHOST_ONLY_PREFIXES:
            if path.startswith(prefix):
                return cors_response(
                    403,
                    {
                        "error": "This endpoint is only accessible from localhost",
                        "detail": f"Access to {prefix} requires direct access to the machine running LocalBooru"
                    }
                )

        # Block write operations for non-localhost
        if method in WRITE_METHODS:
            # TODO: Check user authentication and can_write permission
            # For now, all writes from non-localhost are blocked
            return cors_response(
                403,
                {
                    "error": "Write operations require localhost access",
                    "detail": "Remote access is read-only. Modifications must be made from the host machine."
                }
            )

        # Allow the request
        return await call_next(request)


def check_localhost_only(request: Request) -> bool:
    """
    Dependency to ensure request is from localhost.

    Usage in routes:
        @router.post("/sensitive-action")
        async def sensitive_action(
            _: bool = Depends(check_localhost_only)
        ):
            ...
    """
    access_level = getattr(request.state, "access_level", None)
    if access_level != "localhost":
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="This action is only available from localhost"
        )
    return True
