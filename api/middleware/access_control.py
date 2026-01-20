"""
Access control middleware for LocalBooru network security.

Controls access based on client IP address:
- localhost: Full access (read + write)
- local_network: Full access (read + write) - same WiFi/LAN
- public: Read-only (public internet IPs)

Settings/admin endpoints are always localhost-only.

Authentication enforcement based on auth_required_level setting:
- none: No auth required for any access level
- public: Auth required only for public internet access
- local_network: Auth required for local_network AND public access
- always: Auth required for ALL access including localhost
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Callable, Optional

from ..services.network import classify_ip
from ..services.auth import decode_token
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

# Endpoints exempt from authentication (chicken-egg problem for login)
AUTH_EXEMPT_ENDPOINTS = [
    "/api/users/login",
    "/api/users/verify",
    "/api/network/verify-handshake",
]

# Endpoints under localhost-only prefixes that should still be accessible from network
LOCALHOST_EXEMPTIONS = [
    "/api/network/verify-handshake",
]

# Auth level hierarchy - which access levels require auth at each setting
# Key: auth_required_level setting value
# Value: set of access levels that require authentication
AUTH_LEVEL_REQUIREMENTS = {
    "none": set(),  # No auth required for anyone
    "public": {"public"},  # Only public internet needs auth
    "local_network": {"local_network", "public"},  # LAN and public need auth
    "always": {"localhost", "local_network", "public"},  # Everyone needs auth
}


def get_client_ip(request: Request) -> str:
    """
    Extract the client IP from the request.

    Security: We intentionally ignore X-Forwarded-For and X-Real-IP headers
    because this app is designed for direct access, not behind a reverse proxy.
    Trusting these headers would allow attackers to spoof their IP as localhost.
    """
    # Use only the direct connection IP - never trust proxy headers
    if request.client:
        return request.client.host

    return "unknown"  # Don't default to localhost - that would grant access


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


def extract_bearer_token(request: Request) -> Optional[str]:
    """
    Extract the Bearer token from the Authorization header.

    Returns the token string if present and properly formatted, None otherwise.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    # Expected format: "Bearer <token>"
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    return parts[1]


def is_auth_exempt(path: str) -> bool:
    """Check if the endpoint is exempt from authentication requirements."""
    # Check exact matches for auth-exempt endpoints
    for exempt_path in AUTH_EXEMPT_ENDPOINTS:
        if path == exempt_path or path.startswith(exempt_path + "/"):
            return True
    return False


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
        request.state.user = None  # Will be set if authenticated

        # Debug logging for non-localhost access
        if access_level != "localhost":
            print(f"[AccessControl] {method} {path} from {client_ip} ({access_level})")

        # Get network settings (needed for both auth and access level checks)
        network_settings = get_network_settings()

        # Check authentication requirements based on auth_required_level setting
        auth_required_level = network_settings.get("auth_required_level", "none")
        access_levels_requiring_auth = AUTH_LEVEL_REQUIREMENTS.get(auth_required_level, set())

        # Determine if authentication is required for this request
        auth_required = access_level in access_levels_requiring_auth

        # Check for auth-exempt endpoints (login, verify, etc.)
        if auth_required and is_auth_exempt(path):
            auth_required = False
            print(f"[AccessControl] Auth exempt endpoint: {path}")

        # Process authentication if required or if token is provided
        token = extract_bearer_token(request)
        if token:
            user_payload = decode_token(token)
            if user_payload:
                request.state.user = user_payload
                print(f"[AccessControl] Authenticated user: {user_payload.get('username')}")
            elif auth_required:
                # Token provided but invalid/expired
                return cors_response(
                    401,
                    {
                        "error": "Invalid or expired token",
                        "detail": "The provided authentication token is invalid or has expired. Please log in again."
                    }
                )

        # If auth is required but no valid token, reject the request
        if auth_required and request.state.user is None:
            print(f"[AccessControl] Auth required but no valid token for {method} {path} from {access_level}")
            return cors_response(
                401,
                {
                    "error": "Authentication required",
                    "detail": f"This endpoint requires authentication for {access_level} access. Please provide a valid Bearer token in the Authorization header."
                }
            )

        # Localhost always has full access (after auth check)
        if access_level == "localhost":
            return await call_next(request)

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

        # Block localhost-only endpoints for non-localhost (with exemptions)
        for prefix in LOCALHOST_ONLY_PREFIXES:
            if path.startswith(prefix):
                # Check if this specific path is exempted from localhost restriction
                is_exempted = any(path == exempt or path.startswith(exempt + "/")
                                  for exempt in LOCALHOST_EXEMPTIONS)
                if not is_exempted:
                    return cors_response(
                        403,
                        {
                            "error": "This endpoint is only accessible from localhost",
                            "detail": f"Access to {prefix} requires direct access to the machine running LocalBooru"
                        }
                    )

        # Block write operations for public internet (local network gets full access)
        if method in WRITE_METHODS and access_level == "public":
            return cors_response(
                403,
                {
                    "error": "Write operations require local access",
                    "detail": "Public internet access is read-only. Modifications must be made from localhost or local network."
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
