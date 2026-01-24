"""
JWT authentication utilities for session management.

Provides token generation, validation, and secret key management.
Also handles nonce-based handshake verification for mobile app trust.
"""
import jwt
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
from ..config import get_data_dir


# JWT settings
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def _get_secret_key_path() -> Path:
    """Get path to the JWT secret key file in the data directory."""
    return get_data_dir() / "jwt_secret.key"


def _load_or_create_secret_key() -> str:
    """
    Load the JWT secret key from disk, or create one if it doesn't exist.

    The secret key is stored in the data directory to persist across restarts.
    """
    key_path = _get_secret_key_path()

    if key_path.exists():
        return key_path.read_text().strip()

    # Generate a new secure random key
    secret_key = secrets.token_hex(32)  # 256-bit key

    # Save to disk with restrictive permissions
    key_path.write_text(secret_key)

    return secret_key


# Load the secret key on module import
_SECRET_KEY: Optional[str] = None


def get_secret_key() -> str:
    """Get the JWT secret key, loading/creating it if necessary."""
    global _SECRET_KEY
    if _SECRET_KEY is None:
        _SECRET_KEY = _load_or_create_secret_key()
    return _SECRET_KEY


# Server fingerprint for trust verification
_SERVER_FINGERPRINT: Optional[str] = None


def _get_fingerprint_path() -> Path:
    """Get path to the server fingerprint file in the data directory."""
    return get_data_dir() / "server_fingerprint.key"


def _load_or_create_fingerprint() -> str:
    """
    Load the server fingerprint from disk, or create one if it doesn't exist.

    The fingerprint is a unique identifier for this server instance,
    used by mobile apps to verify they're connecting to a trusted server.
    """
    fingerprint_path = _get_fingerprint_path()

    if fingerprint_path.exists():
        return fingerprint_path.read_text().strip()

    # Generate a new secure random fingerprint
    fingerprint = secrets.token_hex(32)  # 256-bit

    # Save to disk
    fingerprint_path.write_text(fingerprint)

    return fingerprint


def get_server_fingerprint() -> str:
    """Get the server fingerprint, loading/creating it if necessary."""
    global _SERVER_FINGERPRINT
    if _SERVER_FINGERPRINT is None:
        _SERVER_FINGERPRINT = _load_or_create_fingerprint()
    return _SERVER_FINGERPRINT


# Handshake nonce management for QR code verification
# Maps nonce string -> expiry timestamp (Unix epoch)
_PENDING_NONCES: dict[str, float] = {}
NONCE_EXPIRY_SECONDS = 300  # 5 minutes


def _cleanup_expired_nonces() -> None:
    """Remove expired nonces from the pending nonces dict."""
    now = time.time()
    expired = [nonce for nonce, expiry in _PENDING_NONCES.items() if expiry < now]
    for nonce in expired:
        del _PENDING_NONCES[nonce]


def create_handshake_nonce() -> Tuple[str, float]:
    """
    Create a new handshake nonce for QR code verification.

    Generates a cryptographically secure nonce, stores it with a 5-minute
    expiry, and returns both the nonce and its expiry timestamp.

    Returns:
        Tuple of (nonce_string, expiry_timestamp)
    """
    # Clean up expired nonces lazily
    _cleanup_expired_nonces()

    # Generate a secure random nonce
    nonce = secrets.token_hex(32)  # 256-bit nonce
    expiry = time.time() + NONCE_EXPIRY_SECONDS

    _PENDING_NONCES[nonce] = expiry

    return nonce, expiry


def verify_handshake_nonce(nonce: str) -> bool:
    """
    Verify and consume a handshake nonce.

    Checks if the nonce exists, is not expired, and deletes it after use
    (one-time use only).

    Args:
        nonce: The nonce string to verify

    Returns:
        True if valid, False if invalid, expired, or already used
    """
    if nonce not in _PENDING_NONCES:
        return False

    expiry = _PENDING_NONCES[nonce]
    now = time.time()

    # Delete the nonce regardless of validity (one-time use)
    del _PENDING_NONCES[nonce]

    # Check if it's expired
    if expiry < now:
        return False

    return True


def create_token(
    user_id: int,
    username: str,
    access_level: str,
    can_write: bool,
    expiration_hours: int = JWT_EXPIRATION_HOURS
) -> str:
    """
    Create a JWT token for an authenticated user.

    Args:
        user_id: The user's database ID
        username: The user's username
        access_level: The user's access level (localhost, local_network, public)
        can_write: Whether the user has write permissions
        expiration_hours: Token validity period in hours (default 24)

    Returns:
        A signed JWT token string
    """
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "username": username,
        "access_level": access_level,
        "can_write": can_write,
        "iat": now,  # Issued at
        "exp": now + timedelta(hours=expiration_hours)  # Expiration
    }

    return jwt.encode(payload, get_secret_key(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT token string to decode

    Returns:
        The decoded payload dict if valid, None if invalid or expired

    Payload contains:
        - user_id: int
        - username: str
        - access_level: str
        - can_write: bool
        - iat: datetime (issued at)
        - exp: datetime (expiration)
    """
    try:
        payload = jwt.decode(
            token,
            get_secret_key(),
            algorithms=[JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        # Token has expired
        return None
    except jwt.InvalidTokenError:
        # Token is invalid (bad signature, malformed, etc.)
        return None


def refresh_token(token: str, expiration_hours: int = JWT_EXPIRATION_HOURS) -> Optional[str]:
    """
    Refresh a valid token with a new expiration time.

    Args:
        token: The current valid JWT token
        expiration_hours: New validity period in hours

    Returns:
        A new token with extended expiration, or None if the input token is invalid
    """
    payload = decode_token(token)
    if payload is None:
        return None

    return create_token(
        user_id=payload["user_id"],
        username=payload["username"],
        access_level=payload["access_level"],
        can_write=payload["can_write"],
        expiration_hours=expiration_hours
    )
