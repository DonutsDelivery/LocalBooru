"""
User management router - manage user accounts for network access authentication.

All endpoints in this router are localhost-only (enforced by middleware).
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import secrets
import hashlib

from ..database import get_db
from ..models import User, AccessLevel

router = APIRouter()


# Password hashing utilities
def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2 with random salt.

    Returns: "salt:hash" format string
    """
    salt = secrets.token_hex(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations=100000
    )
    hash_hex = hash_bytes.hex()
    return f"{salt}:{hash_hex}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored hash.

    Args:
        password: Plain text password to verify
        stored_hash: "salt:hash" format string from database

    Returns: True if password matches
    """
    try:
        salt, hash_hex = stored_hash.split(":")
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hash_bytes.hex() == hash_hex
    except (ValueError, AttributeError):
        return False


class UserCreate(BaseModel):
    """Request body for creating a user"""
    username: str
    password: str
    access_level: Optional[Literal["localhost", "local_network", "public"]] = "local_network"
    can_write: Optional[bool] = False


class UserUpdate(BaseModel):
    """Request body for updating a user"""
    password: Optional[str] = None
    is_active: Optional[bool] = None
    access_level: Optional[Literal["localhost", "local_network", "public"]] = None
    can_write: Optional[bool] = None


class UserResponse(BaseModel):
    """Response model for user data (excludes password)"""
    id: int
    username: str
    is_active: bool
    access_level: str
    can_write: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


@router.get("")
async def list_users(db: AsyncSession = Depends(get_db)):
    """
    List all user accounts.
    """
    result = await db.execute(select(User).order_by(User.username))
    users = result.scalars().all()

    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "is_active": u.is_active,
                "access_level": u.access_level.value if u.access_level else "local_network",
                "can_write": u.can_write,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "last_login": u.last_login.isoformat() if u.last_login else None
            }
            for u in users
        ]
    }


@router.post("")
async def create_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Create a new user account.
    """
    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")

    # Validate password
    if len(user_data.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    # Create user
    user = User(
        username=user_data.username,
        password_hash=hash_password(user_data.password),
        is_active=True,
        access_level=AccessLevel(user_data.access_level),
        can_write=user_data.can_write
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return {
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "is_active": user.is_active,
            "access_level": user.access_level.value,
            "can_write": user.can_write
        }
    }


@router.get("/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get a specific user by ID.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "username": user.username,
        "is_active": user.is_active,
        "access_level": user.access_level.value if user.access_level else "local_network",
        "can_write": user.can_write,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None
    }


@router.patch("/{user_id}")
async def update_user(user_id: int, updates: UserUpdate, db: AsyncSession = Depends(get_db)):
    """
    Update a user account.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Apply updates
    if updates.password is not None:
        if len(updates.password) < 4:
            raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
        user.password_hash = hash_password(updates.password)

    if updates.is_active is not None:
        user.is_active = updates.is_active

    if updates.access_level is not None:
        user.access_level = AccessLevel(updates.access_level)

    if updates.can_write is not None:
        user.can_write = updates.can_write

    await db.commit()
    await db.refresh(user)

    return {
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "is_active": user.is_active,
            "access_level": user.access_level.value if user.access_level else "local_network",
            "can_write": user.can_write
        }
    }


@router.delete("/{user_id}")
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """
    Delete a user account.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    await db.commit()

    return {"success": True, "deleted_user_id": user_id}


# Authentication endpoint for remote access
class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(credentials: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a user.

    Note: This endpoint is accessible from all access levels since
    authentication is needed for remote access.
    """
    result = await db.execute(
        select(User).where(User.username == credentials.username)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    # TODO: Return a session token for authenticated access
    # For now, just return success
    return {
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "access_level": user.access_level.value if user.access_level else "local_network",
            "can_write": user.can_write
        }
    }
