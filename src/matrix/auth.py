"""Authentication: JWT token generation/verification and password hashing.

Multi-user authentication. Passwords are bcrypt-hashed and stored in the
database. JWT tokens are signed with a secret from .env.
"""

from __future__ import annotations

import datetime
import secrets
from typing import Any

import bcrypt
import jwt


DEFAULT_JWT_EXPIRY_HOURS = 24


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(
    user_id: str,
    secret: str,
    expiry_hours: int = DEFAULT_JWT_EXPIRY_HOURS,
) -> str:
    """Create a JWT token for the given user."""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + datetime.timedelta(hours=expiry_hours),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, secret: str) -> dict[str, Any] | None:
    """Verify a JWT token. Returns payload or None if invalid."""
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None