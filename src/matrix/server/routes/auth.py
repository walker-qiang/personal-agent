"""Authentication routes: register, login and logout.

Multi-user authentication. Username + password → verify against DB → JWT token.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request

from ...auth import create_token, hash_password, verify_password
from ...config import AgentConfig
from ...store import SessionStore

router = APIRouter(prefix="/api/auth", tags=["auth"])

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\u4e00-\u9fff][a-zA-Z0-9_\-.\u4e00-\u9fff]{0,29}$")
_MIN_PASSWORD_LEN = 4


@router.post("/register")
async def register(request: Request) -> dict[str, str]:
    """Register a new user account and return JWT token.

    Request body: {"username": "...", "password": "..."}
    Response: {"token": "..."}
    """
    config: AgentConfig = _get_config(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    username = (body.get("username", "") or "").strip()
    password = (body.get("password", "") or "").strip()

    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 1-30 chars, letters/numbers/Chinese/underscore only",
        )
    if len(password) < _MIN_PASSWORD_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {_MIN_PASSWORD_LEN} characters",
        )

    store: SessionStore = request.app.state.chat.store
    if store.get_user(username) is not None:
        raise HTTPException(status_code=409, detail="Username already exists")

    pwd_hash = hash_password(password)
    created = store.create_user(username, pwd_hash)
    if not created:
        raise HTTPException(status_code=409, detail="Username already exists")

    token = create_token(username, config.jwt_secret)
    return {"token": token}


@router.post("/login")
async def login(request: Request) -> dict[str, str]:
    """Authenticate with username and password, return JWT token.

    Request body: {"username": "...", "password": "..."}
    Response: {"token": "..."}
    """
    config: AgentConfig = _get_config(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    username = (body.get("username", "") or "").strip()
    password = (body.get("password", "") or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    store: SessionStore = request.app.state.chat.store
    user = store.get_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(username, config.jwt_secret)
    return {"token": token}


@router.post("/logout")
async def logout() -> dict[str, str]:
    """Logout — client should remove the token from localStorage."""
    return {"status": "ok"}


def _get_config(request: Request) -> AgentConfig:
    return request.app.state.config