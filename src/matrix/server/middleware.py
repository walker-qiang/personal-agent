"""Auth middleware for protecting API routes with JWT verification."""

from __future__ import annotations

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..auth import verify_token

PUBLIC_PATHS = {
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/logout",
    "/healthz",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT verification middleware.

    Skips public paths. All other routes require a valid Bearer token.
    Injects user_id into request.state for downstream handlers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        if path in PUBLIC_PATHS:
            return await call_next(request)

        config = request.app.state.config

        # Token from header or query param (EventSource has no custom headers)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = request.query_params.get("token", "")
            if not token:
                raise HTTPException(status_code=401, detail="Missing Authorization header")

        payload = verify_token(token, config.jwt_secret)
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        request.state.user_id = payload["sub"]
        return await call_next(request)