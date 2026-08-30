"""Auth middleware for protecting API routes with JWT verification."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..auth import verify_token
from .stream_ticket import StreamTicketStore

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

        # Token from header. EventSource cannot set headers, so SSE clients
        # authenticate with a one-time ticket query param instead of the JWT.
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_token(token, config.jwt_secret)
            if payload is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"},
                )
            request.state.user_id = payload["sub"]
            return await call_next(request)

        ticket = request.query_params.get("ticket", "")
        if ticket:
            tickets: StreamTicketStore = request.app.state.stream_tickets
            user_id = tickets.redeem(ticket)
            if user_id is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid, expired or already-used ticket"},
                )
            request.state.user_id = user_id
            return await call_next(request)

        # Legacy path: raw JWT in query param (kept for backward compatibility
        # with older clients).
        token = request.query_params.get("token", "")
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header"},
            )
        payload = verify_token(token, config.jwt_secret)
        if payload is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"},
            )
        request.state.user_id = payload["sub"]
        return await call_next(request)
