"""One-time, short-lived tickets for EventSource (SSE) connections.

Browsers' EventSource API cannot set custom headers, so SSE endpoints used to
accept the long-lived JWT via a URL query parameter. URLs end up in browser
history, access logs and Referer headers, which leaks the token. Instead the
client exchanges its JWT (via Authorization header) for a single-use ticket
that expires quickly and is bound to the issuing user.
"""

from __future__ import annotations

import secrets
import threading
import time

_TICKET_TTL_SECONDS = 300  # 5 minutes — enough to open the SSE connection
_MAX_TICKETS = 10000  # guard against unbounded growth


class StreamTicketStore:
    """In-memory one-time ticket store. Process-local; tickets are disposable."""

    def __init__(self, ttl_seconds: int = _TICKET_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._tickets: dict[str, tuple[str, float]] = {}  # ticket -> (user_id, expires_at)
        self._lock = threading.Lock()

    def issue(self, user_id: str) -> str:
        """Create a one-time ticket for the user."""
        ticket = secrets.token_urlsafe(24)
        with self._lock:
            self._evict_expired_locked()
            if len(self._tickets) >= _MAX_TICKETS:
                # Drop oldest entries to stay bounded
                for key, _ in sorted(
                    self._tickets.items(), key=lambda kv: kv[1][1]
                )[: _MAX_TICKETS // 10]:
                    self._tickets.pop(key, None)
            self._tickets[ticket] = (user_id, time.time() + self._ttl)
        return ticket

    def redeem(self, ticket: str) -> str | None:
        """Consume a ticket. Returns user_id, or None if invalid/expired/used."""
        if not ticket:
            return None
        with self._lock:
            entry = self._tickets.pop(ticket, None)
        if entry is None:
            return None
        user_id, expires_at = entry
        if time.time() > expires_at:
            return None
        return user_id

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._tickets.items() if exp < now]
        for k in expired:
            self._tickets.pop(k, None)
