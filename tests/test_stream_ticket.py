"""Tests for the one-time SSE stream ticket store."""

import time

from matrix.server.stream_ticket import StreamTicketStore


def test_issue_and_redeem_returns_user():
    store = StreamTicketStore()
    ticket = store.issue("alice")
    assert store.redeem(ticket) == "alice"


def test_ticket_is_single_use():
    store = StreamTicketStore()
    ticket = store.issue("alice")
    assert store.redeem(ticket) == "alice"
    assert store.redeem(ticket) is None


def test_unknown_ticket_rejected():
    store = StreamTicketStore()
    assert store.redeem("nonexistent") is None
    assert store.redeem("") is None


def test_expired_ticket_rejected():
    store = StreamTicketStore(ttl_seconds=0)
    ticket = store.issue("bob")
    time.sleep(0.01)
    assert store.redeem(ticket) is None


def test_tickets_are_unique():
    store = StreamTicketStore()
    tickets = {store.issue("u") for _ in range(100)}
    assert len(tickets) == 100
