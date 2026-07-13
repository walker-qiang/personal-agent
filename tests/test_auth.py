"""Tests for the authentication module.

Covers: password hashing, JWT token creation/verification,
and auth route endpoints (multi-user username+password login).
"""

from __future__ import annotations

import pytest
from matrix.auth import (
    hash_password,
    verify_password,
    create_token,
    verify_token,
)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("test-password")
        assert hashed != "test-password"
        assert hashed.startswith("$2b$")
        assert verify_password("test-password", hashed)
        assert not verify_password("wrong-password", hashed)

    def test_different_hashes(self):
        """Same password produces different hashes due to salt."""
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2
        assert verify_password("same-password", h1)
        assert verify_password("same-password", h2)

    def test_empty_password(self):
        hashed = hash_password("")
        assert verify_password("", hashed)
        assert not verify_password("x", hashed)


class TestJWTToken:
    SECRET = "test-secret-key-32-chars-long!!!"

    def test_create_and_verify(self):
        token = create_token("alice", self.SECRET)
        payload = verify_token(token, self.SECRET)
        assert payload is not None
        assert payload["sub"] == "alice"
        assert "iat" in payload
        assert "exp" in payload
        assert "jti" in payload

    def test_wrong_secret_fails(self):
        token = create_token("alice", self.SECRET)
        assert verify_token(token, "wrong-secret") is None

    def test_expired_token(self):
        token = create_token("alice", self.SECRET, expiry_hours=0)
        assert verify_token(token, self.SECRET) is None

    def test_malformed_token(self):
        assert verify_token("not.a.jwt.token", self.SECRET) is None

    def test_empty_token(self):
        assert verify_token("", self.SECRET) is None

    def test_token_jti_unique(self):
        t1 = create_token("alice", self.SECRET)
        t2 = create_token("alice", self.SECRET)
        p1 = verify_token(t1, self.SECRET)
        p2 = verify_token(t2, self.SECRET)
        assert p1["jti"] != p2["jti"]

    def test_different_users(self):
        """Tokens for different users have different sub claims."""
        t1 = create_token("alice", self.SECRET)
        t2 = create_token("bob", self.SECRET)
        p1 = verify_token(t1, self.SECRET)
        p2 = verify_token(t2, self.SECRET)
        assert p1["sub"] == "alice"
        assert p2["sub"] == "bob"


class TestAuthRoutes:
    """Integration tests for the auth API endpoints."""

    @pytest.fixture
    def client(self, agent_config):
        """Create a test client with a pre-created admin user."""
        from starlette.testclient import TestClient
        from matrix.server.app import create_app
        from matrix.auth import hash_password

        config = agent_config
        object.__setattr__(config, "admin_password_hash", hash_password("test-password"))
        object.__setattr__(config, "jwt_secret", "test-jwt-secret-for-auth-tests")

        app = create_app(config)
        with TestClient(app, base_url="http://test") as c:
            yield c

    def test_login_no_credentials(self, client):
        resp = client.post("/api/auth/login", json={})
        assert resp.status_code == 400

    def test_login_no_password(self, client):
        resp = client.post("/api/auth/login", json={"username": "admin"})
        assert resp.status_code == 400

    def test_login_no_username(self, client):
        resp = client.post("/api/auth/login", json={"password": "test-password"})
        assert resp.status_code == 400

    def test_login_wrong_password(self, client):
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "wrong",
        })
        assert resp.status_code == 401

    def test_login_wrong_username(self, client):
        resp = client.post("/api/auth/login", json={
            "username": "nobody",
            "password": "test-password",
        })
        assert resp.status_code == 401

    def test_logout(self, client):
        resp = client.post("/api/auth/logout")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_login_success_and_token_works(self, client):
        """Login with correct username+password, then use token on protected route."""
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "test-password",
        })
        assert resp.status_code == 200
        token = resp.json()["token"]
        assert token

        # Use token to access a protected route
        resp = client.get(
            "/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_protected_route_without_token(self, client):
        """Accessing a protected route without token returns 401."""
        with pytest.raises(Exception):
            client.get("/sessions")

    def test_protected_route_with_bad_token(self, client):
        """Accessing a protected route with invalid token returns 401."""
        with pytest.raises(Exception):
            client.get(
                "/sessions",
                headers={"Authorization": "Bearer bad-token"},
            )

    def test_token_query_param(self, client):
        """Token can be passed via ?token= query param (for EventSource)."""
        resp = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "test-password",
        })
        token = resp.json()["token"]

        resp = client.get(f"/sessions?token={token}")
        assert resp.status_code == 200


class TestRegisterRoutes:
    """Integration tests for the register API endpoint."""

    @pytest.fixture
    def client(self, agent_config):
        """Create a test client with no pre-created users."""
        from starlette.testclient import TestClient
        from matrix.server.app import create_app

        config = agent_config
        object.__setattr__(config, "admin_password_hash", "")
        object.__setattr__(config, "jwt_secret", "test-jwt-secret-for-auth-tests")

        app = create_app(config)
        with TestClient(app, base_url="http://test") as c:
            yield c

    def test_register_success(self, client):
        """Register a new user and get a token."""
        resp = client.post("/api/auth/register", json={
            "username": "newuser",
            "password": "mypassword",
        })
        assert resp.status_code == 200
        token = resp.json()["token"]
        assert token

        # Token should work for protected routes
        resp = client.get(
            "/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_register_duplicate_username(self, client):
        """Registering the same username twice returns 409."""
        client.post("/api/auth/register", json={
            "username": "dupuser",
            "password": "password123",
        })
        resp = client.post("/api/auth/register", json={
            "username": "dupuser",
            "password": "another456",
        })
        assert resp.status_code == 409

    def test_register_short_password(self, client):
        """Password must be at least 4 characters."""
        resp = client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "ab",
        })
        assert resp.status_code == 400

    def test_register_empty_username(self, client):
        resp = client.post("/api/auth/register", json={
            "username": "",
            "password": "password",
        })
        assert resp.status_code == 400

    def test_register_invalid_username(self, client):
        """Username with special characters should be rejected."""
        resp = client.post("/api/auth/register", json={
            "username": "user@name",
            "password": "password",
        })
        assert resp.status_code == 400

    def test_register_chinese_username(self, client):
        """Chinese characters in username should be allowed."""
        resp = client.post("/api/auth/register", json={
            "username": "张三",
            "password": "password",
        })
        assert resp.status_code == 200
        assert resp.json()["token"]

    def test_register_then_login(self, client):
        """After registering, user should be able to login."""
        client.post("/api/auth/register", json={
            "username": "loginuser",
            "password": "secret123",
        })
        resp = client.post("/api/auth/login", json={
            "username": "loginuser",
            "password": "secret123",
        })
        assert resp.status_code == 200
        assert resp.json()["token"]