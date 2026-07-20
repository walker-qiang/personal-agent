"""Tests for TraceSanitizer — PII redaction before trace persistence."""

from matrix.guardrails import TraceSanitizer


def test_phone_redacted():
    s = TraceSanitizer()
    result = s.sanitize("手机号 13812345678")
    assert "13812345678" not in result
    assert "[REDACTED:phone]" in result


def test_id_card_redacted():
    s = TraceSanitizer()
    result = s.sanitize("身份证 110101199001011234")
    assert "110101199001011234" not in result
    assert "[REDACTED:id_card]" in result


def test_email_redacted():
    s = TraceSanitizer()
    result = s.sanitize("邮箱 test@example.com")
    assert "test@example.com" not in result
    assert "[REDACTED:email]" in result


def test_api_key_redacted():
    s = TraceSanitizer()
    result = s.sanitize("sk-abcdef1234567890abcdef1234567890")
    assert "sk-abcdef" not in result
    assert "[REDACTED:api_key]" in result


def test_ip_redacted():
    s = TraceSanitizer()
    result = s.sanitize("192.168.1.1:8080")
    assert "192.168.1.1" not in result
    assert "[REDACTED:ip]" in result


def test_jwt_redacted():
    s = TraceSanitizer()
    result = s.sanitize("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I0nMjJhS2a")
    assert "[REDACTED:jwt]" in result


def test_none_passes_through():
    s = TraceSanitizer()
    assert s.sanitize(None) is None


def test_dict_recursive():
    s = TraceSanitizer()
    result = s.sanitize({"user": "test@example.com", "data": {"phone": "13812345678"}})
    assert "test@example.com" not in str(result)
    assert "13812345678" not in str(result)


def test_list_recursive():
    s = TraceSanitizer()
    result = s.sanitize(["alice@test.com", "13887654321"])
    assert "alice@test.com" not in str(result)
    assert "13887654321" not in str(result)


def test_normal_text_unchanged():
    s = TraceSanitizer()
    text = "The portfolio value is 100000 CNY"
    result = s.sanitize(text)
    assert result == text


def test_int_unchanged():
    s = TraceSanitizer()
    assert s.sanitize(42) == 42
    assert s.sanitize(3.14) == 3.14