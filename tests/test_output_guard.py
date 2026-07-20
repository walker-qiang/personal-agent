"""Tests for OutputGuard — PII redaction in agent responses."""

from matrix.guardrails import GuardConfig, OutputGuard


def _config(block_mode: bool = False) -> GuardConfig:
    return GuardConfig(output_block_mode=block_mode)


def test_phone_redaction():
    guard = OutputGuard(_config())
    result = guard.check("您的手机号 13812345678 已绑定成功")
    assert result.had_pii is True
    assert "13812345678" not in result.sanitized
    assert "138****5678" in result.sanitized
    assert "phone" in result.flags


def test_id_card_redaction():
    guard = OutputGuard(_config())
    result = guard.check("身份证号 110101199001011234 已登记")
    assert result.had_pii is True
    assert "110101199001011234" not in result.sanitized
    assert "1101**********1234" in result.sanitized
    assert "id_card" in result.flags


def test_email_redaction():
    guard = OutputGuard(_config())
    result = guard.check("请联系 test@example.com 获取更多信息")
    assert result.had_pii is True
    assert "test@example.com" not in result.sanitized
    assert "u***@example.com" in result.sanitized
    assert "email" in result.flags


def test_bank_card_redaction():
    guard = OutputGuard(_config())
    result = guard.check("尾号 6222021234567890 的银行卡")
    assert result.had_pii is True
    assert "6222021234567890" not in result.sanitized
    assert "****7890" in result.sanitized or "bank_card" in result.flags


def test_api_key_redaction():
    guard = OutputGuard(_config())
    result = guard.check("API key: sk-abc123def456ghi789jkl012mno345pqr678stu")
    assert result.had_pii is True
    assert "sk-abc123" not in result.sanitized
    assert "sk-***" in result.sanitized
    assert "api_key" in result.flags


def test_no_pii_text_unchanged():
    guard = OutputGuard(_config())
    text = "投资组合目前表现良好，总收益率为 12.5%。"
    result = guard.check(text)
    assert result.had_pii is False
    assert result.sanitized == text
    assert result.flags == []


def test_empty_text():
    guard = OutputGuard(_config())
    result = guard.check("")
    assert result.had_pii is False
    assert result.sanitized == ""


def test_multiple_pii_types():
    guard = OutputGuard(_config())
    text = "手机 13987654321，邮箱 alice@test.com"
    result = guard.check(text)
    assert result.had_pii is True
    assert "13987654321" not in result.sanitized
    assert "alice@test.com" not in result.sanitized
    assert len(result.flags) >= 2


def test_ip_redaction():
    guard = OutputGuard(_config())
    result = guard.check("服务器地址 192.168.1.1 无法访问")
    assert result.had_pii is True
    assert "192.168.1.1" not in result.sanitized
    assert "x.x.x.x" in result.sanitized


def test_phone_in_context_not_false_positive():
    """Ensure that normal numbers in context are not falsely flagged."""
    guard = OutputGuard(_config())
    text = "当前持仓总价值约为 100000 元，收益率 12.5%"
    result = guard.check(text)
    # Should not flag normal numbers as phone (needs leading 1[3-9])
    # However bank_card may match 16-digit numbers; 100000 is 6 digits, safe
    # 12.5% is not a number pattern
    assert "phone" not in result.flags