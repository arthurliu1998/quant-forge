from trade_forge.safe_exceptions import sanitize_text, KEY_PATTERNS


def test_redacts_anthropic_key():
    text = "Error: Invalid key sk-ant-api03-abcdefghijk-lmnop-12345"
    result = sanitize_text(text)
    assert "sk-ant-api03" not in result
    assert "[REDACTED]" in result


def test_redacts_google_key():
    text = "google key: AIzaSyD-abcdefghijklmnopqrstuvwxyz012345"
    result = sanitize_text(text)
    assert "AIzaSyD" not in result
    assert "[REDACTED]" in result


def test_redacts_telegram_token():
    text = "bot token: bot1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
    result = sanitize_text(text)
    assert "bot1234567890" not in result
    assert "[REDACTED]" in result


def test_preserves_safe_text():
    text = "Normal error: connection timeout after 30 seconds"
    result = sanitize_text(text)
    assert result == text


def test_redacts_multiple_keys():
    text = "key1=sk-ant-api03-aaaa-bbbb key2=AIzaSyD-ccccddddeeeeffffgggg1234567890"
    result = sanitize_text(text)
    assert result.count("[REDACTED]") == 2
    assert "aaaa" not in result
    assert "cccc" not in result
