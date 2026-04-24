import pytest
import yaml
from trade_forge.config import SecureConfig, load_config


def test_secure_config_masks_sensitive_keys():
    cfg = SecureConfig({"api_key": "sk-secret-123", "name": "test"})
    text = repr(cfg)
    assert "sk-secret-123" not in text
    assert "REDACTED" in text
    assert "test" in text


def test_secure_config_str_also_masked():
    cfg = SecureConfig({"token": "bot12345:abc", "debug": True})
    assert "bot12345" not in str(cfg)


def test_load_config_returns_secure_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"watchlist": {"US": ["AAPL"]}}))
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, SecureConfig)
    assert cfg["watchlist"]["US"] == ["AAPL"]


def test_load_config_rejects_inline_telegram_token(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "notification": {"telegram": {"bot_token": "bot123456:REAL_TOKEN_HERE", "chat_id": "12345"}}
    }))
    with pytest.raises(ValueError, match="must use env ref"):
        load_config(str(cfg_file))


def test_load_config_allows_env_ref(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "notification": {"telegram": {"bot_token": "${TELEGRAM_BOT_TOKEN}", "chat_id": "${TELEGRAM_CHAT_ID}"}}
    }))
    cfg = load_config(str(cfg_file))
    assert cfg["notification"]["telegram"]["bot_token"] == "${TELEGRAM_BOT_TOKEN}"
