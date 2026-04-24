import os
import pytest
from trade_forge.secrets import SecretManager


def test_get_from_env(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_KEY", "test-value-123")
    assert SecretManager.get("TEST_SECRET_KEY") == "test-value-123"


def test_get_missing_returns_empty():
    result = SecretManager.get("DEFINITELY_NOT_SET_EVER_12345")
    assert result == ""


def test_is_configured_true(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "some-long-value")
    assert SecretManager.is_configured("TEST_KEY") is True


def test_is_configured_false_when_missing():
    assert SecretManager.is_configured("MISSING_KEY_999") is False


def test_is_configured_false_when_short(monkeypatch):
    monkeypatch.setenv("SHORT_KEY", "abc")
    assert SecretManager.is_configured("SHORT_KEY") is False


def test_masked_returns_partial(monkeypatch):
    monkeypatch.setenv("MASK_TEST", "sk-ant-api03-abcdefghijklmnop")
    result = SecretManager.masked("MASK_TEST")
    assert result.startswith("sk-ant")
    assert result.endswith("mnop")
    assert "***" in result
    assert "abcdefghijklmnop" not in result


def test_masked_missing():
    assert SecretManager.masked("NOPE_KEY_999") == "(not set)"


def test_masked_short_key(monkeypatch):
    monkeypatch.setenv("TINY", "abc")
    assert SecretManager.masked("TINY") == "***"
