import os
import pytest
from trade_forge.providers.safe_subprocess import safe_run
from trade_forge.secrets import SecretManager

SENSITIVE_VARS = SecretManager.SECRETS


def test_strips_sensitive_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("SAFE_VAR", "safe-value")
    result = safe_run(["env"], capture_output=True, text=True)
    assert "sk-test-key" not in result.stdout
    assert "SAFE_VAR=safe-value" in result.stdout


def test_preserves_normal_vars(monkeypatch):
    monkeypatch.setenv("MY_NORMAL_VAR", "hello")
    result = safe_run(["env"], capture_output=True, text=True)
    assert "MY_NORMAL_VAR=hello" in result.stdout


def test_all_sensitive_vars_stripped(monkeypatch):
    for var in SENSITIVE_VARS:
        monkeypatch.setenv(var, f"secret-{var}")
    result = safe_run(["env"], capture_output=True, text=True)
    for var in SENSITIVE_VARS:
        assert var not in result.stdout
