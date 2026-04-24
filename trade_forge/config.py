"""
Secure configuration loader.
- SecureConfig masks sensitive values in repr/str
- load_config rejects inline secrets in YAML
"""
import yaml

_SENSITIVE_SUBSTRINGS = ("key", "token", "secret", "password")


class SecureConfig(dict):
    def __repr__(self):
        masked = {}
        for k, v in self.items():
            if isinstance(v, str) and any(s in k.lower() for s in _SENSITIVE_SUBSTRINGS):
                masked[k] = "***REDACTED***"
            else:
                masked[k] = v
        return f"SecureConfig({masked})"

    def __str__(self):
        return self.__repr__()


def load_config(path: str) -> SecureConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    _validate_no_inline_secrets(raw)
    return SecureConfig(raw)


def _validate_no_inline_secrets(cfg: dict) -> None:
    telegram = cfg.get("notification", {}).get("telegram", {})
    for field in ("bot_token", "chat_id"):
        val = telegram.get(field, "")
        if val and not val.startswith("${"):
            raise ValueError(
                f"config.yaml: notification.telegram.{field} must use env ref "
                f"(e.g., '${{TELEGRAM_BOT_TOKEN}}'), not inline values. "
                f"Move the value to .env or keyring."
            )
