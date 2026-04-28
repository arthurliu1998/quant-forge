"""
Three-tier secret resolution: keyring -> env -> .env file.
"""
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ENV_PATH = os.path.expanduser("~/.quantforge/.env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)


class SecretManager:
    SECRETS = frozenset({
        "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY",
        "AMD_LLM_GATEWAY_KEY", "AMD_LLM_GATEWAY_URL",
        "ALPACA_DATA_KEY", "ALPACA_DATA_SECRET",
        "ALPACA_TRADE_KEY", "ALPACA_TRADE_SECRET",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL",
        "QUANTFORGE_DB_KEY",
    })

    @staticmethod
    def get(key: str) -> str:
        try:
            import keyring as kr
            val = kr.get_password("quantforge", key)
            if val:
                return val
        except Exception:
            pass
        val = os.environ.get(key, "")
        if val:
            if key in SecretManager.SECRETS:
                logger.debug("Secret %s loaded from environment", key)
            return val
        return ""

    @staticmethod
    def is_configured(key: str) -> bool:
        val = SecretManager.get(key)
        return bool(val and len(val) > 5)

    @staticmethod
    def masked(key: str) -> str:
        val = SecretManager.get(key)
        if not val:
            return "(not set)"
        if len(val) <= 8:
            return "***"
        return val[:6] + "***" + val[-4:]
