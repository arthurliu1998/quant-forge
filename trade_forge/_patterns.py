"""Shared redaction patterns for API keys and sensitive data. Single source of truth."""
import re

# Order matters: more specific patterns first to avoid partial matches
KEY_PATTERNS = [
    re.compile(r'sk-ant-api\w{2}-[\w-]+'),        # Anthropic
    re.compile(r'AIza[\w-]{20,}'),                  # Google
    re.compile(r'bot\d{8,}:[\w-]+'),                # Telegram (before Alpaca — more specific)
    re.compile(r'(?:PK|AK|SK)[\w]{16,}'),           # Alpaca (PK/AK/SK prefix, narrower match)
]

AMOUNT_PATTERNS = [
    re.compile(r'\$[\d,]+\.?\d*'),                  # Dollar amounts
    re.compile(r'\d+\s*shares?'),                    # Share counts
]
