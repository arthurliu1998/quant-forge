"""Global exception handler that redacts API keys from tracebacks."""
import sys
import traceback

from trade_forge._patterns import KEY_PATTERNS


def sanitize_text(text: str) -> str:
    for pattern in KEY_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _safe_excepthook(exc_type, exc_value, exc_tb):
    lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    for line in lines:
        sys.stderr.write(sanitize_text(line))


def install_exception_hooks():
    sys.excepthook = _safe_excepthook
