"""
Log sanitization and secure logging setup.

- SanitizingFilter redacts API keys, dollar amounts, share counts from all log output
- setup_logging() configures root logger with sanitization + rotation
- Forces HTTP libraries to WARNING level to prevent auth header leaks
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from trade_forge._patterns import KEY_PATTERNS, AMOUNT_PATTERNS


class SanitizingFilter(logging.Filter):
    """Strip sensitive data from log messages."""

    def filter(self, record):
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern in KEY_PATTERNS:
                msg = pattern.sub("[REDACTED]", msg)
            for pattern in AMOUNT_PATTERNS:
                msg = pattern.sub("***", msg)
            record.msg = msg
        # Clear args to prevent format string reintroducing data
        if record.args:
            new_args = []
            for arg in (record.args if isinstance(record.args, tuple) else (record.args,)):
                if isinstance(arg, str):
                    sanitized = arg
                    for pattern in KEY_PATTERNS:
                        sanitized = pattern.sub("[REDACTED]", sanitized)
                    new_args.append(sanitized)
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        return True


def setup_logging(log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with sanitization filter and optional file rotation.

    Args:
        log_dir: Directory for log files. If None, console only.
        level: Logging level (default INFO).

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("trade_forge")
    logger.setLevel(level)

    # Add sanitizing filter
    sanitizer = SanitizingFilter()
    logger.addFilter(sanitizer)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    console.addFilter(sanitizer)
    logger.addHandler(console)

    # File handler with rotation (if log_dir specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "monitor.log"),
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ))
        file_handler.addFilter(sanitizer)
        logger.addHandler(file_handler)

    # Block HTTP debug logging (prevents auth header leaks)
    for lib in ['urllib3', 'httpx', 'aiohttp', 'requests', 'httpcore',
                'anthropic._base_client', 'google.auth.transport',
                'websockets']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger
