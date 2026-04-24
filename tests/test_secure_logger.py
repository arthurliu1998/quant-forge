import logging
import pytest
from trade_forge.monitor.secure_logger import SanitizingFilter, setup_logging


@pytest.fixture
def sanitizing_filter():
    return SanitizingFilter()


def _make_record(msg, *args):
    """Create a LogRecord for direct filter testing."""
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg=msg, args=args if args else None, exc_info=None,
    )
    return record


class TestSanitizingFilter:
    def test_redacts_anthropic_key(self, sanitizing_filter):
        record = _make_record("Key: sk-ant-api03-abcdefghijk-lmnop-12345")
        sanitizing_filter.filter(record)
        assert "sk-ant-api03" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_telegram_token(self, sanitizing_filter):
        record = _make_record("Token: bot1234567890:ABCdefGHI")
        sanitizing_filter.filter(record)
        assert "bot1234567890" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_dollar_amounts(self, sanitizing_filter):
        record = _make_record("Portfolio value: $250,000.50")
        sanitizing_filter.filter(record)
        assert "$250,000.50" not in record.msg
        assert "***" in record.msg

    def test_redacts_share_counts(self, sanitizing_filter):
        record = _make_record("Bought 50 shares of TSLA")
        sanitizing_filter.filter(record)
        assert "50 shares" not in record.msg
        assert "***" in record.msg

    def test_preserves_normal_messages(self, sanitizing_filter):
        record = _make_record("Normal log message with RSI=65")
        sanitizing_filter.filter(record)
        assert "Normal log message" in record.msg
        assert "RSI=65" in record.msg

    def test_redacts_in_format_args(self, sanitizing_filter):
        record = _make_record("API key is %s", "sk-ant-api03-xxxx-yyyy-zzzz")
        sanitizing_filter.filter(record)
        formatted = record.msg % record.args
        assert "sk-ant-api03" not in formatted


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "trade_forge"

    def test_blocks_http_debug(self):
        setup_logging()
        assert logging.getLogger("urllib3").level >= logging.WARNING
        assert logging.getLogger("httpx").level >= logging.WARNING
