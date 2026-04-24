import pytest
from trade_forge.notify.telegram import TelegramNotifier, format_signal_message
from trade_forge.signals.engine import Signal


class TestFormatSignalMessage:
    def test_low_sensitivity_includes_price(self):
        s = Signal("TSLA", "rsi_oversold", "HIGH", "RSI at 15", 15.0, "bullish")
        msg = format_signal_message(s, sensitivity="low")
        assert "TSLA" in msg
        assert "15" in msg
        assert "HIGH" in msg

    def test_medium_sensitivity_no_dollar_amounts(self):
        s = Signal("TSLA", "ma20_crossover_up", "HIGH",
                   "Price crossed above MA20 ($248.50)", 248.5, "bullish")
        msg = format_signal_message(s, sensitivity="medium")
        assert "TSLA" in msg
        assert "$248.50" not in msg

    def test_high_sensitivity_symbol_and_direction_only(self):
        s = Signal("TSLA", "rsi_oversold", "HIGH", "RSI at 15", 15.0, "bullish")
        msg = format_signal_message(s, sensitivity="high")
        assert "TSLA" in msg
        assert "BULLISH" in msg
        assert "15" not in msg


class TestTelegramNotifier:
    def test_init_without_token_not_configured(self):
        notifier = TelegramNotifier(token="", chat_id="")
        assert notifier.is_configured() is False
