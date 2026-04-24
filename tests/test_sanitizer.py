import pytest
from trade_forge.providers.sanitizer import DataSanitizer


class TestSanitizeForLLM:
    def test_strips_portfolio_value(self):
        data = {"portfolio_value": 250000, "weight_pct": 5.0, "symbol": "TSLA"}
        result = DataSanitizer.sanitize_for_llm(data)
        assert "portfolio_value" not in result
        assert result["weight_pct"] == 5.0
        assert result["symbol"] == "TSLA"

    def test_strips_share_count(self):
        data = {"qty": 50, "pnl_pct": 4.8, "symbol": "AAPL"}
        result = DataSanitizer.sanitize_for_llm(data)
        assert "qty" not in result
        assert result["pnl_pct"] == 4.8

    def test_strips_nested_amounts(self):
        data = {
            "positions": [
                {"symbol": "TSLA", "qty": 50, "avg_cost": 248.0, "weight_pct": 5.0},
                {"symbol": "AAPL", "qty": 100, "avg_cost": 180.0, "weight_pct": 7.0},
            ]
        }
        result = DataSanitizer.sanitize_for_llm(data)
        for pos in result["positions"]:
            assert "qty" not in pos
            assert "avg_cost" not in pos
            assert "weight_pct" in pos

    def test_does_not_modify_original(self):
        data = {"portfolio_value": 250000, "symbol": "TSLA"}
        DataSanitizer.sanitize_for_llm(data)
        assert "portfolio_value" in data  # Original unchanged

    def test_keeps_safe_fields(self):
        data = {"rsi": 65.3, "score": 7.5, "regime": "trending_up", "confidence": 0.72}
        result = DataSanitizer.sanitize_for_llm(data)
        assert result == data


class TestSanitizeText:
    def test_strips_dollar_amounts(self):
        text = "Buy TSLA at $248.50, target $265"
        result = DataSanitizer.sanitize_text(text)
        assert "$248.50" not in result
        assert "$265" not in result
        assert "$***" in result

    def test_strips_share_counts(self):
        text = "Buy 50 shares of TSLA"
        result = DataSanitizer.sanitize_text(text)
        assert "50 shares" not in result
        assert "*** shares" in result

    def test_preserves_non_financial_text(self):
        text = "RSI at 65, MACD golden cross day 3"
        result = DataSanitizer.sanitize_text(text)
        assert result == text
