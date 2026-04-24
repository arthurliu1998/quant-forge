"""Strip sensitive financial data before sending to LLM APIs.

Rule: LLM only sees percentages and ratios, NEVER absolute amounts.
- Position weight: 5% <- OK
- Portfolio value: $250,000 <- STRIP
- P&L: +4.8% <- OK
- P&L: +$12,000 <- STRIP
- Share count: 50 shares <- STRIP
"""
import re


class DataSanitizer:
    """Sanitize data dict before sending to LLM."""

    # Fields that contain absolute amounts — always remove
    STRIP_FIELDS = frozenset({
        "portfolio_value", "account_balance", "position_value",
        "avg_cost", "pnl_dollar", "qty", "shares", "share_count",
        "total_pnl", "daily_pnl_dollar",
    })

    # Fields that are safe (percentages, ratios, scores)
    KEEP_FIELDS = frozenset({
        "weight_pct", "pnl_pct", "stop_pct", "target_pct",
        "sector_exposure_pct", "daily_pnl_pct", "drawdown_pct",
        "rsi", "macd", "score", "confidence", "regime",
    })

    @staticmethod
    def sanitize_for_llm(data: dict) -> dict:
        """Remove absolute financial amounts, keep percentages and ratios.

        Operates on a copy — never modifies the original dict.
        """
        sanitized = {}
        for key, value in data.items():
            if key in DataSanitizer.STRIP_FIELDS:
                continue
            if isinstance(value, dict):
                sanitized[key] = DataSanitizer.sanitize_for_llm(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    DataSanitizer.sanitize_for_llm(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove dollar amounts and share counts from text strings."""
        # Remove dollar amounts: $123, $1,234.56, $12,345,678
        text = re.sub(r'\$[\d,]+\.?\d*', '$***', text)
        # Remove share counts: "50 shares", "100 shares"
        text = re.sub(r'\d+\s*shares?', '*** shares', text)
        return text
