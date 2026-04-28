"""Taiwan stock data fetching via yfinance (TWSE symbols use .TW suffix)."""
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _tw_symbol(symbol: str) -> str:
    """Convert TW stock number to yfinance ticker (e.g., 2330 -> 2330.TW)."""
    if not symbol.endswith(".TW") and not symbol.endswith(".TWO"):
        return f"{symbol}.TW"
    return symbol


def fetch_tw_daily(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch daily OHLCV for a Taiwan stock."""
    try:
        yf_symbol = _tw_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period)
        if df.empty:
            logger.warning("No data for TW:%s", symbol)
            return pd.DataFrame()
        # Normalize to lowercase columns expected by TW scanner
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        return df
    except Exception as e:
        logger.error("Failed to fetch TW:%s: %s", symbol, type(e).__name__)
        return pd.DataFrame()


def fetch_tw_institutional(symbol: str) -> dict:
    """Placeholder for institutional flow data (TWSE API)."""
    return {
        "foreign_buy": 0, "foreign_sell": 0,
        "trust_buy": 0, "trust_sell": 0,
        "dealer_buy": 0, "dealer_sell": 0,
    }
