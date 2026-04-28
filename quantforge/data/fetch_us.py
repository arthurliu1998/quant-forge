"""US stock data fetching via yfinance."""
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for a US stock."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning("No OHLCV data for %s", symbol)
        return df
    except Exception as e:
        logger.error("Failed to fetch %s: %s", symbol, type(e).__name__)
        return pd.DataFrame()


def fetch_current_price(symbol: str) -> float:
    """Fetch current price for a US stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        return float(info.get("lastPrice", 0.0))
    except Exception:
        return 0.0


def fetch_company_info(symbol: str) -> dict:
    """Fetch basic company info for a US stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {"name": "", "sector": "", "industry": "", "market_cap": 0}
