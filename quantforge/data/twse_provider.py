"""Taiwan stock data provider wrapping TWSE API."""
import pandas as pd
from quantforge.data.base import DataProvider
from quantforge.data import fetch_tw


class TWSEProvider(DataProvider):
    def get_ohlcv(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        df = fetch_tw.fetch_tw_daily(symbol)
        if df.empty:
            return df
        return self._normalize_columns(df)

    def get_current_price(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, period="1mo")
        if df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])

    def get_company_info(self, symbol: str) -> dict:
        return {"name": "", "sector": "", "industry": "", "market_cap": 0}

    def get_institutional_flow(self, symbol: str) -> dict:
        return fetch_tw.fetch_tw_institutional(symbol)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        col_map = {"date": "Date", "volume": "Volume", "open": "Open",
                   "high": "High", "low": "Low", "close": "Close"}
        return df.rename(columns=col_map)
