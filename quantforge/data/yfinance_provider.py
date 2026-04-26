"""US stock data provider wrapping yfinance."""
import pandas as pd
from quantforge.data.base import DataProvider
from quantforge.data import fetch_us


class YFinanceProvider(DataProvider):
    def get_ohlcv(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        return fetch_us.fetch_ohlcv(symbol, period=period, interval=interval)

    def get_current_price(self, symbol: str) -> float:
        return fetch_us.fetch_current_price(symbol)

    def get_company_info(self, symbol: str) -> dict:
        return fetch_us.fetch_company_info(symbol)
