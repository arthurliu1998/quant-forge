"""Abstract base class for data providers."""
from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_company_info(self, symbol: str) -> dict:
        ...
