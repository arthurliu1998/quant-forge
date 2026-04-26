import pytest
import pandas as pd
from quantforge.data.base import DataProvider
from quantforge.data.yfinance_provider import YFinanceProvider
from quantforge.data.twse_provider import TWSEProvider


def test_data_provider_is_abstract():
    with pytest.raises(TypeError):
        DataProvider()


def test_yfinance_provider_implements_interface():
    provider = YFinanceProvider()
    assert isinstance(provider, DataProvider)


def test_twse_provider_implements_interface():
    provider = TWSEProvider()
    assert isinstance(provider, DataProvider)


def test_twse_normalize_columns():
    provider = TWSEProvider()
    raw = pd.DataFrame({"date": ["2026/04/01"], "volume": [10000],
                        "open": [100.0], "high": [105.0], "low": [99.0], "close": [103.0]})
    normalized = provider._normalize_columns(raw)
    assert list(normalized.columns) == ["Date", "Volume", "Open", "High", "Low", "Close"]
