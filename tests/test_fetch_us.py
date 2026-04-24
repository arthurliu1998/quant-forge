import pandas as pd
import pytest
from trade_forge.data.fetch_us import fetch_ohlcv, fetch_current_price, fetch_company_info


class TestFetchOHLCV:
    def test_returns_dataframe(self):
        df = fetch_ohlcv("AAPL", period="5d")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    def test_invalid_symbol_returns_empty(self):
        df = fetch_ohlcv("ZZZZZZZNOTREAL", period="5d")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestFetchCurrentPrice:
    def test_returns_float(self):
        price = fetch_current_price("AAPL")
        assert isinstance(price, float)
        assert price > 0

    def test_invalid_symbol_returns_zero(self):
        price = fetch_current_price("ZZZZZZZNOTREAL")
        assert price == 0.0


class TestFetchCompanyInfo:
    def test_returns_dict_with_name(self):
        info = fetch_company_info("AAPL")
        assert isinstance(info, dict)
        assert "name" in info
        assert len(info["name"]) > 0

    def test_invalid_symbol_returns_empty_dict(self):
        info = fetch_company_info("ZZZZZZZNOTREAL")
        assert isinstance(info, dict)
        assert info.get("name", "") == ""
