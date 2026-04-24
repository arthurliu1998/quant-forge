import pandas as pd
import pytest
from trade_forge.data.fetch_tw import fetch_tw_daily, fetch_tw_institutional


class TestFetchTWDaily:
    def test_returns_dataframe(self):
        df = fetch_tw_daily("2330")
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in df.columns

    def test_invalid_symbol_returns_empty(self):
        df = fetch_tw_daily("9999999")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestFetchTWInstitutional:
    def test_returns_dict(self):
        result = fetch_tw_institutional("2330")
        assert isinstance(result, dict)
        for key in ["foreign_net", "trust_net", "dealer_net"]:
            assert key in result
