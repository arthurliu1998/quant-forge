import numpy as np
import pandas as pd
import pytest
from trade_forge.analysis.indicators import (
    compute_ma, compute_rsi, compute_macd, compute_kd,
    compute_bollinger, compute_atr, compute_volume_ratio, compute_all,
)


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.5,
        "High": close + abs(np.random.randn(n)) * 1.5,
        "Low": close - abs(np.random.randn(n)) * 1.5,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    })


class TestMA:
    def test_compute_ma_20(self, sample_ohlcv):
        result = compute_ma(sample_ohlcv["Close"], period=20)
        assert len(result) == len(sample_ohlcv)
        assert pd.isna(result.iloc[0])
        assert not pd.isna(result.iloc[19])


class TestRSI:
    def test_rsi_in_range(self, sample_ohlcv):
        rsi = compute_rsi(sample_ohlcv["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestMACD:
    def test_macd_returns_three_series(self, sample_ohlcv):
        macd, signal, hist = compute_macd(sample_ohlcv["Close"])
        assert len(macd) == len(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)
        assert len(hist) == len(sample_ohlcv)


class TestKD:
    def test_kd_in_range(self, sample_ohlcv):
        k, d = compute_kd(sample_ohlcv)
        k_valid = k.dropna()
        d_valid = d.dropna()
        assert (k_valid >= 0).all() and (k_valid <= 100).all()
        assert (d_valid >= 0).all() and (d_valid <= 100).all()


class TestBollinger:
    def test_bollinger_upper_above_lower(self, sample_ohlcv):
        upper, mid, lower = compute_bollinger(sample_ohlcv["Close"])
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= lower[valid_idx]).all()


class TestATR:
    def test_atr_positive(self, sample_ohlcv):
        atr = compute_atr(sample_ohlcv)
        valid = atr.dropna()
        assert (valid > 0).all()


class TestVolumeRatio:
    def test_volume_ratio_positive(self, sample_ohlcv):
        ratio = compute_volume_ratio(sample_ohlcv["Volume"], period=5)
        valid = ratio.dropna()
        assert (valid > 0).all()


class TestComputeAll:
    def test_compute_all_returns_dict(self, sample_ohlcv):
        result = compute_all(sample_ohlcv)
        assert isinstance(result, dict)
        for key in ["ma_20", "ma_60", "rsi", "macd", "macd_signal", "macd_hist",
                     "k", "d", "bb_upper", "bb_mid", "bb_lower", "atr", "vol_ratio_5d"]:
            assert key in result
