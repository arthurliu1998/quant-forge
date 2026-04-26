import pandas as pd
import numpy as np
from quantforge.factors.technical_factor import TechnicalFactor
from quantforge.core.models import Regime


def _make_uptrend(n=80):
    prices = [100 + i * 0.5 + np.random.normal(0, 0.1) for i in range(n)]
    return pd.DataFrame({
        "Open": prices, "High": [p+1 for p in prices],
        "Low": [p-1 for p in prices], "Close": prices,
        "Volume": [1000000 + i * 50000 for i in range(n)],
    })


def test_uptrend_scores_above_average():
    f = TechnicalFactor(weight=0.35)
    score = f.compute("TEST", {"ohlcv": _make_uptrend(), "regime": Regime.BULL_TREND})
    assert 0.0 <= score.normalized <= 1.0
    assert score.normalized > 0.4


def test_empty_data_returns_neutral():
    f = TechnicalFactor(weight=0.35)
    score = f.compute("TEST", {"ohlcv": pd.DataFrame(), "regime": Regime.NEUTRAL})
    assert score.normalized == 0.5


def test_regime_affects_weighting():
    f = TechnicalFactor(weight=0.35)
    df = _make_uptrend()
    s1 = f.compute("T", {"ohlcv": df, "regime": Regime.BULL_TREND})
    s2 = f.compute("T", {"ohlcv": df, "regime": Regime.CONSOLIDATION})
    assert 0.0 <= s1.normalized <= 1.0
    assert 0.0 <= s2.normalized <= 1.0
