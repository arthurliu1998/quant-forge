import pandas as pd
import numpy as np
from quantforge.core.models import Regime
from quantforge.regime.detector import RegimeDetector


def test_bull_trend():
    d = RegimeDetector()
    assert d.detect(adx=28.0, price=150.0, ma60=140.0, vix=15.0) == Regime.BULL_TREND


def test_bear_trend():
    d = RegimeDetector()
    assert d.detect(adx=30.0, price=130.0, ma60=140.0, vix=20.0) == Regime.BEAR_TREND


def test_consolidation():
    d = RegimeDetector()
    assert d.detect(adx=15.0, price=140.0, ma60=140.0, vix=18.0) == Regime.CONSOLIDATION


def test_crisis_overrides():
    d = RegimeDetector()
    assert d.detect(adx=30.0, price=150.0, ma60=140.0, vix=35.0) == Regime.CRISIS


def test_neutral():
    d = RegimeDetector()
    assert d.detect(adx=22.0, price=141.0, ma60=140.0, vix=20.0) == Regime.NEUTRAL


def test_detect_from_dataframe():
    d = RegimeDetector()
    n = 80
    prices = [100 + i * 0.5 for i in range(n)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 1 for p in prices],
        "Low": [p - 1 for p in prices],
        "Close": prices,
        "Volume": [1000000] * n,
    })
    regime = d.detect_from_data(df, vix=15.0)
    assert isinstance(regime, Regime)


def test_detect_from_empty_data():
    d = RegimeDetector()
    assert d.detect_from_data(pd.DataFrame(), vix=15.0) == Regime.NEUTRAL
    assert d.detect_from_data(None, vix=15.0) == Regime.NEUTRAL
