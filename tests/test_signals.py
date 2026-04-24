import pandas as pd
import pytest
from trade_forge.signals.engine import SignalEngine, Signal


@pytest.fixture
def config():
    return {"ma_crossover": [20, 60], "volume_spike_ratio": 2.0, "rsi_overbought": 80, "rsi_oversold": 20}


class TestSignalEngine:
    def test_detect_returns_list(self, config):
        engine = SignalEngine(config)
        data = {"close": pd.Series([100]*5), "ma_20": pd.Series([100]*5),
                "rsi": pd.Series([50]*5), "volume": pd.Series([1e6]*5), "vol_ratio_5d": pd.Series([1.0]*5)}
        signals = engine.detect("AAPL", data)
        assert isinstance(signals, list)

    def test_rsi_oversold_signal(self, config):
        engine = SignalEngine(config)
        data = {"rsi": pd.Series([50, 45, 30, 18, 15]), "close": pd.Series([100]*5),
                "ma_20": pd.Series([100]*5), "volume": pd.Series([1e6]*5), "vol_ratio_5d": pd.Series([1.0]*5)}
        signals = engine.detect("TSLA", data)
        rsi_signals = [s for s in signals if s.type == "rsi_oversold"]
        assert len(rsi_signals) > 0
        assert rsi_signals[0].priority in ("MEDIUM", "HIGH")

    def test_volume_spike_signal(self, config):
        engine = SignalEngine(config)
        data = {"rsi": pd.Series([50]*5), "close": pd.Series([100]*5), "ma_20": pd.Series([100]*5),
                "volume": pd.Series([1e6,1e6,1e6,1e6,3e6]), "vol_ratio_5d": pd.Series([1.0,1.0,1.0,1.0,2.5])}
        signals = engine.detect("NVDA", data)
        vol_signals = [s for s in signals if s.type == "volume_spike"]
        assert len(vol_signals) > 0

    def test_no_signal_on_neutral_data(self, config):
        engine = SignalEngine(config)
        data = {"rsi": pd.Series([50]*5), "close": pd.Series([100]*5), "ma_20": pd.Series([100]*5),
                "volume": pd.Series([1e6]*5), "vol_ratio_5d": pd.Series([1.0]*5)}
        signals = engine.detect("AAPL", data)
        assert len(signals) == 0


class TestSignalDataclass:
    def test_signal_fields(self):
        s = Signal(symbol="AAPL", type="rsi_oversold", priority="HIGH", message="RSI at 15", value=15.0)
        assert s.symbol == "AAPL"
        assert s.priority == "HIGH"
