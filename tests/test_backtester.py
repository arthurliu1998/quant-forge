import numpy as np
import pandas as pd
import pytest
from trade_forge.backtester import Backtester, BacktestResult, SLIPPAGE


@pytest.fixture
def sample_data():
    """200 days of synthetic trending data."""
    np.random.seed(42)
    n = 200
    trend = np.linspace(100, 130, n) + np.cumsum(np.random.randn(n) * 1.5)
    return pd.DataFrame({
        "Open": trend + np.random.randn(n) * 0.3,
        "High": trend + abs(np.random.randn(n)) * 1.5,
        "Low": trend - abs(np.random.randn(n)) * 1.5,
        "Close": trend,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    })


@pytest.fixture
def signal_config():
    return {
        "ma_crossover": [20, 60],
        "volume_spike_ratio": 2.0,
        "rsi_overbought": 80,
        "rsi_oversold": 20,
    }


class TestBacktester:
    def test_returns_backtest_result(self, sample_data, signal_config):
        bt = Backtester(market="US")
        result = bt.backtest_signal(sample_data, signal_config)
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0

    def test_insufficient_data(self, signal_config):
        short_df = pd.DataFrame({
            "Open": [100]*20, "High": [101]*20, "Low": [99]*20,
            "Close": [100]*20, "Volume": [1_000_000]*20,
        })
        bt = Backtester()
        result = bt.backtest_signal(short_df, signal_config)
        assert result.verdict == "INSUFFICIENT"

    def test_slippage_applied(self):
        assert SLIPPAGE["US"] == 0.0005
        assert SLIPPAGE["TW"] > 0.004  # commission + tax makes TW expensive

    def test_to_dict(self, sample_data, signal_config):
        bt = Backtester()
        result = bt.backtest_signal(sample_data, signal_config)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "verdict" in d
        assert "win_rate" in d
        assert "sharpe" in d

    def test_metrics_in_valid_range(self, sample_data, signal_config):
        bt = Backtester()
        result = bt.backtest_signal(sample_data, signal_config)
        if result.total_trades > 0:
            assert 0 <= result.win_rate <= 100
            assert result.profit_factor >= 0
            assert result.max_drawdown_pct <= 0  # Drawdown is always negative or zero
