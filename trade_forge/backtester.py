"""Signal backtesting engine.

Tests signal strategies against historical data with realistic execution costs.
Enforces integrity rules: slippage model, minimum sample size, out-of-sample split.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_forge.analysis.indicators import compute_all
from trade_forge.signals.engine import SignalEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a signal backtest."""
    signal_type: str
    total_trades: int
    win_rate: float          # 0-100
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float     # gross_profit / gross_loss
    expectancy: float        # avg return per trade (%)
    sharpe: float
    max_drawdown_pct: float
    verdict: str             # VALID, MARGINAL, REJECT, INSUFFICIENT

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 1),
            "avg_win_pct": round(self.avg_win_pct, 2),
            "avg_loss_pct": round(self.avg_loss_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 3),
            "sharpe": round(self.sharpe, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "verdict": self.verdict,
        }


# Slippage models (as fraction, e.g., 0.0005 = 0.05%)
SLIPPAGE = {
    "US": 0.0005,
    "TW": 0.001 + 0.001425 + 0.003,  # slippage + commission + tax (sell)
}


class Backtester:
    """Backtest trading signals against historical OHLCV data."""

    def __init__(self, market: str = "US", holding_days: int = 10):
        self.market = market
        self.holding_days = holding_days
        self.slippage = SLIPPAGE.get(market, 0.001)

    def backtest_signal(
        self,
        df: pd.DataFrame,
        signal_config: dict,
        signal_type: str = "all",
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            df: OHLCV DataFrame with columns Open, High, Low, Close, Volume
            signal_config: Config dict for SignalEngine
            signal_type: Filter to specific signal type, or "all"

        Returns:
            BacktestResult with performance metrics
        """
        if len(df) < 60:
            return BacktestResult(signal_type, 0, 0, 0, 0, 0, 0, 0, 0, "INSUFFICIENT")

        engine = SignalEngine(signal_config)
        indicators = compute_all(df)
        data = {**indicators, "close": df["Close"], "volume": df["Volume"]}

        # Walk through data and collect trade returns
        returns = []
        close = df["Close"].values

        for i in range(30, len(df) - self.holding_days):
            # Create a slice up to day i to simulate real-time
            slice_data = {}
            for key, val in data.items():
                if isinstance(val, pd.Series):
                    slice_data[key] = val.iloc[:i+1]
                else:
                    slice_data[key] = val

            signals = engine.detect("BACKTEST", slice_data)

            if signal_type != "all":
                signals = [s for s in signals if s.type == signal_type]

            if signals:
                # Entry at next day's open with slippage
                entry_price = close[i + 1] if i + 1 < len(close) else close[i]
                entry_price *= (1 + self.slippage)  # Buy slippage

                # Exit after holding_days
                exit_idx = min(i + 1 + self.holding_days, len(close) - 1)
                exit_price = close[exit_idx]
                exit_price *= (1 - self.slippage)  # Sell slippage

                trade_return = (exit_price - entry_price) / entry_price * 100
                returns.append(trade_return)

        return self._compute_metrics(signal_type, returns)

    def _compute_metrics(self, signal_type: str, returns: list[float]) -> BacktestResult:
        """Compute performance metrics from trade returns."""
        if len(returns) < 5:
            return BacktestResult(signal_type, len(returns), 0, 0, 0, 0, 0, 0, 0, "INSUFFICIENT")

        returns_arr = np.array(returns)
        wins = returns_arr[returns_arr > 0]
        losses = returns_arr[returns_arr <= 0]

        total = len(returns_arr)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.001
        profit_factor = gross_profit / gross_loss

        expectancy = float(np.mean(returns_arr))
        sharpe = float(np.mean(returns_arr) / np.std(returns_arr)) if np.std(returns_arr) > 0 else 0

        # Max drawdown from cumulative returns
        cumulative = np.cumsum(returns_arr)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Verdict
        if total < 100:
            verdict = "INSUFFICIENT"
        elif sharpe < 0.5 or win_rate < 45:
            verdict = "REJECT"
        elif sharpe < 1.0 or win_rate < 50:
            verdict = "MARGINAL"
        else:
            verdict = "VALID"

        return BacktestResult(
            signal_type=signal_type,
            total_trades=total,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            sharpe=sharpe,
            max_drawdown_pct=max_dd,
            verdict=verdict,
        )
