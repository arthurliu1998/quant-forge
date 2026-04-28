"""Event-driven backtest engine.

Simulates trading over historical data using the QuantForge factor pipeline.
Applies realistic costs, respects risk limits, and produces analytics.
"""
import asyncio
import pandas as pd
import numpy as np
from quantforge.backtest.cost_model import CostModel
from quantforge.backtest.analytics import compute_metrics, BacktestMetrics
from quantforge.analysis.indicators import compute_atr, compute_all


class BacktestEngine:
    def __init__(self, market: str = "US", cost_model: CostModel = None,
                 atr_stop_mult: float = 2.0, atr_target_mult: float = 3.0,
                 llm_filter=None):
        self.market = market
        self.costs = cost_model or CostModel()
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.llm_filter = llm_filter

    def run(self, df: pd.DataFrame, signals: list[dict],
            initial_capital: float = 100000) -> BacktestMetrics:
        """Run backtest on historical data with given signals.

        Args:
            df: OHLCV DataFrame
            signals: list of {index: int, direction: "long"/"short", score: float}
            initial_capital: starting capital

        Returns:
            BacktestMetrics with full performance analysis
        """
        if df.empty or not signals:
            return compute_metrics([], 0)

        atr = compute_atr(df)
        trade_returns = []

        for sig in signals:
            idx = sig["index"]
            if idx < 1 or idx >= len(df) - 1:
                continue

            entry_price = float(df["Open"].iloc[idx + 1])  # next day open
            current_atr = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0
            if current_atr <= 0 or entry_price <= 0:
                continue

            # Apply entry cost
            effective_entry = self.costs.apply_entry(entry_price, self.market)

            # Set stop and target
            stop = entry_price - self.atr_stop_mult * current_atr
            target = entry_price + self.atr_target_mult * current_atr

            # Walk forward to find exit
            exit_price = None
            for j in range(idx + 2, min(idx + 60, len(df))):  # max 60 days hold
                low = float(df["Low"].iloc[j])
                high = float(df["High"].iloc[j])
                close = float(df["Close"].iloc[j])

                if low <= stop:
                    exit_price = stop  # stopped out
                    break
                if high >= target:
                    exit_price = target  # target hit
                    break

            if exit_price is None:
                # Time exit at last available day
                exit_price = float(df["Close"].iloc[min(idx + 59, len(df) - 1)])

            effective_exit = self.costs.apply_exit(exit_price, self.market)
            ret_pct = (effective_exit - effective_entry) / effective_entry * 100
            trade_returns.append(ret_pct)

        trading_days = len(df)
        return compute_metrics(trade_returns, trading_days)

    async def run_async(self, df: pd.DataFrame, signals: list[dict],
                        initial_capital: float = 100000) -> BacktestMetrics:
        """Run backtest with optional LLM signal filtering.

        Same as run() but supports async LLM filter. If no llm_filter
        is configured, behaves identically to run().

        Args:
            df: OHLCV DataFrame
            signals: list of {index: int, direction: "long"/"short", score: float}
            initial_capital: starting capital

        Returns:
            BacktestMetrics with full performance analysis
        """
        if df.empty or not signals:
            return compute_metrics([], 0)

        filtered_signals = signals
        if self.llm_filter:
            indicators = compute_all(df)
            filtered_signals = await self.llm_filter.filter_signals(
                df, signals, indicators
            )
            if not filtered_signals:
                return compute_metrics([], len(df))

        return self.run(df, filtered_signals, initial_capital)
