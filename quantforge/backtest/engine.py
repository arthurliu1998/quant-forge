"""Event-driven backtest engine.

Simulates trading over historical data using the QuantForge factor pipeline.
Supports two modes:
  - Classic: Fixed ATR stop/target (simple, fast)
  - Multi-mode: Adaptive trend/breakout/wheel/cash via MultiModeStrategy
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
        """Run classic backtest: fixed ATR stop/target, 60d max hold.

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

            entry_price = float(df["Open"].iloc[idx + 1])
            current_atr = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0
            if current_atr <= 0 or entry_price <= 0:
                continue

            effective_entry = self.costs.apply_entry(entry_price, self.market)
            stop = entry_price - self.atr_stop_mult * current_atr
            target = entry_price + self.atr_target_mult * current_atr

            exit_price = None
            for j in range(idx + 2, min(idx + 60, len(df))):
                if float(df["Low"].iloc[j]) <= stop:
                    exit_price = stop
                    break
                if float(df["High"].iloc[j]) >= target:
                    exit_price = target
                    break

            if exit_price is None:
                exit_price = float(df["Close"].iloc[min(idx + 59, len(df) - 1)])

            effective_exit = self.costs.apply_exit(exit_price, self.market)
            ret_pct = (effective_exit - effective_entry) / effective_entry * 100
            trade_returns.append(ret_pct)

        return compute_metrics(trade_returns, len(df))

    def run_multimode(self, df: pd.DataFrame, vix_df: pd.DataFrame = None,
                      strategy=None):
        """Run multi-mode adaptive backtest.

        Uses MultiModeStrategy engine with parallel trend/breakout + wheel.

        Args:
            df: OHLCV DataFrame
            vix_df: VIX OHLCV DataFrame (optional, defaults to VIX=15)
            strategy: MultiModeStrategy instance (uses defaults if None)

        Returns:
            MultiModeResult with trades, equity curve, mode log, wheel stats
        """
        from quantforge.strategy.multimode import (
            MultiModeStrategy,
            compute_regime_series, compute_annualized_vol,
            generate_trend_signals, generate_breakout_signals,
        )

        if strategy is None:
            strategy = MultiModeStrategy()

        indicators = compute_all(df)
        atr_series = compute_atr(df)
        trend_signals = generate_trend_signals(df, indicators)
        breakout_signals = generate_breakout_signals(df, atr_series)
        regimes = compute_regime_series(df, vix_df)
        ann_vol = compute_annualized_vol(df)

        return strategy.run(
            df, trend_signals, breakout_signals, regimes, ann_vol,
            indicators, atr_series,
        )

    async def run_async(self, df: pd.DataFrame, signals: list[dict],
                        initial_capital: float = 100000) -> BacktestMetrics:
        """Run classic backtest with optional LLM signal filtering."""
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
