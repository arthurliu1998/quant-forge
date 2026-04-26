"""Edge Layer 1: Technical factor scoring.
Combines trend and mean-reversion sub-scores based on current regime.
"""
import pandas as pd
import numpy as np
from quantforge.factors.base import Factor
from quantforge.core.models import FactorScore, Regime
from quantforge.analysis.indicators import (
    compute_ma, compute_macd, compute_rsi, compute_kd,
    compute_bollinger, compute_adx, compute_volume_ratio, compute_atr,
)


class TechnicalFactor(Factor):
    def __init__(self, weight: float = 0.35):
        super().__init__("technical", weight)

    def compute(self, symbol: str, data: dict) -> FactorScore:
        df = data.get("ohlcv")
        regime = data.get("regime", Regime.NEUTRAL)

        if df is None or df.empty or len(df) < 30:
            return FactorScore(self.name, 0.5, 0.5, self.weight)

        trend_score = self._trend_score(df)
        reversion_score = self._reversion_score(df)

        if regime in (Regime.BULL_TREND, Regime.BEAR_TREND):
            raw = trend_score * 0.7 + reversion_score * 0.3
        elif regime == Regime.CONSOLIDATION:
            raw = trend_score * 0.3 + reversion_score * 0.7
        else:
            raw = trend_score * 0.5 + reversion_score * 0.5

        # Weekly confirmation: use MA20 as proxy
        ma20 = compute_ma(df["Close"], 20)
        if not pd.isna(ma20.iloc[-1]):
            price_above = df["Close"].iloc[-1] > ma20.iloc[-1]
            if regime == Regime.BULL_TREND and price_above:
                confirm = 1.0
            elif regime == Regime.BEAR_TREND and not price_above:
                confirm = 1.0
            elif price_above:
                confirm = 0.7
            else:
                confirm = 0.3
        else:
            confirm = 0.5

        adjusted = raw * (0.7 + 0.3 * confirm)
        normalized = max(0.0, min(1.0, adjusted))
        return FactorScore(self.name, raw, normalized, self.weight)

    def _trend_score(self, df: pd.DataFrame) -> float:
        close = df["Close"]
        ma20 = compute_ma(close, 20)
        macd, signal, hist = compute_macd(close)
        adx = compute_adx(df)
        vol_ratio = compute_volume_ratio(df["Volume"], 20)

        points = 0.0
        if not pd.isna(ma20.iloc[-1]) and len(ma20) >= 2:
            if close.iloc[-1] > ma20.iloc[-1] and ma20.iloc[-1] > ma20.iloc[-2]:
                points += 1.0
        if not pd.isna(hist.iloc[-1]):
            if hist.iloc[-1] > 0 and macd.iloc[-1] > signal.iloc[-1]:
                points += 1.0
        if not pd.isna(adx.iloc[-1]) and adx.iloc[-1] > 25:
            points += 1.0
        if not pd.isna(vol_ratio.iloc[-1]) and vol_ratio.iloc[-1] > 1.3:
            points += 1.0
        return points / 4.0

    def _reversion_score(self, df: pd.DataFrame) -> float:
        close = df["Close"]
        rsi = compute_rsi(close)
        bb_upper, _, bb_lower = compute_bollinger(close)
        k, d = compute_kd(df)

        points = 0.0
        if not pd.isna(rsi.iloc[-1]):
            if rsi.iloc[-1] < 30:
                points += 1.0
            elif rsi.iloc[-1] > 70:
                points -= 1.0
        if not pd.isna(bb_lower.iloc[-1]):
            if close.iloc[-1] < bb_lower.iloc[-1]:
                points += 1.0
            elif close.iloc[-1] > bb_upper.iloc[-1]:
                points -= 1.0
        if len(k) >= 2 and not pd.isna(k.iloc[-1]):
            if k.iloc[-1] < 20 and k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]:
                points += 1.0
            elif k.iloc[-1] > 80 and k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]:
                points -= 1.0
        return (points + 4.0) / 8.0  # map [-4,4] to [0,1]
