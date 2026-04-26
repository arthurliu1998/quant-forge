"""Market regime detection using ADX, MA60, and VIX."""
from quantforge.core.models import Regime
from quantforge.analysis.indicators import compute_adx, compute_ma
import pandas as pd


class RegimeDetector:
    def __init__(self, adx_trend_threshold: float = 25.0,
                 adx_consolidation_threshold: float = 20.0,
                 vix_crisis_threshold: float = 30.0):
        self.adx_trend = adx_trend_threshold
        self.adx_consol = adx_consolidation_threshold
        self.vix_crisis = vix_crisis_threshold

    def detect(self, adx: float, price: float, ma60: float, vix: float) -> Regime:
        if vix > self.vix_crisis:
            return Regime.CRISIS
        if adx > self.adx_trend and price > ma60:
            return Regime.BULL_TREND
        if adx > self.adx_trend and price < ma60:
            return Regime.BEAR_TREND
        if adx < self.adx_consol:
            return Regime.CONSOLIDATION
        return Regime.NEUTRAL

    def detect_from_data(self, df: pd.DataFrame, vix: float = 15.0) -> Regime:
        if df is None or df.empty or len(df) < 60:
            return Regime.NEUTRAL
        adx_series = compute_adx(df)
        ma60_series = compute_ma(df["Close"], period=60)
        adx_val = adx_series.iloc[-1]
        ma60_val = ma60_series.iloc[-1]
        price = df["Close"].iloc[-1]
        if pd.isna(adx_val) or pd.isna(ma60_val):
            return Regime.NEUTRAL
        return self.detect(adx=adx_val, price=price, ma60=ma60_val, vix=vix)
