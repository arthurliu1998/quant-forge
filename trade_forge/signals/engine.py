"""Rule-based signal detection engine. No LLM required."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class Signal:
    symbol: str
    type: str
    priority: str       # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    value: float = 0.0
    direction: str = ""  # bullish, bearish, neutral


class SignalEngine:
    def __init__(self, config: dict):
        self.ma_periods = config.get("ma_crossover", [20, 60])
        self.vol_spike = config.get("volume_spike_ratio", 2.0)
        self.rsi_ob = config.get("rsi_overbought", 80)
        self.rsi_os = config.get("rsi_oversold", 20)

    def detect(self, symbol: str, data: dict) -> list[Signal]:
        signals = []
        signals.extend(self._check_rsi(symbol, data))
        signals.extend(self._check_volume(symbol, data))
        signals.extend(self._check_ma_crossover(symbol, data))
        return signals

    def _check_rsi(self, symbol: str, data: dict) -> list[Signal]:
        signals = []
        rsi = data.get("rsi")
        if rsi is None or rsi.empty:
            return signals
        latest = rsi.iloc[-1]
        if pd.isna(latest):
            return signals
        if latest <= self.rsi_os:
            signals.append(Signal(symbol=symbol, type="rsi_oversold", priority="MEDIUM",
                message=f"RSI({latest:.1f}) below {self.rsi_os} — oversold", value=latest, direction="bullish"))
        elif latest >= self.rsi_ob:
            signals.append(Signal(symbol=symbol, type="rsi_overbought", priority="MEDIUM",
                message=f"RSI({latest:.1f}) above {self.rsi_ob} — overbought", value=latest, direction="bearish"))
        return signals

    def _check_volume(self, symbol: str, data: dict) -> list[Signal]:
        signals = []
        vol_ratio = data.get("vol_ratio_5d")
        if vol_ratio is None or vol_ratio.empty:
            return signals
        latest = vol_ratio.iloc[-1]
        if pd.isna(latest):
            return signals
        if latest >= self.vol_spike:
            signals.append(Signal(symbol=symbol, type="volume_spike", priority="HIGH",
                message=f"Volume {latest:.1f}x above 5d average", value=latest, direction="neutral"))
        return signals

    def _check_ma_crossover(self, symbol: str, data: dict) -> list[Signal]:
        signals = []
        close = data.get("close")
        if close is None or len(close) < 2:
            return signals
        for period in self.ma_periods:
            ma_key = f"ma_{period}"
            ma = data.get(ma_key)
            if ma is None or len(ma) < 2:
                continue
            curr_close, prev_close = close.iloc[-1], close.iloc[-2]
            curr_ma, prev_ma = ma.iloc[-1], ma.iloc[-2]
            if pd.isna(curr_ma) or pd.isna(prev_ma):
                continue
            if prev_close <= prev_ma and curr_close > curr_ma:
                signals.append(Signal(symbol=symbol, type=f"ma{period}_crossover_up", priority="HIGH",
                    message=f"Price crossed above MA{period} ({curr_ma:.2f})", value=curr_close, direction="bullish"))
            elif prev_close >= prev_ma and curr_close < curr_ma:
                signals.append(Signal(symbol=symbol, type=f"ma{period}_crossover_down", priority="HIGH",
                    message=f"Price crossed below MA{period} ({curr_ma:.2f})", value=curr_close, direction="bearish"))
        return signals
