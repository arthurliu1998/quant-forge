"""Multi-Mode Adaptive Strategy engine.

Runs three engines in parallel per bar:
  1. Trend / breakout positions (multiple simultaneous)
  2. Wheel (simulated CSP/CC options income)
  3. Cash (no new trades during bear/crisis)

Mode selection per bar:
  BEAR/CRISIS           -> CASH
  BULL + vol > threshold -> HIGHVOL_BREAKOUT (4x ATR trail)
  BULL + vol <= threshold-> TREND_FOLLOW     (3x ATR trail)
  CONSOLIDATION/NEUTRAL -> WHEEL            (CSP/CC cycle)

Trend/breakout signals fire in any non-bearish regime (including during
WHEEL periods) so that trending stocks are never blocked from entry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from quantforge.analysis.indicators import compute_atr, compute_adx, compute_ma
from quantforge.backtest.cost_model import CostModel
from quantforge.core.models import Regime
from quantforge.pricing.bsm import bsm_put_price, bsm_call_price
from quantforge.regime.detector import RegimeDetector


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class TradeRecord:
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    ret_pct: float
    exit_reason: str
    signal_type: str
    regime: str
    pos_pct: float
    mode: str = ""


@dataclass
class WheelStats:
    total_premium: float = 0.0
    csp_cycles: int = 0
    cc_cycles: int = 0
    assignments: int = 0
    calls_away: int = 0


@dataclass
class MultiModeResult:
    trades: list[TradeRecord]
    final_equity: float
    equity_curve: list[tuple]     # [(datetime, equity), ...]
    mode_log: list[tuple]         # [(bar_index, mode_str), ...]
    wheel_stats: WheelStats


@dataclass
class _OpenPosition:
    entry_price: float
    entry_idx: int
    highest: float
    stop: float
    atr_mult: float
    mode: str
    signal_type: str
    entry_ma: int
    pct: float


# ── Helpers ───────────────────────────────────────────────────────

R_LBL = {
    Regime.BULL_TREND: "Bull", Regime.BEAR_TREND: "Bear",
    Regime.CONSOLIDATION: "Range", Regime.CRISIS: "Crisis",
    Regime.NEUTRAL: "Neutral",
}


def compute_annualized_vol(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling annualized volatility from log returns."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window=window).std() * np.sqrt(252)


def compute_regime_series(df: pd.DataFrame, vix_df: pd.DataFrame | None = None) -> list[Regime]:
    det = RegimeDetector()
    adx_s = compute_adx(df)
    ma60_s = compute_ma(df["Close"], 60)
    out: list[Regime] = []
    for i in range(len(df)):
        if i < 60 or pd.isna(adx_s.iloc[i]) or pd.isna(ma60_s.iloc[i]):
            out.append(Regime.NEUTRAL)
            continue
        vix = 15.0
        if vix_df is not None:
            m = vix_df.index <= df.index[i]
            if m.any():
                vix = float(vix_df.loc[m, "Close"].iloc[-1])
        out.append(det.detect(float(adx_s.iloc[i]), float(df["Close"].iloc[i]),
                              float(ma60_s.iloc[i]), vix))
    return out


def determine_mode(regime: Regime, ann_vol: float, vol_threshold: float = 0.45) -> str:
    if regime in (Regime.BEAR_TREND, Regime.CRISIS):
        return "CASH"
    if regime == Regime.BULL_TREND:
        return "HIGHVOL_BREAKOUT" if ann_vol > vol_threshold else "TREND_FOLLOW"
    return "WHEEL"


def generate_trend_signals(df: pd.DataFrame, indicators: dict) -> list[dict]:
    """MA crossover + RSI bounce signals."""
    close = df["Close"]
    signals: list[dict] = []
    for i in range(1, len(df)):
        for p in [20, 60]:
            ma = indicators.get(f"ma_{p}")
            if ma is None or pd.isna(ma.iloc[i]) or pd.isna(ma.iloc[i - 1]):
                continue
            if close.iloc[i - 1] <= ma.iloc[i - 1] and close.iloc[i] > ma.iloc[i]:
                sc = 70.0
                r = indicators["rsi"].iloc[i]
                v = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(r) and r < 50:
                    sc += 10
                if not pd.isna(v) and v > 1.5:
                    sc += 10
                signals.append({"index": i, "direction": "long", "score": sc,
                                "type": f"MA{p}_up", "ma_period": p})
        rsi = indicators["rsi"]
        if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i - 1]):
            if rsi.iloc[i - 1] <= 30 and rsi.iloc[i] > 30:
                sc = 65.0
                v = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(v) and v > 1.5:
                    sc += 15
                h = indicators["macd_hist"]
                if not pd.isna(h.iloc[i]) and h.iloc[i] > h.iloc[max(0, i - 1)]:
                    sc += 10
                signals.append({"index": i, "direction": "long", "score": sc,
                                "type": "RSI_bounce", "ma_period": 20})
    return signals


def generate_breakout_signals(df: pd.DataFrame, atr_series: pd.Series) -> list[dict]:
    """20-day high breakout + volume surge + ATR expanding."""
    close = df["Close"]
    high = df["High"]
    volume = df["Volume"]
    signals: list[dict] = []
    for i in range(20, len(df)):
        high_20 = float(high.iloc[i - 20:i].max())
        if float(close.iloc[i]) <= high_20:
            continue
        vol_avg = float(volume.iloc[i - 20:i].mean())
        if vol_avg <= 0 or float(volume.iloc[i]) < 2.0 * vol_avg:
            continue
        if i < 10 or pd.isna(atr_series.iloc[i]) or pd.isna(atr_series.iloc[i - 10]):
            continue
        atr_prev = float(atr_series.iloc[i - 10])
        if atr_prev <= 0:
            continue
        if float(atr_series.iloc[i]) / atr_prev < 1.2:
            continue
        signals.append({"index": i, "direction": "long", "score": 75.0,
                        "type": "breakout_20d"})
    return signals


# ── Strategy class ────────────────────────────────────────────────

class MultiModeStrategy:
    """Configurable multi-mode strategy.

    Parameters
    ----------
    vol_threshold : float
        Annualized vol above which BULL regime uses HIGHVOL_BREAKOUT.
    trend_atr_mult : float
        ATR multiplier for trend trailing stop.
    breakout_atr_mult : float
        ATR multiplier for breakout trailing stop.
    base_pct_bull / base_pct_neutral / base_pct_consol : float
        Base position size (%) per regime.
    risk_cap_bull / risk_cap_neutral / risk_cap_consol : float
        Max capital risk per trade per regime.
    max_exposure_pct : float
        Maximum total trend exposure as % of capital.
    max_hold_days : int
        Maximum bars to hold a trend/breakout position.
    wheel_cycle_bars : int
        Trading days per wheel CSP/CC cycle.
    wheel_otm : float
        OTM percentage for wheel strikes.
    wheel_premium_discount : float
        Bid-ask discount applied to BSM fair value.
    commission_per_contract : float
        Options commission per contract.
    """

    def __init__(
        self,
        vol_threshold: float = 0.45,
        trend_atr_mult: float = 3.0,
        breakout_atr_mult: float = 4.0,
        base_pct_bull: float = 50.0,
        base_pct_neutral: float = 30.0,
        base_pct_consol: float = 20.0,
        risk_cap_bull: float = 0.10,
        risk_cap_neutral: float = 0.05,
        risk_cap_consol: float = 0.03,
        max_exposure_pct: float = 80.0,
        max_hold_days: int = 250,
        wheel_cycle_bars: int = 30,
        wheel_otm: float = 0.10,
        wheel_premium_discount: float = 0.85,
        commission_per_contract: float = 0.65,
    ):
        self.vol_threshold = vol_threshold
        self.trend_atr_mult = trend_atr_mult
        self.breakout_atr_mult = breakout_atr_mult
        self.base_pct = {
            "bull": base_pct_bull,
            "neutral": base_pct_neutral,
            "consol": base_pct_consol,
        }
        self.risk_cap = {
            "bull": risk_cap_bull,
            "neutral": risk_cap_neutral,
            "consol": risk_cap_consol,
        }
        self.max_exposure_pct = max_exposure_pct
        self.max_hold_days = max_hold_days
        self.wheel_cycle_bars = wheel_cycle_bars
        self.wheel_cycle_days = int(wheel_cycle_bars * 1.5)  # calendar approx
        self.wheel_otm = wheel_otm
        self.wheel_premium_discount = wheel_premium_discount
        self.commission_per_contract = commission_per_contract

    # ── sizing helpers ─────────────────────────────────────────

    def _sizing_params(self, regime: Regime) -> tuple[float, float]:
        """Return (base_pct, risk_cap) for the given regime."""
        if regime == Regime.BULL_TREND:
            return self.base_pct["bull"], self.risk_cap["bull"]
        if regime == Regime.NEUTRAL:
            return self.base_pct["neutral"], self.risk_cap["neutral"]
        return self.base_pct["consol"], self.risk_cap["consol"]

    # ── main simulation ───────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        trend_signals: list[dict],
        breakout_signals: list[dict],
        regimes: list[Regime],
        ann_vol_series: pd.Series,
        indicators: dict,
        atr_series: pd.Series,
    ) -> MultiModeResult:
        """Simulate the multi-mode strategy bar-by-bar."""
        cm = CostModel()
        cap = 100.0
        equity_curve: list[tuple] = [(df.index[0], 100.0)]
        trades: list[TradeRecord] = []
        mode_log: list[tuple] = []
        wheel_stats = WheelStats()

        # Signal lookups
        trend_map: dict[int, list[dict]] = {}
        for s in trend_signals:
            trend_map.setdefault(s["index"], []).append(s)
        breakout_map: dict[int, list[dict]] = {}
        for s in breakout_signals:
            breakout_map.setdefault(s["index"], []).append(s)

        ma20 = indicators["ma_20"]
        ma60 = indicators["ma_60"]

        open_positions: list[_OpenPosition] = []

        # Wheel state
        wh_state = "idle"
        wh_cycle_start = 0
        wh_strike = 0.0
        wh_entry_price = 0.0

        for i in range(60, len(df)):
            av = float(ann_vol_series.iloc[i]) if not pd.isna(ann_vol_series.iloc[i]) else 0.25
            regime = regimes[i]
            mode = determine_mode(regime, av, self.vol_threshold)
            mode_log.append((i, mode))

            close_i = float(df["Close"].iloc[i])
            low_i = float(df["Low"].iloc[i])
            high_i = float(df["High"].iloc[i])
            cur_atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0

            is_bearish = regime in (Regime.BEAR_TREND, Regime.CRISIS)
            is_bull_highvol = (regime == Regime.BULL_TREND and av > self.vol_threshold)

            # ── 1. Manage open positions ──
            still_open: list[_OpenPosition] = []
            for pos in open_positions:
                exited = False
                xp = 0.0
                xr = ""

                if close_i > pos.highest:
                    pos.highest = close_i
                    if cur_atr > 0:
                        pos.stop = pos.highest - pos.atr_mult * cur_atr

                if low_i <= pos.stop:
                    xp, xr, exited = pos.stop, "trail_stop", True

                if not exited and is_bearish and i - pos.entry_idx > 5:
                    xp, xr, exited = close_i, "regime_exit", True

                if not exited and i - pos.entry_idx > 10:
                    ma_ref = ma20 if pos.entry_ma == 20 else ma60
                    if (not pd.isna(ma_ref.iloc[i]) and not pd.isna(ma_ref.iloc[i - 1])
                            and close_i < float(ma_ref.iloc[i])):
                        prev_close = float(df["Close"].iloc[i - 1])
                        if prev_close < float(ma_ref.iloc[i - 1]):
                            xp, xr, exited = close_i, "ma_break", True

                if not exited and i - pos.entry_idx >= self.max_hold_days:
                    xp, xr, exited = close_i, f"time_{self.max_hold_days}d", True

                if exited:
                    ee = cm.apply_entry(pos.entry_price, "US")
                    ex = cm.apply_exit(xp, "US")
                    rp = (ex - ee) / ee * 100
                    cap += cap * (rp * pos.pct / 100 / 100)
                    trades.append(TradeRecord(
                        df.index[pos.entry_idx], df.index[i],
                        pos.entry_price, xp, rp, xr, pos.signal_type,
                        R_LBL.get(regimes[pos.entry_idx], "?"), pos.pct, pos.mode,
                    ))
                    equity_curve.append((df.index[i], cap))
                else:
                    still_open.append(pos)
            open_positions = still_open

            # ── 2. Enter new positions ──
            total_exposure = sum(p.pct for p in open_positions)
            room = self.max_exposure_pct - total_exposure

            if i < len(df) - 1 and not is_bearish and room > 1.0:
                sz_base, sz_risk = self._sizing_params(regime)

                # Breakout
                if is_bull_highvol and i in breakout_map and room > 1.0:
                    sig = breakout_map[i][0]
                    ep = float(df["Open"].iloc[i + 1])
                    if ep > 0 and cur_atr > 0:
                        inv_vol = min(1.0, 0.30 / av) if av > 0 else 0.5
                        base = sz_base * inv_vol
                        stop_dist = self.breakout_atr_mult * cur_atr
                        max_val = (cap * sz_risk) / (stop_dist / ep)
                        pct = min(base, (max_val / cap * 100) if cap > 0 else 0, room)
                        if pct > 0.5:
                            open_positions.append(_OpenPosition(
                                entry_price=ep, entry_idx=i + 1, highest=ep,
                                stop=ep - stop_dist, atr_mult=self.breakout_atr_mult,
                                mode="HIGHVOL_BREAKOUT", signal_type=sig["type"],
                                entry_ma=20, pct=pct,
                            ))
                            room -= pct

                # Trend
                if room > 1.0 and i in trend_map:
                    sig = trend_map[i][0]
                    ep = float(df["Open"].iloc[i + 1])
                    if ep > 0 and cur_atr > 0:
                        stop_dist = self.trend_atr_mult * cur_atr
                        max_val = (cap * sz_risk) / (stop_dist / ep)
                        pct = min(sz_base, (max_val / cap * 100) if cap > 0 else 0, room)
                        if pct > 0.5:
                            open_positions.append(_OpenPosition(
                                entry_price=ep, entry_idx=i + 1, highest=ep,
                                stop=ep - stop_dist, atr_mult=self.trend_atr_mult,
                                mode="TREND_FOLLOW", signal_type=sig["type"],
                                entry_ma=sig.get("ma_period", 20), pct=pct,
                            ))

            # ── 3. Wheel ──
            wheel_allowed = mode == "WHEEL"
            T_years = self.wheel_cycle_days / 365.0

            if wheel_allowed:
                spot = close_i
                if wh_state == "idle":
                    wh_strike = spot * (1.0 - self.wheel_otm)
                    prem = bsm_put_price(spot, wh_strike, av, T_years) * 100
                    prem = prem * self.wheel_premium_discount - self.commission_per_contract
                    if prem > 0:
                        notional = wh_strike * 100
                        cap += cap * (prem / notional)
                        wheel_stats.total_premium += prem
                        wheel_stats.csp_cycles += 1
                        wh_state = "put_sold"
                        wh_cycle_start = i

                elif wh_state == "put_sold":
                    if low_i < wh_strike:
                        wh_entry_price = wh_strike
                        wh_state = "stock_held"
                        wheel_stats.assignments += 1
                    elif i - wh_cycle_start >= self.wheel_cycle_bars:
                        wh_state = "idle"

                elif wh_state == "stock_held":
                    call_strike = wh_entry_price * (1.0 + self.wheel_otm)
                    prem = bsm_call_price(spot, call_strike, av, T_years) * 100
                    prem = prem * self.wheel_premium_discount - self.commission_per_contract
                    if prem > 0:
                        notional = wh_entry_price * 100
                        cap += cap * (prem / notional)
                        wheel_stats.total_premium += prem
                        wheel_stats.cc_cycles += 1
                    wh_state = "call_sold"
                    wh_cycle_start = i
                    wh_strike = call_strike

                elif wh_state == "call_sold":
                    if high_i > wh_strike:
                        gain = (wh_strike - wh_entry_price) / wh_entry_price
                        notional = wh_entry_price * 100
                        weight = min(0.30, notional / cap if cap > 0 else 0)
                        cap += cap * gain * weight
                        wheel_stats.calls_away += 1
                        wh_state = "idle"
                    elif i - wh_cycle_start >= self.wheel_cycle_bars:
                        wh_state = "stock_held"

            elif wh_state != "idle":
                if wh_state in ("stock_held", "call_sold"):
                    gain = (close_i - wh_entry_price) / wh_entry_price
                    notional = wh_entry_price * 100
                    weight = min(0.30, notional / cap if cap > 0 else 0)
                    cap += cap * gain * weight
                wh_state = "idle"

            if i % 5 == 0:
                equity_curve.append((df.index[i], cap))

        equity_curve.append((df.index[-1], cap))
        return MultiModeResult(
            trades=trades, final_equity=cap, equity_curve=equity_curve,
            mode_log=mode_log, wheel_stats=wheel_stats,
        )
