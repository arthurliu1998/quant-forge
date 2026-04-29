#!/usr/bin/env python3
"""Compare exit strategies: Fixed ATR Target vs Trailing Stop vs Smart Exit.

Tests 3 exit modes:
  1. Fixed:    Stop at -2 ATR, Target at +3 ATR, max 60 days (current)
  2. Trail:    Trailing stop at -2 ATR from highest, no target cap, max 120 days
  3. Smart:    Trailing stop + regime exit + MA breakdown exit
"""
import numpy as np, pandas as pd
from quantforge.data.fetch_us import fetch_ohlcv
from quantforge.analysis.indicators import (
    compute_all, compute_atr, compute_adx, compute_ma, compute_volume_ratio
)
from quantforge.backtest.cost_model import CostModel
from quantforge.backtest.analytics import compute_metrics
from quantforge.regime.detector import RegimeDetector
from quantforge.core.models import Regime


def generate_signals(df, indicators):
    close = df["Close"]; signals = []
    for i in range(1, len(df)):
        for p in [20, 60]:
            ma = indicators.get(f"ma_{p}")
            if ma is None or pd.isna(ma.iloc[i]) or pd.isna(ma.iloc[i-1]): continue
            if close.iloc[i-1] <= ma.iloc[i-1] and close.iloc[i] > ma.iloc[i]:
                sc = 70.0
                r = indicators["rsi"].iloc[i]; v = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(r) and r < 50: sc += 10
                if not pd.isna(v) and v > 1.5: sc += 10
                signals.append({"index": i, "direction": "long", "score": sc,
                                "type": f"MA{p}_up", "ma_period": p})
        rsi = indicators["rsi"]
        if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i-1]):
            if rsi.iloc[i-1] <= 30 and rsi.iloc[i] > 30:
                sc = 65.0; v = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(v) and v > 1.5: sc += 15
                h = indicators["macd_hist"]
                if not pd.isna(h.iloc[i]) and h.iloc[i] > h.iloc[max(0, i-1)]: sc += 10
                signals.append({"index": i, "direction": "long", "score": sc,
                                "type": "RSI_bounce", "ma_period": 20})
    return signals


def compute_regimes(df, vix_df):
    det = RegimeDetector(); adx_s = compute_adx(df); ma60_s = compute_ma(df["Close"], 60)
    out = []
    for i in range(len(df)):
        if i < 60 or pd.isna(adx_s.iloc[i]) or pd.isna(ma60_s.iloc[i]):
            out.append(Regime.NEUTRAL); continue
        vix = 15.0
        if vix_df is not None:
            m = vix_df.index <= df.index[i]
            if m.any(): vix = float(vix_df.loc[m, "Close"].iloc[-1])
        out.append(det.detect(float(adx_s.iloc[i]), float(df["Close"].iloc[i]),
                              float(ma60_s.iloc[i]), vix))
    return out


def run_backtest(df, signals, regimes, indicators, exit_mode="fixed",
                 base_pct=0.20, max_risk_pct=0.03):
    """
    exit_mode:
      "fixed"  - Stop -2ATR, Target +3ATR, max 60d (current)
      "trail"  - Trailing stop -2ATR from peak, no target, max 120d
      "smart"  - Trail + regime exit + MA breakdown + volume collapse
    """
    atr_series = compute_atr(df)
    cm = CostModel()
    cap = 100.0; peak_cap = 100.0
    c_losses = 0; halt_idx = -1
    trade_rets = []; trade_details = []
    ma20 = indicators["ma_20"]
    ma60 = indicators["ma_60"]
    vol_ratio = indicators["vol_ratio_20d"]

    for sig in signals:
        idx = sig["index"]
        if idx < 1 or idx >= len(df) - 1: continue
        regime = regimes[idx]
        if regime in (Regime.BEAR_TREND, Regime.CRISIS): continue

        # Circuit breaker
        dd = (peak_cap - cap) / peak_cap if peak_cap > 0 else 0
        if dd > 0.15: halt_idx = idx + 22
        if idx < halt_idx: continue
        if c_losses >= 8: continue

        ep = float(df["Open"].iloc[idx + 1])
        ca = float(atr_series.iloc[idx]) if not pd.isna(atr_series.iloc[idx]) else 0
        if ca <= 0 or ep <= 0: continue

        # Position sizing (20% base)
        pos = base_pct
        sd = 2.0 * ca
        if sd > 0 and ep > 0:
            max_val = (cap * max_risk_pct) / (sd / ep)
            pos = min(pos, max_val / cap if cap > 0 else 0)

        # ── EXIT LOGIC ──
        if exit_mode == "fixed":
            xp, xr, xi = _exit_fixed(df, idx, ep, ca)
        elif exit_mode == "trail":
            xp, xr, xi = _exit_trailing(df, idx, ep, ca, atr_series)
        elif exit_mode == "smart":
            xp, xr, xi = _exit_smart(df, idx, ep, ca, atr_series,
                                      regimes, ma20, ma60, vol_ratio,
                                      sig.get("ma_period", 20))

        ee = cm.apply_entry(ep, "US")
        ex = cm.apply_exit(xp, "US")
        rp = (ex - ee) / ee * 100
        hold_days = xi - idx - 1

        cap += cap * (rp * pos / 100)
        peak_cap = max(peak_cap, cap)
        c_losses = 0 if rp > 0 else c_losses + 1
        trade_rets.append(rp)
        trade_details.append({
            "date": df.index[idx+1].strftime("%Y-%m-%d"),
            "ret": rp, "days": hold_days, "reason": xr,
        })

    m = compute_metrics(trade_rets, len(df))
    return cap, m, trade_details


def _exit_fixed(df, idx, entry_price, atr_val):
    """Current: fixed stop/target, 60d max."""
    stop = entry_price - 2.0 * atr_val
    target = entry_price + 3.0 * atr_val
    for j in range(idx + 2, min(idx + 60, len(df))):
        if float(df["Low"].iloc[j]) <= stop:
            return stop, "stop", j
        if float(df["High"].iloc[j]) >= target:
            return target, "target", j
    xi = min(idx + 59, len(df) - 1)
    return float(df["Close"].iloc[xi]), "time_60d", xi


def _exit_trailing(df, idx, entry_price, atr_val, atr_series):
    """Trailing stop: -2 ATR from highest close, no target cap, 120d max."""
    highest = entry_price
    stop = entry_price - 2.0 * atr_val
    max_hold = 120

    for j in range(idx + 2, min(idx + max_hold, len(df))):
        close_j = float(df["Close"].iloc[j])
        low_j = float(df["Low"].iloc[j])
        # Update trailing stop
        if close_j > highest:
            highest = close_j
            cur_atr = float(atr_series.iloc[j]) if not pd.isna(atr_series.iloc[j]) else atr_val
            stop = highest - 2.0 * cur_atr
        if low_j <= stop:
            return stop, "trail_stop", j
    xi = min(idx + max_hold - 1, len(df) - 1)
    return float(df["Close"].iloc[xi]), "time_120d", xi


def _exit_smart(df, idx, entry_price, atr_val, atr_series,
                regimes, ma20, ma60, vol_ratio, entry_ma_period):
    """Smart exit: trailing stop + regime change + MA breakdown + volume collapse."""
    highest = entry_price
    stop = entry_price - 2.0 * atr_val
    max_hold = 250  # ~1 year max
    low_vol_days = 0

    for j in range(idx + 2, min(idx + max_hold, len(df))):
        close_j = float(df["Close"].iloc[j])
        low_j = float(df["Low"].iloc[j])

        # 1. Trailing stop (update on new highs)
        if close_j > highest:
            highest = close_j
            cur_atr = float(atr_series.iloc[j]) if not pd.isna(atr_series.iloc[j]) else atr_val
            stop = highest - 2.0 * cur_atr
            low_vol_days = 0  # reset volume counter on new high
        if low_j <= stop:
            return stop, "trail_stop", j

        # 2. Regime change: Bull/Neutral → Bear/Crisis = exit
        if j < len(regimes) and regimes[j] in (Regime.BEAR_TREND, Regime.CRISIS):
            # Only exit if we've been in the trade > 5 days (avoid whipsaw)
            if j - idx > 5:
                return close_j, "regime_exit", j

        # 3. MA breakdown: price closes below entry MA for 2 consecutive days
        if entry_ma_period == 20 and not pd.isna(ma20.iloc[j]):
            if close_j < float(ma20.iloc[j]):
                # Check if previous day also below
                if j > idx + 2 and not pd.isna(ma20.iloc[j-1]):
                    prev_close = float(df["Close"].iloc[j-1])
                    if prev_close < float(ma20.iloc[j-1]) and j - idx > 10:
                        return close_j, "ma_break", j
        elif entry_ma_period == 60 and not pd.isna(ma60.iloc[j]):
            if close_j < float(ma60.iloc[j]):
                if j > idx + 2 and not pd.isna(ma60.iloc[j-1]):
                    prev_close = float(df["Close"].iloc[j-1])
                    if prev_close < float(ma60.iloc[j-1]) and j - idx > 10:
                        return close_j, "ma_break", j

        # 4. Volume collapse: volume < 0.5x 20d avg for 5+ days
        if not pd.isna(vol_ratio.iloc[j]):
            if float(vol_ratio.iloc[j]) < 0.5:
                low_vol_days += 1
            else:
                low_vol_days = 0
            if low_vol_days >= 5 and j - idx > 15:
                return close_j, "vol_collapse", j

    xi = min(idx + max_hold - 1, len(df) - 1)
    return float(df["Close"].iloc[xi]), "time_250d", xi


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    print("Fetching VIX...")
    vix_df = fetch_ohlcv("^VIX", period="10y", interval="1d")

    symbols = ["GOOGL", "AAPL", "MSFT", "NVDA", "SPY", "QQQ",
               "AMZN", "TSLA", "JPM", "JNJ", "XOM"]
    modes = [
        ("Fixed",  "fixed"),
        ("Trail",  "trail"),
        ("Smart",  "smart"),
    ]

    print(f"\n{'Symbol':<7}", end="")
    for label, _ in modes:
        print(f"  {'$':>1}{label:>8}{'Sharpe':>8}{'Trades':>7}{'AvgDay':>7}", end="")
    print(f"  {'BuyHold':>10}")
    print("=" * 115)

    all_results = {label: [] for label, _ in modes}

    for sym in symbols:
        df = fetch_ohlcv(sym, period="10y", interval="1d")
        if df is None or df.empty: continue
        indicators = compute_all(df)
        signals = generate_signals(df, indicators)
        regimes = compute_regimes(df, vix_df)
        bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0])) * 100

        print(f"{sym:<7}", end="")
        for label, mode in modes:
            eq, m, details = run_backtest(df, signals, regimes, indicators,
                                          exit_mode=mode, base_pct=0.20, max_risk_pct=0.03)
            avg_days = np.mean([d["days"] for d in details]) if details else 0
            all_results[label].append({
                "sym": sym, "eq": eq, "sharpe": m.sharpe_ratio,
                "trades": m.total_trades, "wr": m.win_rate,
                "avg_days": avg_days, "details": details,
                "mdd": m.max_drawdown_pct, "pf": m.profit_factor,
            })
            print(f"  ${eq:>8.0f}{m.sharpe_ratio:>8.2f}{m.total_trades:>7}{avg_days:>7.0f}d", end="")
        print(f"  ${bh:>8.0f}")

    # Summary
    print(f"\n{'='*115}")
    print(f"{'SUMMARY':<7}", end="")
    for label, _ in modes:
        res = all_results[label]
        avg_eq = np.mean([r["eq"] for r in res])
        avg_s = np.mean([r["sharpe"] for r in res])
        avg_t = np.mean([r["trades"] for r in res])
        avg_d = np.mean([r["avg_days"] for r in res])
        print(f"  ${avg_eq:>8.0f}{avg_s:>8.2f}{avg_t:>7.0f}{avg_d:>7.0f}d", end="")
    print()

    print(f"{'MinEq':<7}", end="")
    for label, _ in modes:
        min_eq = min([r["eq"] for r in all_results[label]])
        min_sym = min(all_results[label], key=lambda r: r["eq"])["sym"]
        print(f"  ${min_eq:>8.0f}  ({min_sym:<4})          ", end="")
    print()

    print(f"{'MaxEq':<7}", end="")
    for label, _ in modes:
        max_eq = max([r["eq"] for r in all_results[label]])
        max_sym = max(all_results[label], key=lambda r: r["eq"])["sym"]
        print(f"  ${max_eq:>8.0f}  ({max_sym:<4})          ", end="")
    print()

    # Exit reason breakdown for Smart mode
    print(f"\n{'='*70}")
    print("Smart Exit Reason Breakdown:")
    print(f"{'='*70}")
    print(f"{'Symbol':<7} {'trail':>8} {'regime':>8} {'ma_brk':>8} {'vol_col':>8} {'time':>8}")
    print("-" * 50)
    for r in all_results["Smart"]:
        reasons = {}
        for d in r["details"]:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        print(f"{r['sym']:<7} "
              f"{reasons.get('trail_stop', 0):>8} "
              f"{reasons.get('regime_exit', 0):>8} "
              f"{reasons.get('ma_break', 0):>8} "
              f"{reasons.get('vol_collapse', 0):>8} "
              f"{reasons.get('time_250d', 0):>8}")

    # Hold duration comparison
    print(f"\n{'='*70}")
    print("Average Hold Duration (days):")
    print(f"{'='*70}")
    print(f"{'Symbol':<7} {'Fixed':>10} {'Trail':>10} {'Smart':>10}")
    print("-" * 40)
    for i, sym in enumerate(symbols):
        if i >= len(all_results["Fixed"]): break
        print(f"{sym:<7} "
              f"{all_results['Fixed'][i]['avg_days']:>9.0f}d "
              f"{all_results['Trail'][i]['avg_days']:>9.0f}d "
              f"{all_results['Smart'][i]['avg_days']:>9.0f}d")


if __name__ == "__main__":
    main()
