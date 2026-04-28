#!/usr/bin/env python3
"""End-to-end backtest: fetch data -> generate signals -> run backtest -> validate.

Tests MA crossover + RSI + volume strategies on real historical data
for both US and TW markets.
"""
import sys
import numpy as np
import pandas as pd

from quantforge.data.fetch_us import fetch_ohlcv
from quantforge.analysis.indicators import compute_all, compute_ma
from quantforge.backtest.engine import BacktestEngine
from quantforge.backtest.cost_model import CostModel
from quantforge.backtest.validation import WalkForwardValidator, MonteCarloAnalyzer


def generate_signals(df: pd.DataFrame, indicators: dict,
                     ma_periods: list[int] = None,
                     rsi_oversold: float = 30,
                     volume_spike: float = 1.5) -> list[dict]:
    """Generate backtest signals from indicator crossovers.

    Converts rule-based signal detection into the {index, direction, score, type}
    format that BacktestEngine expects.
    """
    if ma_periods is None:
        ma_periods = [20, 60]

    close = df["Close"]
    signals = []

    for i in range(1, len(df)):
        # --- MA Crossover signals ---
        for period in ma_periods:
            ma_key = f"ma_{period}"
            if ma_key not in indicators:
                continue
            ma = indicators[ma_key]
            if pd.isna(ma.iloc[i]) or pd.isna(ma.iloc[i - 1]):
                continue

            # Bullish crossover: price crosses above MA
            if close.iloc[i - 1] <= ma.iloc[i - 1] and close.iloc[i] > ma.iloc[i]:
                # Score based on RSI confirmation + volume
                score = 70.0
                rsi_val = indicators["rsi"].iloc[i]
                vol_r = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(rsi_val) and rsi_val < 50:
                    score += 10  # oversold confirmation
                if not pd.isna(vol_r) and vol_r > volume_spike:
                    score += 10  # volume confirmation
                signals.append({
                    "index": i, "direction": "long",
                    "score": score, "type": f"MA{period} crossover up",
                })

        # --- RSI oversold bounce ---
        rsi = indicators["rsi"]
        if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i - 1]):
            if rsi.iloc[i - 1] <= rsi_oversold and rsi.iloc[i] > rsi_oversold:
                # RSI bouncing out of oversold
                score = 65.0
                vol_r = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(vol_r) and vol_r > volume_spike:
                    score += 15
                # Check if MACD histogram turning positive
                hist = indicators["macd_hist"]
                if not pd.isna(hist.iloc[i]) and hist.iloc[i] > hist.iloc[max(0, i - 1)]:
                    score += 10
                signals.append({
                    "index": i, "direction": "long",
                    "score": score, "type": "RSI oversold bounce",
                })

    return signals


def run_single_backtest(symbol: str, period: str, market: str,
                        ma_periods: list[int],
                        atr_stop: float = 2.0, atr_target: float = 3.0):
    """Run full backtest pipeline for a single symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol} | {period} | MA{ma_periods} | Market: {market}")
    print(f"  ATR Stop: {atr_stop}x | ATR Target: {atr_target}x")
    print(f"{'='*70}")

    # Fetch data
    print(f"Fetching {symbol} data ({period})...", end=" ")
    df = fetch_ohlcv(symbol, period=period, interval="1d")
    if df is None or df.empty:
        print("FAILED - no data")
        return None
    print(f"{len(df)} days loaded ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

    # Compute indicators
    indicators = compute_all(df)

    # Generate signals
    signals = generate_signals(df, indicators, ma_periods=ma_periods)
    print(f"Signals generated: {len(signals)}")
    if not signals:
        print("  No signals found - skipping")
        return None

    # Count by type
    type_counts = {}
    for s in signals:
        t = s["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  - {t}: {c}")

    # Run backtest
    cost_model = CostModel()
    engine = BacktestEngine(
        market=market, cost_model=cost_model,
        atr_stop_mult=atr_stop, atr_target_mult=atr_target,
    )
    metrics = engine.run(df, signals)

    # Print results
    print(f"\n--- Backtest Results ---")
    print(f"  Total trades:     {metrics.total_trades}")
    print(f"  Win rate:         {metrics.win_rate:.1f}%")
    print(f"  Avg return/trade: {metrics.avg_return_pct:+.2f}%")
    print(f"  Total return:     {metrics.total_return_pct:+.2f}%")
    print(f"  Annualized:       {metrics.annualized_return_pct:+.2f}%")
    print(f"  Sharpe ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown:     {metrics.max_drawdown_pct:.2f}%")
    print(f"  Profit factor:    {metrics.profit_factor:.2f}")
    print(f"  Avg win:          {metrics.avg_win_pct:+.2f}%")
    print(f"  Avg loss:         {metrics.avg_loss_pct:+.2f}%")
    print(f"  Verdict:          {metrics.verdict}")

    # Walk-forward validation
    if metrics.total_trades >= 10:
        # Reconstruct per-trade returns for validation
        trade_returns = _extract_trade_returns(df, signals, engine)
        if len(trade_returns) >= 10:
            print(f"\n--- Walk-Forward Validation ---")
            wf = WalkForwardValidator()
            wf_result = wf.validate(trade_returns, window_size=max(6, len(trade_returns) // 4))
            print(f"  Windows: {wf_result['total_windows']}")
            print(f"  Profitable: {wf_result['profitable_windows']}")
            print(f"  Pass rate: {wf_result.get('pass_rate', 'N/A')}")
            print(f"  WF Verdict: {wf_result['verdict']}")

            # Monte Carlo
            print(f"\n--- Monte Carlo Analysis (1000 sims) ---")
            mc = MonteCarloAnalyzer(n_simulations=1000)
            mc_result = mc.analyze(trade_returns)
            print(f"  Median drawdown:  {mc_result['median_drawdown']:.2f}%")
            print(f"  P95 drawdown:     {mc_result['p95_drawdown']:.2f}%")
            print(f"  Worst drawdown:   {mc_result['worst_drawdown']:.2f}%")

    return metrics


def _extract_trade_returns(df, signals, engine):
    """Re-extract per-trade return list (mirrors engine logic)."""
    from quantforge.analysis.indicators import compute_atr
    atr = compute_atr(df)
    returns = []
    for sig in signals:
        idx = sig["index"]
        if idx < 1 or idx >= len(df) - 1:
            continue
        entry_price = float(df["Open"].iloc[idx + 1])
        current_atr = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0
        if current_atr <= 0 or entry_price <= 0:
            continue
        effective_entry = engine.costs.apply_entry(entry_price, engine.market)
        stop = entry_price - engine.atr_stop_mult * current_atr
        target = entry_price + engine.atr_target_mult * current_atr
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
        effective_exit = engine.costs.apply_exit(exit_price, engine.market)
        returns.append((effective_exit - effective_entry) / effective_entry * 100)
    return returns


def main():
    np.random.seed(42)

    print("=" * 70)
    print("  QuantForge Backtest — Strategy Edge Validation")
    print("=" * 70)

    # ---- Test matrix: multiple symbols, periods, MA configs ----
    test_configs = [
        # (symbol, period, market, ma_periods, atr_stop, atr_target)
        ("GOOGL", "10y", "US", [20, 60], 2.0, 3.0),
        ("AAPL",  "10y", "US", [20, 60], 2.0, 3.0),
        ("MSFT",  "10y", "US", [20, 60], 2.0, 3.0),
        ("NVDA",  "5y",  "US", [20, 60], 2.0, 3.0),
        ("SPY",   "10y", "US", [20, 60], 2.0, 3.0),  # benchmark ETF
        ("QQQ",   "10y", "US", [20, 60], 2.0, 3.0),  # tech ETF
        # Tighter stops
        ("GOOGL", "10y", "US", [20, 60], 1.5, 2.5),
        # Wider stops
        ("GOOGL", "10y", "US", [20, 60], 3.0, 4.0),
        # MA20 only
        ("GOOGL", "10y", "US", [20],     2.0, 3.0),
        # MA60 only
        ("GOOGL", "10y", "US", [60],     2.0, 3.0),
    ]

    all_results = []
    for symbol, period, market, ma_periods, atr_stop, atr_target in test_configs:
        result = run_single_backtest(
            symbol, period, market, ma_periods, atr_stop, atr_target,
        )
        if result:
            all_results.append({
                "symbol": symbol, "period": period,
                "ma": str(ma_periods),
                "atr_s": atr_stop, "atr_t": atr_target,
                "trades": result.total_trades,
                "win%": result.win_rate,
                "ret%": result.total_return_pct,
                "ann%": result.annualized_return_pct,
                "sharpe": result.sharpe_ratio,
                "mdd%": result.max_drawdown_pct,
                "pf": result.profit_factor,
                "verdict": result.verdict,
            })

    # Summary table
    if all_results:
        print(f"\n\n{'='*100}")
        print("  SUMMARY TABLE")
        print(f"{'='*100}")
        header = f"{'Symbol':<7} {'Period':<5} {'MA':<10} {'S/T':>5} {'Trades':>6} {'Win%':>6} {'Ret%':>8} {'Ann%':>8} {'Sharpe':>7} {'MDD%':>7} {'PF':>6} {'Verdict':<12}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(f"{r['symbol']:<7} {r['period']:<5} {r['ma']:<10} {r['atr_s']}/{r['atr_t']:>3} "
                  f"{r['trades']:>6} {r['win%']:>5.1f}% {r['ret%']:>+7.1f}% {r['ann%']:>+7.1f}% "
                  f"{r['sharpe']:>7.2f} {r['mdd%']:>6.1f}% {r['pf']:>5.2f} {r['verdict']:<12}")

        # Edge assessment
        print(f"\n{'='*70}")
        print("  EDGE ASSESSMENT")
        print(f"{'='*70}")
        valid = [r for r in all_results if r["verdict"] == "VALID"]
        marginal = [r for r in all_results if r["verdict"] == "MARGINAL"]
        rejected = [r for r in all_results if r["verdict"] in ("REJECT", "INSUFFICIENT")]
        print(f"  VALID:        {len(valid)}/{len(all_results)}")
        print(f"  MARGINAL:     {len(marginal)}/{len(all_results)}")
        print(f"  REJECT/INSUF: {len(rejected)}/{len(all_results)}")

        avg_sharpe = np.mean([r["sharpe"] for r in all_results])
        avg_win = np.mean([r["win%"] for r in all_results])
        avg_pf = np.mean([r["pf"] for r in all_results])
        print(f"\n  Avg Sharpe:   {avg_sharpe:.2f}  {'(>1.0 = good)' if avg_sharpe > 1 else '(<1.0 = weak)'}")
        print(f"  Avg Win Rate: {avg_win:.1f}%  {'(>50% = good)' if avg_win > 50 else '(<50% = weak)'}")
        print(f"  Avg PF:       {avg_pf:.2f}  {'(>1.5 = good)' if avg_pf > 1.5 else '(<1.5 = weak)'}")

        if avg_sharpe >= 1.0 and avg_win >= 50 and avg_pf >= 1.5:
            print(f"\n  >>> CONCLUSION: Strategy shows viable edge across test matrix <<<")
        elif avg_sharpe >= 0.5 or avg_pf >= 1.2:
            print(f"\n  >>> CONCLUSION: Marginal edge — needs refinement or filtering (try LLM filter) <<<")
        else:
            print(f"\n  >>> CONCLUSION: No reliable edge detected — strategy needs redesign <<<")


if __name__ == "__main__":
    main()
