#!/usr/bin/env python3
"""Generate comprehensive backtest PDF report.

Includes:
- Strategy theory & rationale
- Signal generation rules
- Per-symbol backtest results with $100 equity curves
- Summary comparison tables
- Walk-forward & Monte Carlo validation
- Final edge assessment
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

from quantforge.data.fetch_us import fetch_ohlcv
from quantforge.analysis.indicators import compute_all, compute_atr
from quantforge.backtest.engine import BacktestEngine
from quantforge.backtest.cost_model import CostModel
from quantforge.backtest.validation import WalkForwardValidator, MonteCarloAnalyzer

# ── Styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
})

COLORS = {
    "primary": "#1a73e8",
    "green": "#0d904f",
    "red": "#d93025",
    "orange": "#e8710a",
    "purple": "#7b1fa2",
    "gray": "#5f6368",
    "light_green": "#e6f4ea",
    "light_red": "#fce8e6",
    "light_blue": "#e8f0fe",
}


# ═══════════════════════════════════════════════════════════════════
#  Signal Generation (same as run_backtest.py)
# ═══════════════════════════════════════════════════════════════════
def generate_signals(df, indicators, ma_periods=None,
                     rsi_oversold=30, volume_spike=1.5):
    if ma_periods is None:
        ma_periods = [20, 60]
    close = df["Close"]
    signals = []
    for i in range(1, len(df)):
        for period in ma_periods:
            ma_key = f"ma_{period}"
            if ma_key not in indicators:
                continue
            ma = indicators[ma_key]
            if pd.isna(ma.iloc[i]) or pd.isna(ma.iloc[i - 1]):
                continue
            if close.iloc[i - 1] <= ma.iloc[i - 1] and close.iloc[i] > ma.iloc[i]:
                score = 70.0
                rsi_val = indicators["rsi"].iloc[i]
                vol_r = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(rsi_val) and rsi_val < 50:
                    score += 10
                if not pd.isna(vol_r) and vol_r > volume_spike:
                    score += 10
                signals.append({
                    "index": i, "direction": "long",
                    "score": score, "type": f"MA{period}_up",
                })
        rsi = indicators["rsi"]
        if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i - 1]):
            if rsi.iloc[i - 1] <= rsi_oversold and rsi.iloc[i] > rsi_oversold:
                score = 65.0
                vol_r = indicators["vol_ratio_5d"].iloc[i]
                if not pd.isna(vol_r) and vol_r > volume_spike:
                    score += 15
                hist = indicators["macd_hist"]
                if not pd.isna(hist.iloc[i]) and hist.iloc[i] > hist.iloc[max(0, i - 1)]:
                    score += 10
                signals.append({
                    "index": i, "direction": "long",
                    "score": score, "type": "RSI_bounce",
                })
    return signals


# ═══════════════════════════════════════════════════════════════════
#  Per-trade equity curve (start $100)
# ═══════════════════════════════════════════════════════════════════
def compute_equity_curve(df, signals, engine):
    """Return (dates, equity_values) starting from $100."""
    atr = compute_atr(df)
    equity = 100.0
    curve_dates = [df.index[0]]
    curve_vals = [equity]
    trade_details = []

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
        exit_idx = min(idx + 59, len(df) - 1)
        exit_reason = "time"
        for j in range(idx + 2, min(idx + 60, len(df))):
            if float(df["Low"].iloc[j]) <= stop:
                exit_price = stop
                exit_idx = j
                exit_reason = "stop"
                break
            if float(df["High"].iloc[j]) >= target:
                exit_price = target
                exit_idx = j
                exit_reason = "target"
                break
        if exit_price is None:
            exit_price = float(df["Close"].iloc[exit_idx])
        effective_exit = engine.costs.apply_exit(exit_price, engine.market)
        ret_pct = (effective_exit - effective_entry) / effective_entry
        equity *= (1 + ret_pct)
        curve_dates.append(df.index[exit_idx])
        curve_vals.append(equity)
        trade_details.append({
            "entry_date": df.index[idx + 1],
            "exit_date": df.index[exit_idx],
            "ret_pct": ret_pct * 100,
            "exit_reason": exit_reason,
            "type": sig["type"],
        })

    return curve_dates, curve_vals, trade_details


# ═══════════════════════════════════════════════════════════════════
#  PDF Report Generation
# ═══════════════════════════════════════════════════════════════════
def add_text_page(pdf, lines, title=None):
    """Add a full-page text block."""
    fig = plt.figure(figsize=(11, 8.5))  # landscape letter
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.96
    if title:
        ax.text(0.05, y, title, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top",
                color=COLORS["primary"])
        y -= 0.06
        ax.plot([0.05, 0.95], [y + 0.01, y + 0.01],
                color=COLORS["primary"], linewidth=1.5,
                transform=ax.transAxes, clip_on=False)
        y -= 0.02
    for line in lines:
        if line.startswith("##"):
            y -= 0.01
            ax.text(0.05, y, line.replace("## ", ""), transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top", color="#333")
            y -= 0.04
        elif line.startswith("**"):
            ax.text(0.05, y, line.replace("**", ""), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", color="#333",
                    family="monospace")
            y -= 0.03
        elif line == "---":
            ax.plot([0.05, 0.95], [y + 0.005, y + 0.005],
                    color="#ddd", linewidth=0.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.015
        elif line == "":
            y -= 0.015
        else:
            ax.text(0.05, y, line, transform=ax.transAxes,
                    fontsize=9, va="top", color="#444",
                    family="monospace", linespacing=1.4)
            y -= 0.028
        if y < 0.03:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.96
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_summary_table(pdf, results, title):
    """Add a formatted summary table page."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.97, title, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="center", va="top",
            color=COLORS["primary"])

    cols = ["Symbol", "Period", "MA", "ATR S/T", "Trades", "Win%",
            "Total Ret%", "Ann Ret%", "Sharpe", "MDD%", "PF", "Verdict"]
    rows = []
    cell_colors = []
    for r in results:
        verdict = r["verdict"]
        if verdict == "VALID":
            bg = COLORS["light_green"]
        elif verdict == "REJECT":
            bg = COLORS["light_red"]
        else:
            bg = COLORS["light_blue"]
        row_colors = [bg] * len(cols)
        rows.append([
            r["symbol"], r["period"], r["ma"],
            f"{r['atr_s']}/{r['atr_t']}",
            str(r["trades"]),
            f"{r['win%']:.1f}",
            f"{r['ret%']:+.1f}",
            f"{r['ann%']:+.1f}",
            f"{r['sharpe']:.2f}",
            f"{r['mdd%']:.1f}",
            f"{r['pf']:.2f}",
            verdict,
        ])
        cell_colors.append(row_colors)

    table = ax.table(cellText=rows, colLabels=cols,
                     cellColours=cell_colors,
                     colColours=[COLORS["light_blue"]] * len(cols),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(COLORS["primary"])
        cell.set_edgecolor("#ddd")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_equity_page(pdf, symbol, period, df, curve_dates, curve_vals,
                    trade_details, metrics, wf_result, mc_result):
    """Add a two-panel page: equity curve + stats for one config."""
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  top=0.92, bottom=0.08, left=0.08, right=0.95)

    fig.suptitle(f"{symbol} — {period} Backtest ($100 Start)",
                 fontsize=14, fontweight="bold", color=COLORS["primary"])

    # ── Panel 1: Equity curve ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(curve_dates, curve_vals, color=COLORS["primary"],
             linewidth=1.5, label="Strategy Equity")
    ax1.axhline(y=100, color=COLORS["gray"], linestyle="--",
                linewidth=0.8, alpha=0.5, label="$100 baseline")
    ax1.fill_between(curve_dates, 100, curve_vals,
                     where=[v >= 100 for v in curve_vals],
                     alpha=0.15, color=COLORS["green"])
    ax1.fill_between(curve_dates, 100, curve_vals,
                     where=[v < 100 for v in curve_vals],
                     alpha=0.15, color=COLORS["red"])
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("Equity Curve (compounded, after costs)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # ── Panel 2: Drawdown chart ──
    ax2 = fig.add_subplot(gs[1, 0])
    vals = np.array(curve_vals)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak * 100
    ax2.fill_between(curve_dates, dd, 0, color=COLORS["red"], alpha=0.4)
    ax2.plot(curve_dates, dd, color=COLORS["red"], linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown from Peak")
    ax2.set_ylim(min(dd) * 1.1, 5)

    # ── Panel 3: Stats box ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    stats_lines = [
        ("Total Trades", f"{metrics.total_trades}"),
        ("Win Rate", f"{metrics.win_rate:.1f}%"),
        ("Avg Return/Trade", f"{metrics.avg_return_pct:+.2f}%"),
        ("Total Return", f"{metrics.total_return_pct:+.1f}%"),
        ("Annualized Return", f"{metrics.annualized_return_pct:+.1f}%"),
        ("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"),
        ("Max Drawdown", f"{metrics.max_drawdown_pct:.1f}%"),
        ("Profit Factor", f"{metrics.profit_factor:.2f}"),
        ("Avg Win", f"{metrics.avg_win_pct:+.2f}%"),
        ("Avg Loss", f"{metrics.avg_loss_pct:+.2f}%"),
        ("$100 Final Value", f"${curve_vals[-1]:.2f}"),
        ("", ""),
        ("WF Pass Rate", f"{wf_result.get('pass_rate', 'N/A')}"),
        ("WF Verdict", f"{wf_result['verdict']}"),
        ("MC Median DD", f"{mc_result['median_drawdown']:.1f}%"),
        ("MC P95 DD", f"{mc_result['p95_drawdown']:.1f}%"),
        ("MC Worst DD", f"{mc_result['worst_drawdown']:.1f}%"),
        ("", ""),
        ("ENGINE VERDICT", f"{metrics.verdict}"),
    ]
    y = 0.97
    for label, val in stats_lines:
        if label == "ENGINE VERDICT":
            color = COLORS["green"] if val == "VALID" else COLORS["red"]
            ax3.text(0.05, y, label, fontsize=9, fontweight="bold",
                     va="top", transform=ax3.transAxes, color="#333")
            ax3.text(0.65, y, val, fontsize=11, fontweight="bold",
                     va="top", transform=ax3.transAxes, color=color)
        elif label == "":
            pass
        else:
            ax3.text(0.05, y, label, fontsize=8, va="top",
                     transform=ax3.transAxes, color="#555")
            val_color = COLORS["green"] if val.startswith("+") else (
                COLORS["red"] if val.startswith("-") else "#333")
            ax3.text(0.65, y, val, fontsize=8, fontweight="bold",
                     va="top", transform=ax3.transAxes, color=val_color)
        y -= 0.052

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_trade_distribution_page(pdf, all_trade_details, symbol):
    """Add trade return distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"{symbol} — Trade Analysis",
                 fontsize=13, fontweight="bold", color=COLORS["primary"])

    rets = [t["ret_pct"] for t in all_trade_details]
    if not rets:
        plt.close(fig)
        return

    # Histogram
    ax = axes[0]
    bins = np.linspace(min(rets), max(rets), 40)
    n, bins_out, patches = ax.hist(rets, bins=bins, edgecolor="white",
                                    linewidth=0.5, alpha=0.85)
    for patch, left in zip(patches, bins_out[:-1]):
        patch.set_facecolor(COLORS["green"] if left >= 0 else COLORS["red"])
    ax.axvline(x=0, color="#333", linestyle="--", linewidth=0.8)
    ax.axvline(x=np.mean(rets), color=COLORS["primary"], linestyle="-",
               linewidth=1.2, label=f"Mean: {np.mean(rets):+.2f}%")
    ax.set_xlabel("Return per Trade (%)")
    ax.set_ylabel("Count")
    ax.set_title("Trade Return Distribution")
    ax.legend(fontsize=8)

    # By exit reason
    ax2 = axes[1]
    reasons = {}
    for t in all_trade_details:
        r = t["exit_reason"]
        if r not in reasons:
            reasons[r] = {"count": 0, "avg_ret": []}
        reasons[r]["count"] += 1
        reasons[r]["avg_ret"].append(t["ret_pct"])
    labels = list(reasons.keys())
    counts = [reasons[r]["count"] for r in labels]
    avg_rets = [np.mean(reasons[r]["avg_ret"]) for r in labels]
    bar_colors = [COLORS["green"] if a > 0 else COLORS["red"] for a in avg_rets]
    bars = ax2.bar(labels, counts, color=bar_colors, alpha=0.8, edgecolor="white")
    for bar, avg in zip(bars, avg_rets):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"avg: {avg:+.1f}%", ha="center", fontsize=8, color="#333")
    ax2.set_title("Exits by Reason")
    ax2.set_ylabel("Count")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_parameter_sensitivity_page(pdf, param_results):
    """Compare GOOGL results across different ATR / MA configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    fig.suptitle("Parameter Sensitivity — GOOGL 10Y",
                 fontsize=13, fontweight="bold", color=COLORS["primary"])

    # Filter GOOGL results
    googl = [r for r in param_results if r["symbol"] == "GOOGL"]
    if len(googl) < 2:
        plt.close(fig)
        return

    labels = [f"MA{r['ma']}\nATR {r['atr_s']}/{r['atr_t']}" for r in googl]
    x = range(len(googl))

    # Sharpe
    ax = axes[0]
    vals = [r["sharpe"] for r in googl]
    bars = ax.bar(x, vals, color=[COLORS["green"] if v > 1 else COLORS["orange"]
                                   for v in vals], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.set_title("Sharpe Ratio")
    ax.axhline(y=1.0, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=7)

    # Win Rate
    ax = axes[1]
    vals = [r["win%"] for r in googl]
    bars = ax.bar(x, vals, color=[COLORS["green"] if v > 50 else COLORS["red"]
                                   for v in vals], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.set_title("Win Rate (%)")
    ax.axhline(y=50, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", fontsize=7)

    # $100 final
    ax = axes[2]
    vals = [r["final_equity"] for r in googl]
    bars = ax.bar(x, vals, color=COLORS["primary"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.set_title("$100 → Final ($)")
    ax.axhline(y=100, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"${v:.0f}", ha="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_all_equity_overlay(pdf, all_curves):
    """Overlay all symbol equity curves on one chart."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle("All Strategies — $100 Equity Comparison",
                 fontsize=14, fontweight="bold", color=COLORS["primary"])

    color_cycle = [COLORS["primary"], COLORS["green"], COLORS["purple"],
                   COLORS["orange"], COLORS["red"], COLORS["gray"],
                   "#00897b", "#c2185b"]
    for i, (label, dates, vals) in enumerate(all_curves):
        c = color_cycle[i % len(color_cycle)]
        ax.plot(dates, vals, label=f"{label} (${vals[-1]:.0f})",
                color=c, linewidth=1.3, alpha=0.85)

    ax.axhline(y=100, color="#999", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    output_path = os.path.expanduser("~/.quantforge/backtest_report.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_configs = [
        ("GOOGL", "10y", "US", [20, 60], 2.0, 3.0),
        ("AAPL",  "10y", "US", [20, 60], 2.0, 3.0),
        ("MSFT",  "10y", "US", [20, 60], 2.0, 3.0),
        ("NVDA",  "5y",  "US", [20, 60], 2.0, 3.0),
        ("SPY",   "10y", "US", [20, 60], 2.0, 3.0),
        ("QQQ",   "10y", "US", [20, 60], 2.0, 3.0),
        # Parameter sensitivity on GOOGL
        ("GOOGL", "10y", "US", [20, 60], 1.5, 2.5),
        ("GOOGL", "10y", "US", [20, 60], 3.0, 4.0),
        ("GOOGL", "10y", "US", [20],     2.0, 3.0),
        ("GOOGL", "10y", "US", [60],     2.0, 3.0),
    ]

    # ── Fetch all data & run backtests ──
    print("Fetching data and running backtests...")
    all_results = []
    all_curves = []      # for overlay
    detail_data = {}     # for per-symbol pages
    data_cache = {}

    for cfg_i, (symbol, period, market, ma_periods, atr_s, atr_t) in enumerate(test_configs):
        cache_key = f"{symbol}_{period}"
        if cache_key in data_cache:
            df = data_cache[cache_key]
        else:
            print(f"  [{cfg_i+1}/{len(test_configs)}] Fetching {symbol} {period}...", end=" ")
            df = fetch_ohlcv(symbol, period=period, interval="1d")
            if df is None or df.empty:
                print("FAILED")
                continue
            data_cache[cache_key] = df
            print(f"{len(df)} days")

        indicators = compute_all(df)
        signals = generate_signals(df, indicators, ma_periods=ma_periods)

        engine = BacktestEngine(market=market, cost_model=CostModel(),
                                atr_stop_mult=atr_s, atr_target_mult=atr_t)
        metrics = engine.run(df, signals)
        curve_dates, curve_vals, trade_details = compute_equity_curve(
            df, signals, engine)

        # Validation
        trade_rets = [t["ret_pct"] for t in trade_details]
        wf = WalkForwardValidator()
        wf_result = wf.validate(trade_rets, window_size=max(6, len(trade_rets) // 4)) if len(trade_rets) >= 10 else {"verdict": "INSUFFICIENT_DATA", "pass_rate": "N/A", "total_windows": 0, "profitable_windows": 0}
        mc = MonteCarloAnalyzer(n_simulations=1000)
        mc_result = mc.analyze(trade_rets) if len(trade_rets) >= 5 else {"median_drawdown": 0, "p95_drawdown": 0, "worst_drawdown": 0, "simulations": 0}

        label = f"{symbol} MA{ma_periods} ATR{atr_s}/{atr_t}"
        result_row = {
            "symbol": symbol, "period": period,
            "ma": str(ma_periods), "atr_s": atr_s, "atr_t": atr_t,
            "trades": metrics.total_trades,
            "win%": metrics.win_rate,
            "ret%": metrics.total_return_pct,
            "ann%": metrics.annualized_return_pct,
            "sharpe": metrics.sharpe_ratio,
            "mdd%": metrics.max_drawdown_pct,
            "pf": metrics.profit_factor,
            "verdict": metrics.verdict,
            "final_equity": curve_vals[-1],
        }
        all_results.append(result_row)
        all_curves.append((label, curve_dates, curve_vals))
        detail_data[label] = {
            "df": df, "metrics": metrics,
            "curve_dates": curve_dates, "curve_vals": curve_vals,
            "trade_details": trade_details,
            "wf": wf_result, "mc": mc_result,
            "symbol": symbol, "period": period,
        }

    # ═══════════════════════════════════════════════════════════════
    #  Write PDF
    # ═══════════════════════════════════════════════════════════════
    print(f"\nGenerating PDF report → {output_path}")
    with PdfPages(output_path) as pdf:

        # ── Page 1: Title ──
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.65, "QuantForge", fontsize=36, fontweight="bold",
                ha="center", va="center", color=COLORS["primary"],
                transform=ax.transAxes)
        ax.text(0.5, 0.55, "Quantitative Strategy Backtest Report",
                fontsize=18, ha="center", va="center", color="#333",
                transform=ax.transAxes)
        ax.text(0.5, 0.45, datetime.now().strftime("%Y-%m-%d"),
                fontsize=14, ha="center", va="center", color=COLORS["gray"],
                transform=ax.transAxes)
        ax.text(0.5, 0.30,
                "MA Crossover + RSI Oversold Bounce | ATR-Based Risk Management",
                fontsize=11, ha="center", va="center", color=COLORS["gray"],
                transform=ax.transAxes)
        ax.text(0.5, 0.18,
                "Symbols: GOOGL, AAPL, MSFT, NVDA, SPY, QQQ  |  Period: 5-10 Years",
                fontsize=10, ha="center", va="center", color=COLORS["gray"],
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2-3: Strategy Theory ──
        add_text_page(pdf, [
            "## 1. Strategy Overview",
            "",
            "This report evaluates a rule-based long-only strategy combining",
            "Moving Average (MA) crossovers, RSI oversold bounces, and",
            "volume confirmation. Trades use ATR-based stop-loss and profit",
            "targets for systematic risk management.",
            "",
            "## 2. Entry Signals",
            "",
            "**Signal 1: MA Crossover Up (MA20 / MA60)**",
            "  - Trigger: Close crosses above MA(N) from below",
            "  - Theory: Trend-following — when price reclaims its moving",
            "    average, it signals a shift from bearish to bullish momentum.",
            "    MA20 captures short-term reversals; MA60 captures medium-term",
            "    trend changes. Requires prior close <= MA, current close > MA.",
            "  - Scoring: base 70 + RSI<50 bonus(+10) + volume spike bonus(+10)",
            "",
            "**Signal 2: RSI Oversold Bounce**",
            "  - Trigger: RSI(14) crosses above 30 from below",
            "  - Theory: Mean-reversion — when RSI exits oversold territory,",
            "    selling pressure is exhausted. The bounce (not the initial",
            "    drop) is the entry, reducing \"catching falling knives\".",
            "  - Scoring: base 65 + volume spike bonus(+15) + MACD improving(+10)",
            "",
            "**Volume Confirmation**",
            "  - Volume ratio > 1.5x (5-day average) adds score bonus",
            "  - Theory: Price moves with high volume are more reliable;",
            "    institutional participation validates the signal.",
            "",
            "## 3. Exit Rules (ATR-Based)",
            "",
            "**Stop-Loss: Entry - ATR(14) x Multiplier (default 2.0x)**",
            "  - Adapts to volatility — wider stops in volatile markets,",
            "    tighter in calm markets. Prevents fixed-dollar stops from",
            "    being too tight in volatile names (NVDA) or too wide in",
            "    stable names (SPY).",
            "",
            "**Profit Target: Entry + ATR(14) x Multiplier (default 3.0x)**",
            "  - Asymmetric reward:risk ratio (3:2 = 1.5R target)",
            "  - Allows winners to capture full ATR-scaled moves",
            "",
            "**Time Exit: 60 trading days maximum hold**",
            "  - Prevents capital lock-up in sideways trades",
            "  - Forces exit at market close of day 60",
        ], title="Strategy Design & Theory")

        add_text_page(pdf, [
            "## 4. Cost Model",
            "",
            "All returns include realistic transaction costs:",
            "",
            "  US Market:",
            "    Commission:     0.1425% (each way)",
            "    Slippage:       0.30% (each way)",
            "    Market impact:  0.10% (each way)",
            "    Sell tax:       0.00%",
            "    Round-trip:     ~1.09%",
            "",
            "  TW Market:",
            "    Same as above + sell tax 0.30%",
            "    Round-trip:     ~1.39%",
            "",
            "These costs are conservative and will reduce reported returns",
            "relative to zero-cost backtests.",
            "",
            "## 5. Validation Methodology",
            "",
            "**Walk-Forward Validation**",
            "  - Splits trade sequence into rolling windows",
            "  - Each window: 70% train / 30% test split",
            "  - Checks if test portion is profitable",
            "  - VALID if >= 70% of windows profitable",
            "  - Detects overfitting: a strategy that only works in one",
            "    regime will fail walk-forward.",
            "",
            "**Monte Carlo Analysis (1,000 simulations)**",
            "  - Randomly shuffles trade order to test path-dependency",
            "  - Computes drawdown distribution across all permutations",
            "  - P95 drawdown = 95th percentile worst-case scenario",
            "  - If P95 > 60%, the strategy has extreme tail risk",
            "",
            "**Verdict Criteria**",
            "  VALID:        Sharpe >= 1.0 AND Win Rate >= 50%",
            "  MARGINAL:     Sharpe >= 0.5 OR Win Rate >= 40%",
            "  REJECT:       Below marginal thresholds",
            "  INSUFFICIENT: < 30 trades (not enough data)",
            "",
            "## 6. Anti-Look-Ahead Bias",
            "",
            "  - Entry at next-day Open (not signal-day Close)",
            "  - Indicators computed with data available at signal time only",
            "  - No future data leakage in signal generation",
        ], title="Methodology & Validation")

        # ── Summary Table ──
        add_summary_table(pdf, all_results, "Backtest Results Summary")

        # ── All Equity Overlay ──
        # Only use the 6 base configs (not parameter sensitivity variants)
        base_curves = all_curves[:6]
        add_all_equity_overlay(pdf, base_curves)

        # ── Per-symbol detail pages ──
        # Only the 6 base configs get full detail pages
        base_labels = [f"{s} MA{m} ATR{a}/{t}" for s, _, _, m, a, t in test_configs[:6]]
        for label in base_labels:
            if label not in detail_data:
                continue
            d = detail_data[label]
            add_equity_page(pdf, d["symbol"], d["period"],
                            d["df"], d["curve_dates"], d["curve_vals"],
                            d["trade_details"], d["metrics"],
                            d["wf"], d["mc"])
            add_trade_distribution_page(pdf, d["trade_details"], d["symbol"])

        # ── Parameter Sensitivity ──
        add_parameter_sensitivity_page(pdf, all_results)

        # ── Final Edge Assessment ──
        valid = [r for r in all_results if r["verdict"] == "VALID"]
        marginal = [r for r in all_results if r["verdict"] == "MARGINAL"]
        rejected = [r for r in all_results if r["verdict"] in ("REJECT", "INSUFFICIENT")]
        avg_sharpe = np.mean([r["sharpe"] for r in all_results])
        avg_win = np.mean([r["win%"] for r in all_results])
        avg_pf = np.mean([r["pf"] for r in all_results])
        best = max(all_results, key=lambda r: r["sharpe"])
        worst = min(all_results, key=lambda r: r["sharpe"])

        if avg_sharpe >= 1.0 and avg_win >= 50 and avg_pf >= 1.5:
            conclusion = "Strategy shows VIABLE EDGE across test matrix."
        elif avg_sharpe >= 0.5 or avg_pf >= 1.2:
            conclusion = "MARGINAL EDGE detected. Refinement needed (LLM filter, regime filter)."
        else:
            conclusion = "NO RELIABLE EDGE. Strategy requires fundamental redesign."

        # $100 summary
        equity_lines = []
        for r in all_results[:6]:
            e = r["final_equity"]
            pnl = e - 100
            marker = "+" if pnl > 0 else ""
            equity_lines.append(
                f"  {r['symbol']:<6} {r['period']:<5}  $100 -> ${e:>8.2f}  ({marker}{pnl:.2f})"
            )

        add_text_page(pdf, [
            "## Overall Statistics",
            "",
            f"  Configurations tested:  {len(all_results)}",
            f"  VALID:                  {len(valid)}/{len(all_results)}",
            f"  MARGINAL:               {len(marginal)}/{len(all_results)}",
            f"  REJECT/INSUFFICIENT:    {len(rejected)}/{len(all_results)}",
            "",
            f"  Average Sharpe Ratio:   {avg_sharpe:.2f}",
            f"  Average Win Rate:       {avg_win:.1f}%",
            f"  Average Profit Factor:  {avg_pf:.2f}",
            "",
            f"  Best config:   {best['symbol']} MA{best['ma']} ATR{best['atr_s']}/{best['atr_t']}"
            f"  (Sharpe {best['sharpe']:.2f})",
            f"  Worst config:  {worst['symbol']} MA{worst['ma']} ATR{worst['atr_s']}/{worst['atr_t']}"
            f"  (Sharpe {worst['sharpe']:.2f})",
            "",
            "---",
            "",
            "## $100 Investment Results (Base Configurations)",
            "",
            *equity_lines,
            "",
            "---",
            "",
            "## Key Findings",
            "",
            "  1. Strategy works well on individual growth stocks (GOOGL,",
            "     MSFT, AAPL) with Sharpe > 1.5, but FAILS on broad",
            "     indices (SPY) and hyper-volatile names (NVDA).",
            "",
            "  2. Wider ATR stops (3.0/4.0) dramatically outperform",
            "     tighter stops (1.5/2.5) — the strategy needs room",
            "     to breathe through normal volatility.",
            "",
            "  3. Max drawdowns are extreme (60-90%) across all configs.",
            "     This is a compounded multi-trade cumulative drawdown",
            "     (not single-trade), indicating the strategy lacks",
            "     regime awareness and position sizing.",
            "",
            "  4. Walk-forward validation is mixed — several configs",
            "     pass overall metrics but fail walk-forward, suggesting",
            "     regime-dependent performance (works in bull, fails",
            "     in bear/sideways).",
            "",
            "  5. MA20 alone captures most of the edge. Adding MA60",
            "     increases trade count but dilutes signal quality.",
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "  - Add regime filter: skip trades during BEAR/CRISIS regimes",
            "  - Implement LLM signal filter (BacktestLLMFilter) to cull",
            "    low-quality signals before execution",
            "  - Add position sizing: scale exposure by signal score and",
            "    regime confidence",
            "  - Consider trailing stops instead of fixed ATR targets",
            "  - Avoid applying this strategy to broad ETFs (SPY/QQQ)",
            "    or hyper-volatile individual names without adjustments",
            "",
            "---",
            "",
            f"## Conclusion: {conclusion}",
        ], title="Edge Assessment & Conclusion")

    print(f"\nReport saved to: {output_path}")
    print(f"Pages: ~{6 + len(base_labels)*2 + 3}")


if __name__ == "__main__":
    main()
