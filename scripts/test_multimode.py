#!/usr/bin/env python3
"""QuantForge -- Multi-Mode Adaptive Strategy Backtest

Uses quantforge.strategy.MultiModeStrategy engine.
Compares against baseline (Mode B) and produces PDF report.
"""
import os, shutil, time
import numpy as np, pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from quantforge.data.fetch_us import fetch_ohlcv
from quantforge.analysis.indicators import compute_all, compute_atr
from quantforge.backtest.cost_model import CostModel
from quantforge.backtest.analytics import compute_metrics
from quantforge.core.models import Regime
from quantforge.strategy.multimode import (
    MultiModeStrategy, TradeRecord,
    compute_regime_series, compute_annualized_vol,
    generate_trend_signals, generate_breakout_signals,
)

# ── Style ──
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.2,
})
CL = {"pri": "#1a73e8", "grn": "#0d904f", "red": "#d93025",
      "org": "#e8710a", "pur": "#7b1fa2", "gry": "#5f6368",
      "teal": "#00897b", "pink": "#c2185b",
      "lgrn": "#e6f4ea", "lred": "#fce8e6", "lblu": "#e8f0fe"}
R_LBL = {Regime.BULL_TREND: "Bull", Regime.BEAR_TREND: "Bear",
         Regime.CONSOLIDATION: "Range", Regime.CRISIS: "Crisis",
         Regime.NEUTRAL: "Neutral"}
MODE_COLORS = {
    "TREND_FOLLOW": CL["grn"], "HIGHVOL_BREAKOUT": CL["org"],
    "WHEEL": CL["pur"], "CASH": CL["gry"],
}

SYMBOLS = [
    ("GOOGL", "10y", "Watchlist"), ("AAPL", "10y", "Watchlist"),
    ("MSFT", "10y", "Watchlist"),  ("NVDA", "10y", "Watchlist"),
    ("SPY", "10y", "Watchlist"),   ("QQQ", "10y", "Watchlist"),
    ("AMZN", "10y", "Non-WL"),     ("TSLA", "10y", "Non-WL"),
    ("JPM", "10y", "Non-WL"),      ("JNJ", "10y", "Non-WL"),
    ("XOM", "10y", "Non-WL"),
]


# ═══════════════════════════════════════════════════════════════════
#  Baseline simulation (Mode B from final report)
# ═══════════════════════════════════════════════════════════════════

def simulate_baseline(df, signals, regimes):
    """Regime filter + MA crossover + 2x ATR stop, 3x ATR target, 60d max, 20% sizing."""
    atr = compute_atr(df)
    cm = CostModel()
    cap = 100.0
    trades = []
    for sig in signals:
        idx = sig["index"]
        if idx < 1 or idx >= len(df) - 1:
            continue
        regime = regimes[idx]
        if regime in (Regime.BEAR_TREND, Regime.CRISIS):
            continue
        ep = float(df["Open"].iloc[idx + 1])
        ca = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0
        if ca <= 0 or ep <= 0:
            continue
        stop = ep - 2.0 * ca
        tgt = ep + 3.0 * ca
        xp = None
        xr = "time"
        xi = min(idx + 59, len(df) - 1)
        for j in range(idx + 2, min(idx + 60, len(df))):
            if float(df["Low"].iloc[j]) <= stop:
                xp, xr, xi = stop, "stop", j
                break
            if float(df["High"].iloc[j]) >= tgt:
                xp, xr, xi = tgt, "target", j
                break
        if xp is None:
            xp = float(df["Close"].iloc[xi])
        ee = cm.apply_entry(ep, "US")
        ex = cm.apply_exit(xp, "US")
        rp = (ex - ee) / ee * 100
        cap += cap * (rp / 100)
        rl = R_LBL.get(regime, "?")
        trades.append(TradeRecord(df.index[idx + 1], df.index[xi], ep, xp, rp,
                                  xr, sig["type"], rl, 20.0, "BASELINE"))
    return trades, cap


# ═══════════════════════════════════════════════════════════════════
#  PDF helpers
# ═══════════════════════════════════════════════════════════════════

def text_page(pdf, lines, title=None):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.96
    if title:
        ax.text(0.05, y, title, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", color=CL["pri"])
        y -= 0.05
        ax.plot([0.05, 0.95], [y + 0.005] * 2, color=CL["pri"], lw=1.5,
                transform=ax.transAxes, clip_on=False)
        y -= 0.025
    for line in lines:
        if line.startswith("##"):
            y -= 0.01
            ax.text(0.05, y, line[3:], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top", color="#333")
            y -= 0.04
        elif line.startswith("**"):
            ax.text(0.05, y, line.replace("**", ""), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", color="#333", family="monospace")
            y -= 0.03
        elif line == "---":
            ax.plot([0.05, 0.95], [y + 0.005] * 2, color="#ddd", lw=0.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.015
        elif line == "":
            y -= 0.012
        else:
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=8.5,
                    va="top", color="#444", family="monospace", linespacing=1.35)
            y -= 0.026
        if y < 0.03:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.96
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def summary_table_page(pdf, rows, title):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="center", va="top", color=CL["pri"])
    cols = ["Symbol", "Old$", "New$", "OldSharpe", "NewSharpe", "Winner", "PrimaryMode"]
    cd, td = [], []
    for r in rows:
        winner = r.get("winner", "")
        bg = CL["lgrn"] if winner == "NEW" else (CL["lred"] if winner == "OLD" else CL["lblu"])
        cd.append([bg] * len(cols))
        td.append([r["sym"], f"${r['old_eq']:.0f}", f"${r['new_eq']:.0f}",
                   f"{r['old_sharpe']:.2f}", f"{r['new_sharpe']:.2f}",
                   winner, r.get("primary_mode", "")])
    table = ax.table(cellText=td, colLabels=cols, cellColours=cd,
                     colColours=[CL["lblu"]] * len(cols), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(CL["pri"])
        cell.set_edgecolor("#ddd")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def equity_comparison_chart(pdf, sym, old_trades, old_eq, new_curve, df, mode_log):
    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.35, top=0.92, bottom=0.08,
                  left=0.08, right=0.95, height_ratios=[3, 1])
    fig.suptitle(f"{sym} -- Baseline vs Multi-Mode ($100 start)",
                 fontsize=13, fontweight="bold", color=CL["pri"])
    ax1 = fig.add_subplot(gs[0])
    d_old, v_old, e = [df.index[0]], [100.0], 100.0
    for t in old_trades:
        e *= (1 + t.ret_pct / 100)
        d_old.append(t.exit_date)
        v_old.append(e)
    ax1.plot(d_old, v_old, color=CL["gry"], lw=1.3, alpha=0.7,
             label=f"Baseline (${v_old[-1]:.0f})")
    nd = [x[0] for x in new_curve]
    nv = [x[1] for x in new_curve]
    ax1.plot(nd, nv, color=CL["pri"], lw=1.5, alpha=0.9,
             label=f"Multi-Mode (${nv[-1]:.0f})")
    ax1.axhline(y=100, color="#999", ls="--", lw=0.7)
    ax1.set_ylabel("Portfolio ($)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    ax2 = fig.add_subplot(gs[1])
    if mode_log:
        mode_dates = [df.index[ml[0]] for ml in mode_log]
        mode_vals = [ml[1] for ml in mode_log]
        for j in range(len(mode_dates) - 1):
            ax2.axvspan(mode_dates[j], mode_dates[j + 1],
                        color=MODE_COLORS.get(mode_vals[j], CL["gry"]), alpha=0.4)
        ax2.axvspan(mode_dates[-1], df.index[-1],
                    color=MODE_COLORS.get(mode_vals[-1], CL["gry"]), alpha=0.4)
    ax2.set_yticks([])
    ax2.set_ylabel("Mode")
    patches = [mpatches.Patch(color=c, label=m, alpha=0.6) for m, c in MODE_COLORS.items()]
    ax2.legend(handles=patches, fontsize=6, loc="upper left", ncol=4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def overlay_chart(pdf, old_curves, new_curves, title):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=CL["pri"])
    for _, d, v in old_curves:
        ax.plot(d, v, color=CL["gry"], lw=0.8, alpha=0.4)
    for lbl, d, v in new_curves:
        ax.plot(d, v, color=CL["pri"], lw=1.0, alpha=0.7, label=lbl)
    ax.axhline(y=100, color="#999", ls="--", lw=0.8)
    ax.set_ylabel("Portfolio ($)")
    ax.legend(fontsize=5, loc="upper left", ncol=3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def mode_distribution_chart(pdf, all_mode_data):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Mode Distribution by Symbol (% of trading days)",
                 fontsize=13, fontweight="bold", color=CL["pri"])
    syms = list(all_mode_data.keys())
    modes_list = ["TREND_FOLLOW", "HIGHVOL_BREAKOUT", "WHEEL", "CASH"]
    x = np.arange(len(syms))
    bottoms = np.zeros(len(syms))
    for mode in modes_list:
        vals = []
        for sym in syms:
            counts = all_mode_data[sym]
            total = sum(counts.values())
            vals.append(counts.get(mode, 0) / total * 100 if total > 0 else 0)
        ax.bar(x, vals, bottom=bottoms, label=mode,
               color=MODE_COLORS.get(mode, CL["gry"]), alpha=0.85, width=0.6)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(syms, fontsize=8, rotation=45)
    ax.set_ylabel("% Time")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    out_path = os.path.expanduser("~/.quantforge/multimode_report.pdf")
    repo_path = "/home/arliu/workspace/quant-forge/reports/multimode_report.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()

    print("Fetching VIX...")
    vix_df = fetch_ohlcv("^VIX", period="10y", interval="1d")

    # Strategy engine with default parameters
    strategy = MultiModeStrategy()

    summary_rows = []
    all_detail = {}
    old_curves = []
    new_curves = []
    all_mode_dist = {}
    all_wheel_stats = {}

    for sym, period, group in SYMBOLS:
        print(f"\n{'=' * 60}\n  {sym} {period} [{group}]\n{'=' * 60}")
        df = fetch_ohlcv(sym, period=period, interval="1d")
        if df is None or df.empty:
            print("  SKIP")
            continue
        print(f"  {len(df)} days ({df.index[0].date()} ~ {df.index[-1].date()})")

        indicators = compute_all(df)
        atr_series = compute_atr(df)
        signals = generate_trend_signals(df, indicators)
        breakout_sigs = generate_breakout_signals(df, atr_series)
        regimes = compute_regime_series(df, vix_df)
        ann_vol = compute_annualized_vol(df)

        bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0])) * 100

        # Baseline
        old_trades, old_eq = simulate_baseline(df, signals, regimes)
        old_m = compute_metrics([t.ret_pct for t in old_trades], len(df))

        # Multi-mode engine
        result = strategy.run(df, signals, breakout_sigs, regimes, ann_vol,
                              indicators, atr_series)

        new_m = compute_metrics([t.ret_pct for t in result.trades], len(df))

        # Mode distribution
        mode_counts = {}
        for _, m in result.mode_log:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        all_mode_dist[sym] = mode_counts
        primary = max(mode_counts, key=mode_counts.get) if mode_counts else "CASH"

        winner = "NEW" if result.final_equity > old_eq else (
            "OLD" if old_eq > result.final_equity else "TIE")

        ws = result.wheel_stats
        print(f"  Baseline:   {len(old_trades):>3}t  ${old_eq:>8.0f}  "
              f"Sharpe {old_m.sharpe_ratio:>5.2f}  B&H ${bh:.0f}")
        print(f"  MultiMode:  {len(result.trades):>3}t  ${result.final_equity:>8.0f}  "
              f"Sharpe {new_m.sharpe_ratio:>5.2f}  Wheel: "
              f"${ws.total_premium:.0f} prem, "
              f"{ws.csp_cycles}csp/{ws.cc_cycles}cc/{ws.assignments}asgn")
        print(f"  Winner: {winner}  Primary: {primary}")

        summary_rows.append({
            "sym": sym, "group": group,
            "old_eq": old_eq, "new_eq": result.final_equity, "bh": bh,
            "old_sharpe": old_m.sharpe_ratio, "new_sharpe": new_m.sharpe_ratio,
            "old_trades": old_m.total_trades, "new_trades": new_m.total_trades,
            "old_wr": old_m.win_rate, "new_wr": new_m.win_rate,
            "winner": winner, "primary_mode": primary,
        })
        all_wheel_stats[sym] = ws
        all_detail[sym] = {
            "df": df, "old_trades": old_trades, "old_eq": old_eq,
            "new_curve": result.equity_curve, "new_eq": result.final_equity,
            "mode_log": result.mode_log, "wstats": ws,
        }

        # Curves for overlay
        d_old, v_old, e = [df.index[0]], [100.0], 100.0
        for t in old_trades:
            e *= (1 + t.ret_pct / 100)
            d_old.append(t.exit_date)
            v_old.append(e)
        old_curves.append((f"{sym}-old", d_old, v_old))
        nd = [x[0] for x in result.equity_curve]
        nv = [x[1] for x in result.equity_curve]
        new_curves.append((f"{sym} (${result.final_equity:.0f})", nd, nv))

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    #  PDF Report
    # ═══════════════════════════════════════════════════════════════
    print(f"\nGenerating PDF -> {out_path}")
    with PdfPages(out_path) as pdf:

        # 1. Title
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.62, "QuantForge", fontsize=36, fontweight="bold",
                ha="center", color=CL["pri"], transform=ax.transAxes)
        ax.text(0.5, 0.52, "Multi-Mode Adaptive Strategy Report",
                fontsize=18, ha="center", color="#333", transform=ax.transAxes)
        ax.text(0.5, 0.44, datetime.now().strftime("%Y-%m-%d"),
                fontsize=14, ha="center", color=CL["gry"], transform=ax.transAxes)
        ax.text(0.5, 0.36, "Trend Follow + High-Vol Breakout + Wheel (Options) + Cash",
                fontsize=11, ha="center", color=CL["gry"], transform=ax.transAxes)
        ax.text(0.5, 0.30, "11 Symbols x 10Y backtest",
                fontsize=10, ha="center", color=CL["gry"], transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2. Architecture
        text_page(pdf, [
            "## Strategy Architecture: Parallel Trend + Wheel System",
            "",
            "  Two independent engines run simultaneously:",
            "",
            "  ENGINE 1 - Trend/Breakout (fires when signal present):",
            "    if regime == BEAR/CRISIS       -> CASH (no new entries)",
            "    elif BULL + vol > 45%          -> HIGHVOL_BREAKOUT (4x ATR)",
            "    else (BULL/CONSOL/NEUTRAL)     -> TREND_FOLLOW (3x ATR)",
            "",
            "  ENGINE 2 - Wheel (runs in parallel during consolidation):",
            "    if regime == CONSOLIDATION/NEUTRAL -> WHEEL (CSP/CC)",
            "    else                              -> wind down",
            "",
            "  Multiple trend positions can be open simultaneously.",
            "  Total exposure capped at 80% of capital.",
            "",
            "---",
            "",
            "## Regime-Adaptive Position Sizing",
            "",
            f"  BULL_TREND:     {strategy.base_pct['bull']:.0f}% base, "
            f"{strategy.risk_cap['bull']*100:.0f}% risk cap",
            f"  NEUTRAL:        {strategy.base_pct['neutral']:.0f}% base, "
            f"{strategy.risk_cap['neutral']*100:.0f}% risk cap",
            f"  CONSOLIDATION:  {strategy.base_pct['consol']:.0f}% base, "
            f"{strategy.risk_cap['consol']*100:.0f}% risk cap",
            "",
            "---",
            "",
            "## Exit Rules (all modes)",
            "",
            f"  Trailing stop:  {strategy.trend_atr_mult:.0f}x ATR (trend) / "
            f"{strategy.breakout_atr_mult:.0f}x ATR (breakout)",
            "  Regime exit:    bull->bear/crisis after 5 bars",
            "  MA breakdown:   2 consecutive closes below entry MA",
            f"  Time limit:     {strategy.max_hold_days}d max hold",
            "",
            "## Wheel Parameters",
            "",
            f"  Cycle:          {strategy.wheel_cycle_bars} trading days "
            f"(~{strategy.wheel_cycle_days} calendar)",
            f"  OTM:            {strategy.wheel_otm*100:.0f}%",
            f"  Premium disc:   {strategy.wheel_premium_discount*100:.0f}% of BSM",
            f"  Commission:     ${strategy.commission_per_contract}/contract",
        ], title="Strategy Architecture")

        # 3. Summary table
        summary_table_page(pdf, summary_rows, "Multi-Mode vs Baseline Summary")

        # 4. Overlay
        overlay_chart(pdf, old_curves, new_curves,
                      "All Symbols: Baseline (gray) vs Multi-Mode (blue)")

        # 5. Mode distribution
        mode_distribution_chart(pdf, all_mode_dist)

        # 6. Per-symbol charts
        for sym, d in all_detail.items():
            equity_comparison_chart(pdf, sym, d["old_trades"], d["old_eq"],
                                    d["new_curve"], d["df"], d["mode_log"])

        # 7. Wheel detail
        wheel_lines = [
            "## Wheel Strategy Income Detail",
            "",
            f"  {'Symbol':<7} {'CSP':>5} {'CC':>5} {'Asgn':>5} {'Called':>6} {'Premium$':>10}",
            "  " + "-" * 45,
        ]
        for sym in [s[0] for s in SYMBOLS]:
            ws = all_wheel_stats.get(sym)
            if ws:
                wheel_lines.append(
                    f"  {sym:<7} {ws.csp_cycles:>5} {ws.cc_cycles:>5} "
                    f"{ws.assignments:>5} {ws.calls_away:>6} ${ws.total_premium:>9.0f}"
                )
        text_page(pdf, wheel_lines, title="Wheel Strategy Detail")

        # 8. Conclusion
        wins_new = len([r for r in summary_rows if r["winner"] == "NEW"])
        wins_old = len([r for r in summary_rows if r["winner"] == "OLD"])
        avg_old = np.mean([r["old_eq"] for r in summary_rows]) if summary_rows else 100
        avg_new = np.mean([r["new_eq"] for r in summary_rows]) if summary_rows else 100
        avg_old_s = np.mean([r["old_sharpe"] for r in summary_rows]) if summary_rows else 0
        avg_new_s = np.mean([r["new_sharpe"] for r in summary_rows]) if summary_rows else 0

        concl = [
            "## Results Summary",
            "",
            f"  Multi-Mode wins: {wins_new}/{len(summary_rows)} symbols",
            f"  Baseline wins:   {wins_old}/{len(summary_rows)} symbols",
            f"  Avg Final Value: Old ${avg_old:.0f} vs New ${avg_new:.0f}",
            f"  Avg Sharpe:      Old {avg_old_s:.2f} vs New {avg_new_s:.2f}",
            "",
            "---",
            "",
            "## Per-Symbol Comparison",
            "",
        ]
        for r in summary_rows:
            delta = r["new_eq"] - r["old_eq"]
            arrow = "+" if delta > 0 else ""
            concl.append(
                f"  {r['sym']:<6} Old:${r['old_eq']:>7.0f}  New:${r['new_eq']:>7.0f}  "
                f"({arrow}{delta:.0f})  [{r['primary_mode']}]"
            )
        concl += ["", "---", "",
                   f"  Total backtest time: {elapsed:.0f}s"]
        text_page(pdf, concl, title="Conclusion")

    print(f"\nReport: {out_path}")
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)
    shutil.copy2(out_path, repo_path)
    print(f"Copied: {repo_path}")


if __name__ == "__main__":
    main()
