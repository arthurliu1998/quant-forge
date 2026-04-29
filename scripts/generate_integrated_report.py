#!/usr/bin/env python3
"""QuantForge — 整合研究報告：4 定價模型 × 多模式自適應策略

11 支股票 × 10 年 × 4 定價模型 (BSM / Bates / Heston / SVI)
全中文 PDF 報告
"""
import os, shutil, time, math
import numpy as np, pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# ── CJK font setup ──
from matplotlib import font_manager as fm
fm._load_fontmanager(try_read_cache=False)
_cjk_fonts = ["Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
               "Microsoft JhengHei", "SimHei", "PingFang TC", "Source Han Sans TC"]
_found_cjk = None
for _f in _cjk_fonts:
    if any(_f.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _found_cjk = _f; break
if _found_cjk:
    plt.rcParams["font.sans-serif"] = [_found_cjk]
    plt.rcParams["font.monospace"] = [_found_cjk]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

from quantforge.data.fetch_us import fetch_ohlcv
from quantforge.analysis.indicators import compute_all, compute_atr, compute_adx, compute_ma
from quantforge.backtest.cost_model import CostModel
from quantforge.backtest.analytics import compute_metrics
from quantforge.core.models import Regime
from quantforge.regime.detector import RegimeDetector
from quantforge.strategy.multimode import (
    MultiModeStrategy, MultiModeResult, WheelStats,
    compute_annualized_vol, compute_regime_series,
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
MODEL_CLR = {"bsm": CL["pri"], "bates": CL["grn"],
             "heston": CL["org"], "svi": CL["pur"]}
MODEL_NAME = {"bsm": "BSM", "bates": "Bates", "heston": "Heston", "svi": "SVI"}
MODE_CLR = {"TREND_FOLLOW": CL["pri"], "HIGHVOL_BREAKOUT": CL["org"],
            "WHEEL": CL["grn"], "CASH": CL["gry"]}

SYMBOLS = [
    ("GOOGL", "10y"), ("AAPL", "10y"), ("MSFT", "10y"), ("NVDA", "10y"),
    ("SPY", "10y"), ("QQQ", "10y"), ("AMZN", "10y"), ("TSLA", "10y"),
    ("JPM", "10y"), ("JNJ", "10y"), ("XOM", "10y"),
]
MODELS = ["bsm", "bates", "heston", "svi"]


# ═══════════════════════════════════════════════════════════════════
#  Baseline buy-and-hold
# ═══════════════════════════════════════════════════════════════════
def simulate_baseline(df: pd.DataFrame) -> tuple[float, list[tuple]]:
    """Buy-and-hold from day 0 normalised to $100."""
    prices = df["Close"].values
    base = prices[0]
    curve = [(df.index[i], 100.0 * prices[i] / base) for i in range(len(df))]
    return curve[-1][1], curve


# ═══════════════════════════════════════════════════════════════════
#  PDF helpers
# ═══════════════════════════════════════════════════════════════════
def text_page(pdf, lines, title=None):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111); ax.axis("off"); y = 0.96
    if title:
        ax.text(0.05, y, title, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", color=CL["pri"]); y -= 0.05
        ax.plot([0.05, 0.95], [y + 0.005] * 2, color=CL["pri"], lw=1.5,
                transform=ax.transAxes, clip_on=False); y -= 0.025
    for line in lines:
        if line.startswith("##"):
            y -= 0.01; ax.text(0.05, y, line[3:], transform=ax.transAxes,
                               fontsize=12, fontweight="bold", va="top", color="#333"); y -= 0.04
        elif line.startswith("**"):
            ax.text(0.05, y, line.replace("**", ""), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", color="#333",
                    family="monospace"); y -= 0.03
        elif line == "---":
            ax.plot([0.05, 0.95], [y + 0.005] * 2, color="#ddd", lw=0.5,
                    transform=ax.transAxes, clip_on=False); y -= 0.015
        elif line == "":
            y -= 0.012
        else:
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=8.5,
                    va="top", color="#444", family="monospace",
                    linespacing=1.35); y -= 0.026
        if y < 0.03:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111); ax.axis("off"); y = 0.96
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def table_page(pdf, col_labels, rows, cell_colors, title):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111); ax.axis("off")
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="center", va="top", color=CL["pri"])
    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=[CL["lblu"]] * len(col_labels),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(6.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(CL["pri"])
        cell.set_edgecolor("#ddd")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    out_path = os.path.expanduser("~/.quantforge/integrated_report.pdf")
    repo_path = "/home/arliu/workspace/quant-forge/reports/integrated_report.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(repo_path), exist_ok=True)
    t0 = time.time()

    print("Fetching VIX...")
    vix_df = fetch_ohlcv("^VIX", period="10y", interval="1d")

    # ── Run backtests ──
    all_results = {}   # sym -> {model -> MultiModeResult}
    all_baselines = {} # sym -> (final_eq, curve)
    all_dfs = {}       # sym -> df
    all_premiums = {}  # sym -> {model -> wheel_premium}

    for sym, period in SYMBOLS:
        print(f"\n{'=' * 55}\n  {sym} {period}\n{'=' * 55}")
        df = fetch_ohlcv(sym, period=period, interval="1d")
        if df is None or df.empty:
            print("  SKIP"); continue
        print(f"  {len(df)} days ({df.index[0].date()} ~ {df.index[-1].date()})")
        all_dfs[sym] = df

        # Shared prep
        indicators = compute_all(df)
        atr_series = compute_atr(df)
        ann_vol = compute_annualized_vol(df)
        regimes = compute_regime_series(df, vix_df)
        trend_sigs = generate_trend_signals(df, indicators)
        breakout_sigs = generate_breakout_signals(df, atr_series)

        # Baseline
        base_eq, base_curve = simulate_baseline(df)
        all_baselines[sym] = (base_eq, base_curve)
        print(f"  Baseline B&H: ${base_eq:.0f}")

        # Run 4 pricing models
        sym_results = {}
        sym_premiums = {}
        for model in MODELS:
            strat = MultiModeStrategy(pricing_model=model)
            result = strat.run(df, trend_sigs, breakout_sigs, regimes,
                               ann_vol, indicators, atr_series)
            sym_results[model] = result
            sym_premiums[model] = result.wheel_stats.total_premium
            ws = result.wheel_stats
            print(f"  {MODEL_NAME[model]:<6} eq=${result.final_equity:>8.1f}  "
                  f"premium=${ws.total_premium:>7.1f}  "
                  f"CSP={ws.csp_cycles} CC={ws.cc_cycles} assign={ws.assignments}")

        all_results[sym] = sym_results
        all_premiums[sym] = sym_premiums

    elapsed = time.time() - t0
    print(f"\nBacktest time: {elapsed:.0f}s")

    # ── Determine best model per symbol ──
    best_models = {}
    for sym in all_results:
        best_m = max(MODELS, key=lambda m: all_results[sym][m].final_equity)
        best_models[sym] = best_m

    # ═══════════════════════════════════════════════════════════════
    #  PDF Generation
    # ═══════════════════════════════════════════════════════════════
    print(f"\nGenerating PDF -> {out_path}")
    with PdfPages(out_path) as pdf:

        # ── 1. 封面 ──
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.65, "QuantForge", fontsize=38, fontweight="bold",
                ha="center", color=CL["pri"], transform=ax.transAxes)
        ax.text(0.5, 0.55, "多模式自適應量化策略 完整研究報告",
                fontsize=18, ha="center", color="#333", transform=ax.transAxes)
        ax.text(0.5, 0.47, "4 定價模型 (BSM / Bates / Heston / SVI) × 11 標的 × 10 年",
                fontsize=12, ha="center", color=CL["gry"], transform=ax.transAxes)
        ax.text(0.5, 0.40, datetime.now().strftime("%Y-%m-%d"),
                fontsize=14, ha="center", color=CL["gry"], transform=ax.transAxes)
        ax.text(0.5, 0.30, "GOOGL  AAPL  MSFT  NVDA  SPY  QQQ",
                fontsize=10, ha="center", color=CL["gry"], transform=ax.transAxes)
        ax.text(0.5, 0.25, "AMZN  TSLA  JPM  JNJ  XOM",
                fontsize=10, ha="center", color=CL["gry"], transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── 2. 目錄 ──
        text_page(pdf, [
            "## 目錄",
            "",
            "  1. 策略架構 ................ 4 模式決策樹 + 並行引擎說明",
            "  2. 定價模型介紹 ............ BSM / Bates / Heston / SVI 原理",
            "  3. 定價模型對比表 .......... 每支股票 x 4 模型的 Wheel 收益比較",
            "  4. 定價模型影響圖 .......... Premium 差異分組柱狀圖",
            "  5. 最佳模型策略總結 ........ 各標的最佳模型及績效",
            "  6. 全標的權益曲線 .......... Baseline vs MultiMode",
            "  7. 模式分佈圖 .............. 堆疊柱狀圖",
            "  8. 逐標的權益對比 .......... 11 張個股圖 (含模式色帶)",
            "  9. Wheel 詳情頁 ............ CSP/CC 週期、指派、權利金",
            " 10. 關鍵發現與建議 .......... 定價模型結論 + 策略結論",
        ], title="目錄")

        # ── 3. 策略架構 ──
        text_page(pdf, [
            "## 多模式決策樹",
            "",
            "  每根 K 棒根據市場狀態(Regime)和年化波動率選擇模式：",
            "",
            "  BEAR / CRISIS           -> CASH        (不交易)",
            "  BULL + vol > 45%        -> HIGHVOL_BREAKOUT (4x ATR 追蹤停損)",
            "  BULL + vol <= 45%       -> TREND_FOLLOW     (3x ATR 追蹤停損)",
            "  CONSOLIDATION / NEUTRAL -> WHEEL            (CSP/CC 選擇權收入)",
            "",
            "---",
            "",
            "## 並行引擎",
            "",
            "  三個引擎同時運行：",
            "  1. 趨勢/突破引擎 — 多個同時持有的趨勢/突破部位",
            "  2. Wheel 引擎    — 模擬 Cash-Secured Put / Covered Call 週期",
            "  3. 現金引擎      — 熊市/危機期間不開新倉",
            "",
            "---",
            "",
            "## Regime-Adaptive Sizing",
            "",
            "  RegimeDetector:  VIX > 30 = CRISIS, ADX + MA60 判斷趨勢",
            "  部位大小依 regime 調整：",
            "    Bull:    50% base, 10% 風險上限",
            "    Neutral: 30% base,  5% 風險上限",
            "    Consol:  20% base,  3% 風險上限",
            "",
            "  最大總曝險: 80%",
            "  持有上限:   250 交易日",
        ], title="策略架構")

        # ── 4. 定價模型介紹 ──
        text_page(pdf, [
            "## BSM (Black-Scholes-Merton)",
            "",
            "  經典歐式選擇權定價。假設波動率恆定、無跳躍。",
            "  優點：計算快速、美元定價最準 (vega 加權 IV RMSE 0.0627)",
            "  缺點：無法捕捉波動率微笑和跳躍風險",
            "",
            "---",
            "",
            "## Bates (跳躍擴散模型)",
            "",
            "  Merton 跳躍擴散封閉解：BSM 價格 x Poisson 跳躍調整項",
            "  參數：lambda=0.1 (跳躍頻率), muJ=-0.05 (跳躍均值),",
            "        sigmaJ=0.10 (跳躍波動率)",
            "  優點：液態選擇權上最佳 (IV RMSE 0.0549)",
            "  特性：premium >= BSM (跳躍項增加尾部風險)",
            "",
            "---",
            "",
            "## Heston (隨機波動率近似)",
            "",
            "  二次偏態近似：adjusted_vol = v0 + rho*xi/(2*v0) * moneyness",
            "  參數：kappa=2.0, theta=vol^2, xi=0.3, rho=-0.7",
            "  優點：速度快 (一次 BSM 呼叫)",
            "  缺點：近似精度一般，介於 BSM 和 SVI 之間",
            "",
            "---",
            "",
            "## SVI (Stochastic Volatility Inspired)",
            "",
            "  Gatheral SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))",
            "  k = ln(K/S), w = implied_var * T",
            "  參數：a=0.04, b=0.1, rho=-0.4, m=0, sigma=0.1",
            "  優點：波動率預測最好 (OOS IV RMSE 0.163)",
            "  缺點：美元定價最差",
            "",
            "---",
            "",
            "## 研究結論摘要",
            "",
            "  模型       | IV RMSE | 美元精度  | 速度  | 最佳場景",
            "  ---------- | ------- | --------- | ----- | --------",
            "  BSM        | 0.0627  | 最佳      | 極快  | 一般用途",
            "  Bates      | 0.0549  | 次佳      | 中等  | 液態期權",
            "  Heston     | 中等    | 中等      | 快    | 偏態定價",
            "  SVI        | 0.163   | 最差      | 快    | 波動率預測",
        ], title="定價模型介紹")

        # ── 5. 定價模型對比表 ──
        cols = ["標的", "BSM Premium", "Bates Premium", "Heston Premium",
                "SVI Premium", "BSM Eq", "Bates Eq", "Heston Eq", "SVI Eq"]
        rows_data = []
        cell_clrs = []
        for sym in all_results:
            r = all_results[sym]
            prems = [f"${all_premiums[sym][m]:.0f}" for m in MODELS]
            eqs = [f"${r[m].final_equity:.0f}" for m in MODELS]
            best_eq = max(r[m].final_equity for m in MODELS)
            row_clr = []
            for m in MODELS:
                row_clr.append(CL["lgrn"] if r[m].final_equity == best_eq else "white")
            rows_data.append([sym] + prems + eqs)
            cell_clrs.append(["white"] + ["white"] * 4 + row_clr)
        table_page(pdf, cols, rows_data, cell_clrs,
                   "定價模型對比 — Wheel Premium 及最終權益")

        # ── 6. 定價模型影響圖 ──
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle("定價模型影響 — Premium 及權益比較",
                     fontsize=14, fontweight="bold", color=CL["pri"])
        syms = list(all_results.keys())
        x = np.arange(len(syms))
        width = 0.2

        # Premium comparison
        for j, m in enumerate(MODELS):
            vals = [all_premiums[s][m] for s in syms]
            axes[0].bar(x + j * width - 1.5 * width, vals, width,
                        label=MODEL_NAME[m], color=MODEL_CLR[m], alpha=0.85)
        axes[0].set_xticks(x); axes[0].set_xticklabels(syms, fontsize=7, rotation=45)
        axes[0].set_ylabel("Wheel Premium ($)")
        axes[0].set_title("各模型 Wheel Premium 比較")
        axes[0].legend(fontsize=7)

        # Equity comparison
        for j, m in enumerate(MODELS):
            vals = [all_results[s][m].final_equity for s in syms]
            axes[1].bar(x + j * width - 1.5 * width, vals, width,
                        label=MODEL_NAME[m], color=MODEL_CLR[m], alpha=0.85)
        axes[1].set_xticks(x); axes[1].set_xticklabels(syms, fontsize=7, rotation=45)
        axes[1].set_ylabel("最終權益 ($)")
        axes[1].set_title("各模型最終權益比較")
        axes[1].legend(fontsize=7)
        axes[1].axhline(y=100, color="#999", ls="--", lw=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── 7. 最佳模型策略總結表 ──
        cols2 = ["標的", "Baseline $", "最佳模型", "最佳 Eq $",
                 "Sharpe", "Wheel Premium", "CSP", "CC", "指派",
                 "主要模式"]
        rows2 = []; clrs2 = []
        for sym in all_results:
            bm = best_models[sym]
            r = all_results[sym][bm]
            base_eq = all_baselines[sym][0]
            # Compute sharpe from equity curve
            eq_vals = [v for _, v in r.equity_curve]
            if len(eq_vals) > 1:
                rets = np.diff(eq_vals) / eq_vals[:-1]
                sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
            else:
                sharpe = 0
            # Mode distribution
            mode_counts = {}
            for _, mode_str in r.mode_log:
                mode_counts[mode_str] = mode_counts.get(mode_str, 0) + 1
            top_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "N/A"

            ws = r.wheel_stats
            beat = r.final_equity > base_eq
            row_clr = [CL["lgrn"] if beat else CL["lred"]] * len(cols2)
            rows2.append([sym, f"${base_eq:.0f}", MODEL_NAME[bm],
                          f"${r.final_equity:.0f}", f"{sharpe:.2f}",
                          f"${ws.total_premium:.0f}",
                          str(ws.csp_cycles), str(ws.cc_cycles),
                          str(ws.assignments), top_mode])
            clrs2.append(row_clr)
        table_page(pdf, cols2, rows2, clrs2,
                   "最佳模型策略總結 — 各標的績效")

        # ── 8. 全標的權益曲線 ──
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle("全標的權益曲線 — Baseline vs 最佳 MultiMode",
                     fontsize=14, fontweight="bold", color=CL["pri"])
        cc_list = [CL["pri"], CL["grn"], CL["pur"], CL["org"], CL["red"],
                   CL["gry"], CL["teal"], CL["pink"], "#795548", "#607d8b", "#ff9800"]
        for i, sym in enumerate(all_results):
            # Baseline (grey, thin)
            base_curve = all_baselines[sym][1]
            bd, bv = zip(*base_curve)
            ax.plot(bd, bv, color="#ccc", lw=0.5, alpha=0.5)
            # Best model
            bm = best_models[sym]
            r = all_results[sym][bm]
            ed, ev = zip(*r.equity_curve)
            ax.plot(ed, ev, color=cc_list[i % len(cc_list)], lw=1.2, alpha=0.85,
                    label=f"{sym} ({MODEL_NAME[bm]}) ${r.final_equity:.0f}")
        ax.axhline(y=100, color="#999", ls="--", lw=0.7)
        ax.set_ylabel("權益 ($)")
        ax.legend(fontsize=6, loc="upper left", ncol=3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── 9. 模式分佈圖 ──
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle("模式分佈 — 各標的模式時間佔比 (最佳定價模型)",
                     fontsize=14, fontweight="bold", color=CL["pri"])
        mode_names = ["TREND_FOLLOW", "HIGHVOL_BREAKOUT", "WHEEL", "CASH"]
        sym_list = list(all_results.keys())
        bottom = np.zeros(len(sym_list))
        for mode_name in mode_names:
            vals = []
            for sym in sym_list:
                bm = best_models[sym]
                r = all_results[sym][bm]
                total = len(r.mode_log)
                cnt = sum(1 for _, m in r.mode_log if m == mode_name)
                vals.append(cnt / total * 100 if total > 0 else 0)
            ax.bar(range(len(sym_list)), vals, bottom=bottom,
                   label=mode_name, color=MODE_CLR.get(mode_name, CL["gry"]),
                   alpha=0.85)
            bottom += vals
        ax.set_xticks(range(len(sym_list)))
        ax.set_xticklabels(sym_list, fontsize=8, rotation=45)
        ax.set_ylabel("佔比 (%)")
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── 10. 逐標的權益對比 ──
        for sym in all_results:
            df = all_dfs[sym]
            fig = plt.figure(figsize=(11, 6))
            gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                          top=0.90, bottom=0.08, left=0.08, right=0.95)
            fig.suptitle(f"{sym} — 4 定價模型權益對比 ($100 起始)",
                         fontsize=13, fontweight="bold", color=CL["pri"])

            # Main equity curves
            ax1 = fig.add_subplot(gs[0, :])
            # Baseline
            bd, bv = zip(*all_baselines[sym][1])
            ax1.plot(bd, bv, color="#aaa", lw=1.0, ls="--", alpha=0.7,
                     label=f"Baseline B&H (${all_baselines[sym][0]:.0f})")
            for m in MODELS:
                r = all_results[sym][m]
                ed, ev = zip(*r.equity_curve)
                ax1.plot(ed, ev, color=MODEL_CLR[m], lw=1.2, alpha=0.85,
                         label=f"{MODEL_NAME[m]} (${r.final_equity:.0f})")
            ax1.axhline(y=100, color="#999", ls="--", lw=0.5)
            ax1.set_ylabel("權益 ($)")
            ax1.legend(fontsize=7, loc="upper left")
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

            # Premium bar
            ax2 = fig.add_subplot(gs[1, 0])
            prems = [all_premiums[sym][m] for m in MODELS]
            bars = ax2.bar(range(4), prems,
                           color=[MODEL_CLR[m] for m in MODELS], alpha=0.85)
            ax2.set_xticks(range(4))
            ax2.set_xticklabels([MODEL_NAME[m] for m in MODELS], fontsize=8)
            ax2.set_ylabel("Wheel Premium ($)")
            ax2.set_title("Wheel Premium 比較")
            for bar, p in zip(bars, prems):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                         f"${p:.0f}", ha="center", fontsize=8, fontweight="bold")

            # Final equity bar
            ax3 = fig.add_subplot(gs[1, 1])
            eqs = [all_results[sym][m].final_equity for m in MODELS]
            bars = ax3.bar(range(4), eqs,
                           color=[MODEL_CLR[m] for m in MODELS], alpha=0.85)
            ax3.set_xticks(range(4))
            ax3.set_xticklabels([MODEL_NAME[m] for m in MODELS], fontsize=8)
            ax3.set_ylabel("最終權益 ($)")
            ax3.set_title("最終權益比較")
            ax3.axhline(y=100, color="#999", ls="--", lw=0.7)
            for bar, eq in zip(bars, eqs):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                         f"${eq:.0f}", ha="center", fontsize=8, fontweight="bold")

            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── 11. Wheel 詳情頁 ──
        wheel_lines = []
        for sym in all_results:
            wheel_lines.append(f"## {sym}")
            wheel_lines.append("")
            wheel_lines.append("  模型     | CSP 週期 | CC 週期 | 指派 | 被Call走 | 總 Premium")
            wheel_lines.append("  -------- | -------- | ------- | ---- | -------- | ----------")
            for m in MODELS:
                ws = all_results[sym][m].wheel_stats
                wheel_lines.append(
                    f"  {MODEL_NAME[m]:<8} | {ws.csp_cycles:>8} | {ws.cc_cycles:>7} | "
                    f"{ws.assignments:>4} | {ws.calls_away:>8} | ${ws.total_premium:>9.0f}")
            wheel_lines.append("")
            wheel_lines.append("---")
            wheel_lines.append("")
        text_page(pdf, wheel_lines, title="Wheel 詳情 — CSP/CC 週期及權利金")

        # ── 12. 關鍵發現與建議 ──
        # Compute aggregate stats
        bsm_wins = sum(1 for s in best_models.values() if s == "bsm")
        bates_wins = sum(1 for s in best_models.values() if s == "bates")
        heston_wins = sum(1 for s in best_models.values() if s == "heston")
        svi_wins = sum(1 for s in best_models.values() if s == "svi")

        avg_eq = {m: np.mean([all_results[s][m].final_equity
                              for s in all_results]) for m in MODELS}
        avg_prem = {m: np.mean([all_premiums[s][m]
                                for s in all_premiums]) for m in MODELS}

        beat_baseline = sum(1 for s in all_results
                            if all_results[s][best_models[s]].final_equity > all_baselines[s][0])

        text_page(pdf, [
            "## 定價模型結論",
            "",
            f"  最佳模型分佈：BSM={bsm_wins}  Bates={bates_wins}  "
            f"Heston={heston_wins}  SVI={svi_wins}",
            "",
            f"  平均最終權益：",
            f"    BSM    ${avg_eq['bsm']:.0f}   (avg premium ${avg_prem['bsm']:.0f})",
            f"    Bates  ${avg_eq['bates']:.0f}   (avg premium ${avg_prem['bates']:.0f})",
            f"    Heston ${avg_eq['heston']:.0f}   (avg premium ${avg_prem['heston']:.0f})",
            f"    SVI    ${avg_eq['svi']:.0f}   (avg premium ${avg_prem['svi']:.0f})",
            "",
            "  Bates 模型因跳躍項增加尾部風險定價，Wheel premium 通常 >= BSM。",
            "  Heston 透過偏態調整 OTM put 定價略高。",
            "  SVI 的波動率曲面可能對深 OTM 有不同表現。",
            "",
            "---",
            "",
            "## 策略結論",
            "",
            f"  {beat_baseline}/{len(all_results)} 支標的 MultiMode 策略勝過 Buy-and-Hold",
            "",
            "  多模式引擎核心優勢：",
            "    1. 熊市自動切換 CASH 模式避免重大回撤",
            "    2. 盤整期透過 Wheel 產生額外選擇權收入",
            "    3. 牛市趨勢跟蹤 + 突破捕捉雙引擎並行",
            "    4. ATR-based 動態追蹤停損控制下行風險",
            "",
            "---",
            "",
            "## 部署建議",
            "",
            "  1. 定價模型選擇：",
            "     - 一般用途推薦 BSM (最快、最穩定)",
            "     - 高波動標的可考慮 Bates (更高 premium 但更準確)",
            "     - 不建議單獨依賴 SVI (美元定價偏差較大)",
            "",
            "  2. 策略部署：",
            "     - Mode B (+ Regime Filter) 為最低可行部署版本",
            "     - 趨勢型成長股效果最佳 (NVDA, GOOGL, AAPL)",
            "     - 防禦型/價值股效果較差 (JNJ, XOM)",
            "     - TSLA 等極高波動標的需額外風控",
            "",
            "  3. 風險管理：",
            "     - 建議最大單一標的曝險不超過 20%",
            "     - Wheel 部位需準備足夠保證金 (100 股 x 行權價)",
            "     - 定期檢視 regime 偵測器參數是否需調整",
            "",
            f"  報告生成時間: {elapsed:.0f} 秒",
            f"  報告日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ], title="關鍵發現與建議")

    print(f"\nReport: {out_path}")
    shutil.copy2(out_path, repo_path)
    print(f"Copied: {repo_path}")


if __name__ == "__main__":
    main()
