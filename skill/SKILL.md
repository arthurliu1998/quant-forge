---
name: quantforge
description: >
  Multi-agent trading analysis system for US and Taiwan stocks.
  Coordinates 9 agents (Lead, Technical, Market, TW/US Flow, Sentinel,
  Quant, Risk, Executor) for signal detection, analysis, and semi-automated
  trading. Includes trajectory tracking, diagnostic codes (SIG/RGM/RSK/BHV),
  trade case studies, and learning experience system.
  Trigger: "analyze TSLA", "scan watchlist", "scan full", "market brief",
  "/quantforge", "review trade", "trade journal".
user-invocable: true
argument-hint: "[analyze <SYMBOL> | scan [full] | brief | portfolio | review | journal | load <name>]"
---

# QuantForge: Multi-Agent Trading Analysis System

## Quick Reference

| Command | Effect |
|---------|--------|
| `/quantforge` | New team setup wizard |
| `/quantforge analyze <SYMBOL>` | Full analysis of a single stock |
| `/quantforge scan` | Quick scan watchlist for signals |
| `/quantforge scan full` | Scan + parallel multi-agent analysis for top signals |
| `/quantforge brief` | Morning/evening market brief |
| `/quantforge portfolio` | Show positions and P&L |
| `/quantforge review [SYMBOL]` | Post-trade review with Socratic analysis |
| `/quantforge journal` | View/update trading learning log |
| `/quantforge load <name>` | Resume saved team |

## Mode Dispatch

Parse `$ARGUMENTS`:
- **(empty)** → New Team Setup wizard
- **`analyze <SYMBOL>`** → Single Stock Analysis
- **`scan`** → Quick Watchlist Scan (signals only)
- **`scan full`** → Full Watchlist Scan with Parallel Multi-Agent Analysis
- **`brief`** → Market Brief
- **`portfolio`** → Portfolio View
- **`review [SYMBOL]`** → Trade Review
- **`journal`** → Learning Journal
- **`load <name>`** → Resume Team

---

## New Team Setup

Walk through one at a time:

1. **Goal**: "What is your trading goal?"
2. **Roles**: Recommend based on goal:
   - Analyze/scan: Lead, Technical, Market, Quant
   - Full+flow: Lead, Technical, Market, TW/US Flow, Sentinel, Quant, Risk
   - Monitor: Lead, Technical, Risk, Executor
   - Deep dive: All 9
3. **Watchlist**: "Symbols? (comma-separated, or 'use config')"
4. **Confirm & Spawn**:
   - `TeamCreate({ team_name: "quantforge-<date>" })`
   - Read `${CLAUDE_SKILL_DIR}/roles/<role>.md`, fill placeholders
   - Spawn agents, begin as Lead

---

## Single Stock Analysis (`analyze <SYMBOL>`)

Spawn Technical + Market + Risk. Analyze → synthesize → report.

---

## Quick Watchlist Scan (`scan`)

Run `QuantScanner.score_stock()` across all symbols in config watchlist.
Output scored signals sorted by quant score. No multi-agent analysis — fast check only.

```bash
python3 -c "
from quantforge.scanner import QuantScanner
scanner = QuantScanner()
# Score each stock with available data (see QuantScanner.score_stock API)
"
```

---

## Full Watchlist Scan with Auto-Triage (`scan full`)

Scans all watchlist symbols, then launches **parallel multi-agent analysis** for the top signals.

### Flow

```
1. Scan all symbols → collect signals
2. Triage: group by symbol, rank by priority (CRITICAL > HIGH > MEDIUM > LOW)
3. Select top N symbols (max 5) with highest-priority signals
4. Spawn parallel Scan-Analyst subagents (one per symbol, up to 5 concurrent)
5. Collect results → synthesize unified watchlist report
6. Symbols with no signals: listed as "No Signal" (not analyzed)
```

### Step 1: Signal Scan

```bash
python3 -c "
from quantforge.scanner import QuantScanner
scanner = QuantScanner()
# Score each stock with available data (see QuantScanner.score_stock API)
"
```

### Step 2: Triage

Group signals by symbol. For each symbol, take the highest-priority signal.
Sort symbols by their top signal priority:

| Priority | Action |
|----------|--------|
| CRITICAL | Always analyze (first in queue) |
| HIGH | Analyze if slots available (max 5 total) |
| MEDIUM | Analyze if slots available |
| LOW | Skip (report signal only, no deep analysis) |

If more than 5 symbols have HIGH+ signals, pick the 5 with most signals or highest volume spike.

### Step 3: Parallel Analysis

For each selected symbol, spawn a **Scan-Analyst** subagent:

```
Agent({
  prompt: <filled scan-analyst.md template>,
  name: "scan-<SYMBOL>",
  subagent_type: "general-purpose",
  run_in_background: true
})
```

Read the template from `${CLAUDE_SKILL_DIR}/roles/scan-analyst.md`.
Fill placeholders: `{{SYMBOL}}`, `{{MARKET}}`, `{{SIGNALS}}`, `{{REGIME_CONTEXT}}`.

**All subagents are launched in a single message** for true parallelism.

### Step 4: Synthesize Report

After all subagents return, synthesize into a unified report:

```
══════════════════════════════════════════════════════
QuantForge Watchlist Scan Report — YYYY-MM-DD HH:MM
══════════════════════════════════════════════════════
Regime: BULL-CALM | SPY: $XXX (>200MA) | VIX: XX

Rank | Symbol | Signal        | Tech | Mkt | Risk | Rec    | Confidence
─────┼────────┼───────────────┼──────┼─────┼──────┼────────┼───────────
  1  | NVDA   | RSI oversold  | 8/10 | 7   | 7    | BUY    | 75%
  2  | AAPL   | MA20 cross up | 7/10 | 7   | 6    | BUY    | 65%
  3  | 2330   | Vol spike 3x  | 6/10 | 5   | 5    | WATCH  | 45%
  ── | TSLA   | No signal     |  —   |  —  |  —   |  —     |  —
  ── | MSFT   | No signal     |  —   |  —  |  —   |  —     |  —
══════════════════════════════════════════════════════
Top pick: NVDA — RSI oversold in uptrend, strong RS vs sector
══════════════════════════════════════════════════════
```

### Step 5: Follow-Up

After presenting the report, ask user:
1. "要深入分析哪一檔？" → launch full `/quantforge analyze <SYMBOL>`
2. "要執行嗎？" → proceed to Executor flow (with user confirmation)
3. "下次再看" → done

---

## Trade Review (`review [SYMBOL]`)

Post-trade analysis using Socratic questioning. Flow:

1. If SYMBOL given, pull trade history from portfolio; otherwise ask user for details
2. Walk through the review using the Socratic flow from `~/.claude/notes/trading/diagnostics.md`:
   - What happened? (entry/exit/P&L)
   - What type of issue? (SIG/RGM/RSK/BHV)
   - Root cause hypothesis?
   - How to confirm?
   - What would you do differently?
3. Offer to record as case study in `~/.claude/notes/trading/cases/`
4. Check if patterns are emerging across cases

Reference files:
- `~/.claude/notes/trading/playbook.md` — decision framework
- `~/.claude/notes/trading/diagnostics.md` — diagnostic codes
- `~/.claude/notes/trading/cases.md` — case index
- `~/.claude/notes/trading/patterns.md` — confirmed patterns

---

## Learning Journal (`journal`)

View and update trading skill progression:

1. Read `~/.claude/notes/trading/learning-log.md`
2. Show current skill self-assessment
3. Ask: "Want to update any skill ratings? Or add a learning entry?"
4. If updating, walk through:
   - Which skill changed?
   - New level (1-5)?
   - What experience led to the change?
5. Save updates

---

## Lead Operating Instructions

After spawn, you ARE the Lead. See `roles/lead.md`.

Key rules:
- Coordinate, don't implement
- Enforce discipline rules (circuit breakers, anti-FOMO)
- One task per member at a time
- Ask user for confirmation before any trade action
- Track decisions via trajectory
- Prompt for trade review after each closed trade

## Trajectory Tracking

Record every trade decision via:
```bash
python3 -c "
from quantforge.trajectory import init_trajectory, record_decision
init_trajectory('{{TRAJECTORY_PATH}}')
record_decision('{{TRAJECTORY_PATH}}', symbol='XXX', direction='long/short',
                signal_type='technical/flow/...', diagnostic_code='SIG-XXX',
                entry_price=0, exit_price=0, pnl_pct=0,
                status='open/closed/stopped')
"
```

View history: `from quantforge.trajectory import get_summary; get_summary('{{TRAJECTORY_PATH}}')`

## Experience System Integration

The trading experience system lives at `~/.claude/notes/trading/`:

| File | Purpose |
|------|---------|
| `playbook.md` | Decision loop, signal classification, regime detection, sizing |
| `diagnostics.md` | Diagnostic codes: SIG-T/F (signals), RGM (regime), RSK (risk), BHV (behavioral) |
| `cases.md` | Trade case study index with statistics |
| `cases/case-NNN-*.md` | Individual trade case studies |
| `patterns.md` | Confirmed reusable trading patterns |
| `learning-log.md` | Skill self-assessment (12 skills, rated 1-5) |

### Diagnostic Code Quick Reference

| Code | Category | Example |
|------|----------|---------|
| SIG-T01 | Technical signal reversed | RSI oversold but price kept falling |
| SIG-T02 | MA crossover whipsaw | Choppy market, no trend |
| SIG-T03 | RSI oversold in bear market | Oversold ≠ buy in downtrend |
| SIG-F01 | Flow signal no follow-through | One-off rebalancing |
| SIG-F02 | Options unusual activity no move | Hedge, not directional |
| RGM-01 | Regime classification wrong | Used lagging indicators |
| RGM-02 | Right regime, wrong filter | Bull signals in bear regime |
| RSK-01 | Position too large | Gap risk, correlated positions |
| RSK-02 | Stop hit then recovered | Stop too tight for volatility |
| RSK-03 | Took profit too early | No trailing stop system |
| BHV-01 | Chased entry (FOMO) | Entered > 2% past signal |
| BHV-02 | Held loser too long | No stop or moved stop wider |
| BHV-03 | Overtrading | Boredom/revenge trading |
| BHV-04 | Cut winner, held loser | Loss aversion pattern |

### After Each Trade Closes

Lead must prompt:
> 要不要記錄這筆交易？我會用 `trading/cases/case-000-template.md` 建案例。

### Pattern Extraction

When 2-3 cases share a common lesson, propose extracting to `patterns.md`.

## Security Rules (ALL roles)

- NEVER read/cat .env or files with API keys
- NEVER run: echo $API_KEY, env | grep KEY, cat ~/.quantforge/.env
- NEVER log/print secrets
- Check keys: `SecretManager.is_configured('KEY_NAME')`
- LLM data via DataSanitizer only — no absolute dollar amounts
