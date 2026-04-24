---
name: trade-forge
description: >
  Multi-agent trading analysis system for US and Taiwan stocks.
  Coordinates 9 agents (Lead, Technical, Market, TW/US Flow, Sentinel,
  Quant, Risk, Executor) for signal detection and semi-automated trading.
  Trigger: "analyze TSLA", "scan watchlist", "market brief", "/trade-forge".
user-invocable: true
argument-hint: "[analyze <SYMBOL> | scan | brief | portfolio | load <name>]"
---

# TradeForge

## Commands

Parse `$ARGUMENTS`:
- **(empty)** → New Team Setup wizard
- **`analyze <SYMBOL>`** → Single Stock Analysis
- **`scan`** → Watchlist Scan
- **`brief`** → Market Brief
- **`portfolio`** → Portfolio View
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
   - `TeamCreate({ team_name: "trade-forge-<date>" })`
   - Read `${CLAUDE_SKILL_DIR}/roles/<role>.md`, fill placeholders
   - Spawn agents, begin as Lead

---

## Single Stock Analysis (`analyze <SYMBOL>`)

Spawn Technical + Market + Risk. Analyze → synthesize → report.

---

## Lead Operating Instructions

After spawn, you ARE the Lead. See `roles/lead.md`.
Key: coordinate don't implement, enforce discipline, confirm before trades.

## Security Rules (ALL roles)

- NEVER read/cat .env or files with API keys
- NEVER run: echo $API_KEY, env | grep KEY, cat ~/.trade-forge/.env
- NEVER log/print secrets
- Check keys: `SecretManager.is_configured('KEY_NAME')`
- LLM data via DataSanitizer only — no absolute dollar amounts
