# Trading Decision Playbook

## Decision Loop

Every trade decision follows this cycle:
```
Signal → Classify regime → Validate (backtest) → Size position → Execute → Review
```

## Step 1: Signal Classification

Identify the type of signal and its source:

| Signal Type | Source | Examples |
|-------------|--------|----------|
| Technical | Price/Volume | RSI oversold, MA crossover, volume spike, breakout |
| Flow | Institutional data | Foreign net buy/sell (TW), options unusual activity (US) |
| Fundamental | News/Earnings | Earnings beat, analyst upgrade, guidance change |
| Macro | Market regime | Sector rotation, rate decision, VIX spike |
| Sentiment | Market behavior | Panic selling, euphoria, squeeze setup |

### Signal Quality Checklist

Before acting on any signal, verify:
- [ ] Is signal confirmed by >1 timeframe?
- [ ] Does volume support the signal?
- [ ] Is the market regime favorable for this signal type?
- [ ] Is the signal from a high-win-rate pattern (backtest data)?
- [ ] Are there contradicting signals? If so, which has more weight?

## Step 2: Regime Detection

Use these metrics to classify market regime:

| Metric | Where to Find | Meaning |
|--------|---------------|---------|
| SPY trend (50/200 MA) | Technical Analyst | Broad market direction |
| VIX level & trend | Market Analyst | Fear/complacency gauge |
| Breadth (AD line) | Market Analyst | How many stocks participating |
| Sector rotation phase | Market Analyst | Risk-on vs risk-off |
| TW foreign flow trend | TW Flow Analyst | Institutional conviction (TW) |

### Regime Decision Tree

```
                    SPY above 200MA?
                   /                 \
               YES                    NO
                /                       \
         VIX < 20?                  VIX > 30?
         /       \                  /       \
      YES         NO             YES        NO
       |           |              |          |
   BULL-CALM   BULL-VOLATILE  BEAR-PANIC  BEAR-GRIND
   Full size   Reduce size   Contrarian   Reduce size
   All signals  Quality only  Buy fear    Defensive only
```

### Regime Adjustments

| Regime | Position Size | Signal Filter | Stop Width |
|--------|--------------|---------------|------------|
| BULL-CALM | 100% | All signals valid | Standard ATR |
| BULL-VOLATILE | 70% | Only high-confidence | Wider (1.5x ATR) |
| BEAR-GRIND | 50% | Defensive + short only | Tight (0.8x ATR) |
| BEAR-PANIC | 30-50% | Contrarian buy only | Wide (2x ATR) |

## Step 3: Validation (Quant Check)

Before executing, the Quant must validate:
- Backtest Sharpe ratio ≥ 0.5
- Win rate ≥ 45%
- Max drawdown acceptable for account size
- Signal hasn't degraded in recent regime

**Quant veto is final** unless Lead overrides with explicit, documented reasoning.

## Step 4: Position Sizing

Use half-Kelly formula adjusted by regime:

```
Position % = (Kelly / 2) × Regime_Factor × Conviction_Factor

Where:
  Kelly = Win_Rate - (Loss_Rate / Win_Loss_Ratio)
  Regime_Factor = see table above (0.3 to 1.0)
  Conviction_Factor = 0.5 (low) to 1.0 (high)
```

### Hard Limits (Non-Negotiable)

```yaml
max_single_position: 10%    # of portfolio
max_sector_exposure: 25%     # of portfolio
max_open_positions: 8
max_correlated_positions: 3  # same sector/theme
daily_loss_limit: -2%
weekly_loss_limit: -5%
```

## Step 5: Execution Rules

- Always use limit orders (never market orders outside hours)
- Set stop-loss BEFORE entry confirmation
- Record entry rationale in trade journal
- No averaging down — ever
- No chasing — if missed entry, wait for pullback

## Step 6: Review

After each trade closes:
1. Was the signal valid? Did it play out as expected?
2. Was the sizing appropriate for the regime?
3. Was the exit timely? (Too early? Too late?)
4. What would I do differently?
5. Record in case study if notable

## Quick Reference: Data Commands

```bash
# Fetch US stock data
python3 -c "from quantforge.data.fetch_us import fetch_ohlcv; print(fetch_ohlcv('AAPL', period='3mo'))"

# Fetch TW institutional flow
python3 -c "from quantforge.data.fetch_tw import fetch_tw_institutional; print(fetch_tw_institutional('2330'))"

# Compute indicators
python3 -c "from quantforge.analysis.indicators import compute_all; from quantforge.data.fetch_us import fetch_ohlcv; print(compute_all(fetch_ohlcv('AAPL')))"

# Score a stock using the full factor pipeline
python3 -c "from quantforge.scanner import QuantScanner; scanner = QuantScanner()"

# Backtest
python3 -c "from quantforge.backtest.engine import BacktestEngine; engine = BacktestEngine(market='US')"
```
