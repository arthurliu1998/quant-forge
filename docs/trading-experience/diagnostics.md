# Trading Diagnostics: Signal → Outcome Map

When a trade goes wrong (or right), use this map to categorize what happened.
Each entry includes: what you observed → possible cause → how to confirm → lesson.

---

## Signal Symptoms (SIG-XXX)

### SIG-T01: Technical Signal Triggered but Price Reversed
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Signal was against the prevailing trend | Check higher timeframe (weekly) trend direction | Don't trade counter-trend signals in strong trends unless regime is BEAR-PANIC |
| Low volume confirmation | Compare signal-day volume to 20-day average | Require volume ≥ 1.5x average for breakout/reversal signals |
| Signal at major resistance/support | Check if price was at a key level when signal triggered | Signals at resistance (for longs) need extra confirmation |

### SIG-T02: MA Crossover Whipsaw
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Choppy/range-bound market | Check ADX — below 20 means no trend | Don't trade MA crossovers when ADX < 20 |
| Using too-short MA period | Compare 5/10 vs 20/50 vs 50/200 crossover results | Use longer MAs in volatile markets; shorter in trending |
| Late entry after crossover | Time between crossover and entry > 2 bars | Enter on crossover bar or first pullback, not later |

### SIG-T03: RSI Oversold but Price Kept Falling
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Bear market — RSI stays oversold for weeks | Check regime: SPY below 200MA? | In bear markets, RSI oversold is not a buy signal — it's confirmation of weakness |
| Fundamental catalyst driving price down | Check Sentinel for news | RSI doesn't capture fundamental shifts; check news before buying RSI oversold |
| Capitulation wave not done | Check volume: is it climactic? VIX spike? | Wait for volume exhaustion + VIX spike before contrarian buy |

### SIG-F01: Institutional Flow Signal but No Follow-Through
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| One-off rebalancing, not conviction | Check if flow continued for 3+ consecutive days | Require 3-day sustained flow before acting |
| Hedging activity masked as buying | Check options data for concurrent put buying | Cross-reference flow with options activity |
| Market-wide rotation, not stock-specific | Check if entire sector showed same flow | Distinguish stock-specific vs sector-wide flow |

### SIG-F02: Options Unusual Activity but No Move
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Hedge, not directional bet | Check if paired with stock position or other options | Large options trades need context — isolated flow is more reliable |
| Expiry too far out | Check time to expiry > 90 days | Long-dated options may be portfolio hedges, not short-term signals |
| Premiums already priced in the move | Check implied volatility vs historical | High IV = market already expects the move |

---

## Regime Symptoms (RGM-XXX)

### RGM-01: Regime Classification Was Wrong
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Used lagging indicators (200MA) for fast transitions | Price crashed through 200MA in days | Add VIX and breadth as leading indicators; don't rely on MA alone |
| Ignored macro event | Check if Fed decision, geopolitics, or earnings season | During macro events, regime can shift intraday — reduce size preemptively |
| Regional divergence | US bull but TW bear or vice versa | Classify regimes separately for US and TW markets |

### RGM-02: Right Regime but Wrong Signal Filter
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Used bull-market signals in bear regime | Check if signal type matched regime filter table | Stick to regime-appropriate signal types (see playbook) |
| Didn't adjust position size for regime | Took full size in volatile regime | Always apply regime factor to position sizing |
| Ignored sector-level regime | Stock's sector was weak while market was strong | Check sector RS (relative strength) before entry |

---

## Risk & Sizing Symptoms (RSK-XXX)

### RSK-01: Position Too Large (Loss Exceeded Expectation)
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Didn't account for gap risk | Stock gapped past stop-loss | For earnings/events, use options or reduce size; gaps bypass stops |
| Correlated positions amplified loss | Multiple positions in same sector/theme moved together | Count correlated positions as one; limit to 3 correlated max |
| Forgot to apply regime factor | Full size in BEAR-GRIND or BULL-VOLATILE | Always multiply by regime factor before sizing |

### RSK-02: Stop-Loss Hit Then Price Recovered
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Stop too tight for the stock's volatility | Compare stop width to ATR; was stop < 1x ATR? | Stop ≥ 1.5x ATR for volatile stocks; ≥ 1x ATR for calm |
| Stop at obvious level where market makers hunt | Stop was at round number or prior day low | Place stops slightly beyond obvious levels (e.g., -0.5% below support) |
| Normal pullback in an uptrend | Price recovered within 2-3 days | Consider time-based stops for swing trades; allow for pullbacks |

### RSK-03: Took Profit Too Early
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| No profit target plan — exited on emotion | Did you have a target before entry? | Set target at entry: R:R ≥ 2:1; trail stop for runners |
| Scared by minor pullback | Price pulled back < 1 ATR before continuing | Use trailing stop (e.g., 2x ATR below high) instead of discretionary exit |
| Anchored to round-number gain | Exited at +10% or +$X instead of technical target | Use technical levels (resistance, Fibonacci) for targets, not dollar amounts |

---

## Behavioral Symptoms (BHV-XXX)

### BHV-01: Chased Entry After Missing Signal
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| FOMO — saw move happening, jumped in late | Entry was > 2% above signal trigger price | Apply 30-min cooldown after missed signal; wait for pullback |
| Extended from key level at entry | Entry was > 1 ATR from support/MA | If extended, either skip or use tiny size (50% of normal) |
| Rationalized "still room to run" | Re-read entry reasoning — was it emotional? | Write entry thesis BEFORE placing order; if can't, don't trade |

### BHV-02: Held Loser Too Long (Disposition Effect)
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| No stop-loss set at entry | Check if stop was defined pre-entry | MANDATORY: set stop at entry, before confirmation |
| Moved stop-loss lower to avoid being stopped out | Check order history for stop modifications | Never widen a stop-loss. Period. |
| Averaged down | Additional buys after initial loss | No averaging down — this is a hard rule in discipline |

### BHV-03: Overtrading (Too Many Positions / Too Frequent)
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Boredom trading | Low-quality signals; traded because "something should work" | Only trade when ALL checklist items pass; boredom is not a signal |
| Revenge trading after loss | Multiple trades same day after a loss | Circuit breaker: 3 consecutive losses = stop trading for the day |
| Confused activity with productivity | Many trades but no thesis for each | Journal every trade with thesis; review weekly |

### BHV-04: Cut Winner Short, Held Loser Long
| Possible Cause | How to Confirm | Lesson |
|---|---|---|
| Loss aversion (prospect theory) | Average winning hold time < average losing hold time | Use `behavioral/tracker.py` to detect this pattern (BHV-04); enforce symmetric rules |
| No trailing stop system | Exited winners manually but let losers "recover" | Automate: trailing stop for winners, hard stop for losers |

---

## How to Use This Document

1. After a trade closes, identify the most prominent symptom
2. Find the matching code (SIG-XXX, RGM-XXX, RSK-XXX, BHV-XXX)
3. Check each possible cause using the "How to Confirm" column
4. Record the confirmed cause and lesson in `cases/case-NNN.md`
5. If same lesson appears 3+ times, extract to `patterns.md`
6. Review monthly to catch behavioral patterns
