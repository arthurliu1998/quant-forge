"""LLM-based signal filter for backtesting with anti-look-ahead bias.

De-identifies market data (removes symbols, dates, absolute prices) before
sending to the LLM, so it cannot exploit knowledge of historical events.
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quantforge.providers.base import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a quantitative trading signal evaluator. Judge ONLY from the "
    "technical data provided. Do NOT identify the stock or use external "
    "knowledge.\n\n"
    "CRITICAL: Your entire response must be a single JSON object. "
    "No explanation, no markdown, no text before or after the JSON.\n\n"
    "Required format:\n"
    '{"confidence": <0-100>, "action": "ENTER" or "SKIP", '
    '"reason": "<one sentence>"}'
)


@dataclass
class LLMSignalVerdict:
    """Result of LLM evaluation for a single signal."""
    action: str          # "ENTER" or "SKIP"
    confidence: int      # 0-100
    reason: str
    cached: bool = False


class BacktestLLMFilter:
    """Filters backtest signals through an LLM with anti-look-ahead bias.

    De-identifies all market data before sending to the LLM:
    - Replaces stock symbol with "STOCK_A"
    - Replaces real dates with relative day offsets
    - Normalizes prices to percentage changes
    - Only sends technical indicator values

    Args:
        provider: LLM provider instance (e.g. ClaudeProvider)
        confidence_threshold: minimum confidence to accept (0-100)
        batch_delay: seconds between API calls for rate limiting
        cache_results: whether to cache responses by data hash
    """

    def __init__(
        self,
        provider: LLMProvider,
        confidence_threshold: int = 60,
        batch_delay: float = 0.5,
        cache_results: bool = True,
    ):
        self.provider = provider
        self.confidence_threshold = confidence_threshold
        self.batch_delay = batch_delay
        self._cache: dict[str, LLMSignalVerdict] = {} if cache_results else None

    async def filter_signals(
        self,
        df: pd.DataFrame,
        signals: list[dict],
        indicators: dict[str, pd.Series],
    ) -> list[dict]:
        """Filter signals through LLM, returning only approved ones.

        Args:
            df: OHLCV DataFrame
            signals: list of {index, direction, score, ...}
            indicators: dict of indicator name -> pd.Series

        Returns:
            Filtered list of signals where LLM said ENTER with
            confidence >= threshold.
        """
        if not signals:
            return []

        approved = []
        for i, sig in enumerate(signals):
            verdict = await self._evaluate_signal(df, sig, indicators)
            if verdict.action == "ENTER" and verdict.confidence >= self.confidence_threshold:
                approved.append(sig)
                logger.debug(
                    "Signal %d APPROVED (confidence=%d): %s",
                    sig["index"], verdict.confidence, verdict.reason,
                )
            else:
                logger.debug(
                    "Signal %d REJECTED (action=%s, confidence=%d): %s",
                    sig["index"], verdict.action, verdict.confidence, verdict.reason,
                )

            # Rate limiting between API calls (skip if cached)
            if not verdict.cached and i < len(signals) - 1:
                await asyncio.sleep(self.batch_delay)

        logger.info(
            "LLM filter: %d/%d signals approved (threshold=%d)",
            len(approved), len(signals), self.confidence_threshold,
        )
        return approved

    async def _evaluate_signal(
        self,
        df: pd.DataFrame,
        signal: dict,
        indicators: dict[str, pd.Series],
    ) -> LLMSignalVerdict:
        """Evaluate a single signal through the LLM."""
        deidentified = self._deidentify(df, signal["index"], indicators)
        prompt_data = {
            "signal_type": signal.get("type", "MA crossover"),
            "direction": signal.get("direction", "long"),
            **deidentified,
        }

        # Check cache
        if self._cache is not None:
            cache_key = self._compute_cache_key(prompt_data)
            if cache_key in self._cache:
                verdict = self._cache[cache_key]
                return LLMSignalVerdict(
                    action=verdict.action,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                    cached=True,
                )

        # Call LLM with retry
        verdict = None
        for attempt in range(3):
            try:
                response = await self.provider.analyze(SYSTEM_PROMPT, prompt_data)
                verdict = self._parse_response(response)
                break
            except Exception as e:
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))  # exponential backoff
        if verdict is None:
            verdict = LLMSignalVerdict(action="SKIP", confidence=0, reason="API error")

        # Store in cache
        if self._cache is not None:
            self._cache[cache_key] = verdict

        return verdict

    def _deidentify(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        indicators: dict[str, pd.Series],
    ) -> dict:
        """De-identify market data to prevent look-ahead bias.

        Removes stock symbol, real dates, and absolute prices.
        Returns only normalized percentage changes and indicator values.
        """
        # Price -> normalized % changes over lookback window
        lookback = 20
        start = max(0, signal_idx - lookback)
        window = df["Close"].iloc[start:signal_idx + 1]
        base = float(window.iloc[0]) if len(window) > 0 else 1.0
        if base <= 0:
            base = 1.0
        normalized_trend = ((window / base) - 1.0) * 100
        price_trend = [round(v, 2) for v in normalized_trend.tolist()]

        # Indicator snapshot at signal time — only current values
        snapshot = {}
        indicator_keys = [
            "rsi", "macd_hist", "adx", "vol_ratio_5d",
            "k", "d", "bb_upper", "bb_mid", "bb_lower",
        ]
        for key in indicator_keys:
            if key in indicators:
                val = indicators[key].iloc[signal_idx]
                if pd.notna(val):
                    # For Bollinger bands, express as % distance from close
                    if key in ("bb_upper", "bb_mid", "bb_lower"):
                        close = float(df["Close"].iloc[signal_idx])
                        if close > 0:
                            snapshot[f"{key}_pct"] = round(
                                (float(val) - close) / close * 100, 2
                            )
                    else:
                        snapshot[key] = round(float(val), 2)

        # MACD histogram trend (last 3 values)
        if "macd_hist" in indicators:
            hist_start = max(0, signal_idx - 2)
            hist_vals = indicators["macd_hist"].iloc[hist_start:signal_idx + 1]
            hist_clean = [round(float(v), 4) for v in hist_vals if pd.notna(v)]
            if hist_clean:
                snapshot["macd_hist_trend"] = hist_clean

        return {
            "stock": "STOCK_A",
            "day": "Day 0",
            "price_trend_pct": price_trend,
            **snapshot,
        }

    def _parse_response(self, response: dict) -> LLMSignalVerdict:
        """Parse LLM response into a verdict, with fallback for bad output."""
        content = response.get("content", "")

        # Try to extract JSON from the response
        parsed = None
        text = content.strip()
        # Strategy 1: direct parse (response is pure JSON)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        # Strategy 2: extract from markdown code block
        if parsed is None and "```" in text:
            try:
                block = text.split("```", 2)[1]
                if block.startswith("json"):
                    block = block[4:]
                parsed = json.loads(block.strip())
            except (json.JSONDecodeError, IndexError, ValueError):
                pass
        # Strategy 3: find JSON object anywhere in the text
        if parsed is None:
            import re
            match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+?"[^{}]*\}', text)
            if match:
                try:
                    parsed = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass
        if parsed is None:
            logger.warning("LLM returned non-JSON response, defaulting to SKIP")
            return LLMSignalVerdict(action="SKIP", confidence=0, reason="parse error")

        action = str(parsed.get("action", "SKIP")).upper()
        if action not in ("ENTER", "SKIP"):
            action = "SKIP"

        confidence = parsed.get("confidence", 0)
        try:
            confidence = int(confidence)
            confidence = max(0, min(100, confidence))
        except (ValueError, TypeError):
            confidence = 0

        reason = str(parsed.get("reason", ""))[:200]

        return LLMSignalVerdict(action=action, confidence=confidence, reason=reason)

    @staticmethod
    def _compute_cache_key(data: dict) -> str:
        """Compute a stable hash for the de-identified data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    @property
    def cache_size(self) -> int:
        """Number of cached LLM responses."""
        return len(self._cache) if self._cache is not None else 0

    def clear_cache(self):
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()
