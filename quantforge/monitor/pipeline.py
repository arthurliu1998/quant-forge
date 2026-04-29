"""AnalysisPipeline: orchestrates signal -> scoring -> LLM -> report -> notify.

Includes multi-mode strategy recommendation (TREND_FOLLOW / HIGHVOL_BREAKOUT /
WHEEL / CASH) based on regime and annualized volatility.
"""
import asyncio
import logging
from typing import Optional

from quantforge.core.models import QuantSignal, Regime
from quantforge.monitor.report_builder import ReportBuilder
from quantforge.monitor.report_store import ReportStore
from quantforge.signals.engine import Signal
from quantforge.strategy.multimode import determine_mode, compute_annualized_vol

logger = logging.getLogger(__name__)

SIGNAL_ANALYSIS_PROMPT = """You are a quantitative trading analyst. Analyze the following stock signal data and provide:
1. Event interpretation — what is driving this signal
2. Risk factors — what could go wrong
3. Confidence assessment — how reliable is this signal
Keep your analysis concise (under 200 words). Use percentages, not absolute dollar amounts."""

BRIEFING_PROMPT = """You are a market briefing analyst. Given the following watchlist scores and market regime data, produce a concise pre-market briefing:
1. Market environment summary (2-3 sentences)
2. Top opportunities with reasoning
3. Key risks to watch today
Keep it under 300 words. Use percentages, not absolute dollar amounts."""


class AnalysisPipeline:
    """Orchestrate scoring -> LLM analysis -> report generation -> notification."""

    def __init__(self, config: dict, notifier, llm_router, finbert=None):
        self.config = config
        self.notifier = notifier
        self.llm = llm_router
        self.finbert = finbert

        # Lazy imports to avoid import-time failures when data modules
        # have unresolved dependencies (e.g., fetch_us not present yet)
        from quantforge.scanner import QuantScanner
        from quantforge.regime.detector import RegimeDetector
        self.scanner = QuantScanner()
        self.regime_detector = RegimeDetector()
        self.data_provider = None  # lazily initialized in _score_symbol

        monitor_cfg = config.get("monitor", {})
        reports_cfg = monitor_cfg.get("reports", {})
        self.report_store = ReportStore(
            base_dir=reports_cfg.get("output_dir", "~/.quantforge/reports"),
            retention_days=reports_cfg.get("retention_days", 30),
        )
        self._semaphore = asyncio.Semaphore(3)

    def _should_analyze(self, signal: Signal) -> bool:
        """Only analyze HIGH and CRITICAL signals."""
        return signal.priority in ("HIGH", "CRITICAL")

    def _get_data_provider(self):
        if self.data_provider is None:
            from quantforge.data.yfinance_provider import YFinanceProvider
            self.data_provider = YFinanceProvider()
        return self.data_provider

    def _score_symbol(self, symbol: str, market: str = "US",
                      sentiment_data: dict = None) -> Optional[QuantSignal]:
        """Run QuantScanner on a single symbol."""
        try:
            ohlcv = self._get_data_provider().get_ohlcv(symbol, period="6mo")
            if ohlcv.empty:
                return None
            return self.scanner.score_stock(
                symbol=symbol, market=market, ohlcv=ohlcv,
                sentiment_data=sentiment_data,
            )
        except Exception as e:
            logger.error("Scoring failed for %s: %s", symbol, type(e).__name__)
            return None

    async def run(self, signal: Signal) -> None:
        """Analyze a detected signal: score -> LLM -> report -> notify."""
        if not self._should_analyze(signal):
            return

        async with self._semaphore:
            logger.info("Analyzing signal: [%s] %s %s",
                        signal.priority, signal.symbol, signal.type)

            market = self._detect_market(signal.symbol)
            quant_signal = self._score_symbol(signal.symbol, market)
            if quant_signal is None:
                logger.warning("Could not score %s -- skipping", signal.symbol)
                return

            # Multi-mode strategy recommendation
            mode, mode_reason = self._recommend_mode(signal.symbol, market)

            llm_result = None
            if self.llm and self.llm.has_providers():
                try:
                    llm_result = await self.llm.analyze(SIGNAL_ANALYSIS_PROMPT, {
                        "symbol": signal.symbol,
                        "signal_type": signal.type,
                        "quant_score": quant_signal.quant_score,
                        "regime": quant_signal.regime.value,
                        "signal_level": quant_signal.signal_level,
                        "recommended_mode": mode,
                        "mode_reason": mode_reason,
                    })
                except Exception as e:
                    logger.warning("LLM analysis failed: %s", type(e).__name__)

            summary, full = ReportBuilder.build_signal(
                quant_signal, llm_result, trigger=signal.type,
                mode=mode, mode_reason=mode_reason)
            self.report_store.save("signal", signal.symbol, full)
            self.notifier.send_text(summary)
            logger.info("Analysis complete for %s: %s (mode: %s)",
                        signal.symbol, quant_signal.signal_level, mode)

    async def run_briefing(self, briefing_type: str) -> None:
        """Generate scheduled briefing for all watchlist symbols."""
        async with self._semaphore:
            logger.info("Generating %s briefing", briefing_type)
            watchlist = self.config.get("watchlist", {})
            all_symbols = [(s, "US") for s in watchlist.get("US", [])]
            all_symbols += [(s, "TW") for s in watchlist.get("TW", [])]

            signals = []
            mode_recs = {}
            for symbol, market in all_symbols:
                qs = self._score_symbol(symbol, market)
                if qs:
                    signals.append(qs)
                    mode, reason = self._recommend_mode(symbol, market)
                    mode_recs[symbol] = {"mode": mode, "reason": reason}

            regime = signals[0].regime if signals else Regime.NEUTRAL
            vix = 15.0

            llm_result = None
            if self.llm and self.llm.has_providers():
                try:
                    llm_result = await self.llm.analyze(BRIEFING_PROMPT, {
                        "briefing_type": briefing_type,
                        "regime": regime.value,
                        "vix": vix,
                        "signals": [
                            {"symbol": s.symbol, "score": s.quant_score,
                             "level": s.signal_level,
                             "mode": mode_recs.get(s.symbol, {}).get("mode", "")}
                            for s in signals
                        ],
                    })
                except Exception as e:
                    logger.warning("LLM briefing failed: %s", type(e).__name__)

            summary, full = ReportBuilder.build_briefing(
                briefing_type, signals, regime, vix, llm_result,
                mode_recs=mode_recs)
            self.report_store.save("briefing", briefing_type, full)
            self.notifier.send_text(summary)
            logger.info("Briefing %s complete", briefing_type)

    async def run_recalc(self, symbol: str, trigger: str) -> None:
        """Instant edge recalculation -- always runs full analysis."""
        async with self._semaphore:
            logger.info("Instant recalc for %s (trigger: %s)", symbol, trigger)
            market = self._detect_market(symbol)
            quant_signal = self._score_symbol(symbol, market)
            if quant_signal is None:
                return

            mode, mode_reason = self._recommend_mode(symbol, market)

            llm_result = None
            if self.llm and self.llm.has_providers():
                try:
                    llm_result = await self.llm.analyze(SIGNAL_ANALYSIS_PROMPT, {
                        "symbol": symbol,
                        "signal_type": f"recalc:{trigger}",
                        "quant_score": quant_signal.quant_score,
                        "regime": quant_signal.regime.value,
                        "signal_level": quant_signal.signal_level,
                        "recommended_mode": mode,
                        "mode_reason": mode_reason,
                    })
                except Exception as e:
                    logger.warning("LLM recalc failed: %s", type(e).__name__)

            summary, full = ReportBuilder.build_signal(
                quant_signal, llm_result, trigger=f"recalc:{trigger}",
                mode=mode, mode_reason=mode_reason)
            self.report_store.save("recalc", symbol, full)
            self.notifier.send_text(summary)

    def _detect_market(self, symbol: str) -> str:
        """Detect if symbol is US or TW based on watchlist config."""
        tw_list = self.config.get("watchlist", {}).get("TW", [])
        return "TW" if symbol in tw_list else "US"

    def _recommend_mode(self, symbol: str, market: str = "US") -> tuple[str, str]:
        """Recommend a multi-mode strategy for the symbol.

        Returns (mode, reason) based on current regime and annualized volatility.
        """
        try:
            import pandas as pd
            ohlcv = self._get_data_provider().get_ohlcv(symbol, period="6mo")
            if ohlcv.empty or len(ohlcv) < 60:
                return "CASH", "Insufficient data"

            regime = self.regime_detector.detect_from_data(ohlcv)
            ann_vol = compute_annualized_vol(ohlcv)
            latest_vol = float(ann_vol.iloc[-1]) if not pd.isna(ann_vol.iloc[-1]) else 0.25
            mode = determine_mode(regime, latest_vol)

            reasons = {
                "TREND_FOLLOW": f"Bull trend, vol {latest_vol:.0%} -- trail 3x ATR",
                "HIGHVOL_BREAKOUT": f"Bull + high vol {latest_vol:.0%} -- trail 4x ATR, inverse-vol sizing",
                "WHEEL": f"Sideways ({regime.value}), vol {latest_vol:.0%} -- sell CSP/CC for income",
                "CASH": f"Bear/crisis ({regime.value}) -- no new trades",
            }
            return mode, reasons.get(mode, mode)
        except Exception as e:
            logger.warning("Mode recommendation failed for %s: %s", symbol, type(e).__name__)
            return "CASH", "Error computing mode"
