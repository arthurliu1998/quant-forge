"""Watchlist scanner — scans all symbols and returns detected signals."""
import logging
from trade_forge.data.fetch_us import fetch_ohlcv
from trade_forge.data.fetch_tw import fetch_tw_daily
from trade_forge.analysis.indicators import compute_all
from trade_forge.signals.engine import SignalEngine, Signal

logger = logging.getLogger(__name__)


class WatchlistScanner:
    """Scan a watchlist of symbols for trading signals."""

    def __init__(self, config: dict):
        self.watchlist_us = config.get("watchlist", {}).get("US", [])
        self.watchlist_tw = config.get("watchlist", {}).get("TW", [])
        self.signal_engine = SignalEngine(config.get("signals", {}))

    def scan_all(self) -> list[Signal]:
        """Scan all watchlist symbols. Returns list of detected signals."""
        signals = []
        for symbol in self.watchlist_us:
            signals.extend(self._scan_us(symbol))
        for symbol in self.watchlist_tw:
            signals.extend(self._scan_tw(symbol))
        return signals

    def scan_symbol(self, symbol: str, market: str = "US") -> list[Signal]:
        """Scan a single symbol."""
        if market.upper() == "TW":
            return self._scan_tw(symbol)
        return self._scan_us(symbol)

    def _scan_us(self, symbol: str) -> list[Signal]:
        """Fetch US data, compute indicators, detect signals."""
        try:
            df = fetch_ohlcv(symbol, period="6mo")
            if df.empty:
                logger.warning("No data for US:%s", symbol)
                return []
            indicators = compute_all(df)
            data = {**indicators, "close": df["Close"], "volume": df["Volume"]}
            return self.signal_engine.detect(symbol, data)
        except Exception as e:
            logger.error("Error scanning US:%s: %s", symbol, e)
            return []

    def _scan_tw(self, symbol: str) -> list[Signal]:
        """Fetch TW data, compute indicators, detect signals."""
        try:
            df = fetch_tw_daily(symbol)
            if df.empty:
                logger.warning("No data for TW:%s", symbol)
                return []
            # TW data uses lowercase columns — normalize to match indicators
            df_norm = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            indicators = compute_all(df_norm)
            data = {**indicators, "close": df_norm["Close"], "volume": df_norm["Volume"]}
            return self.signal_engine.detect(symbol, data)
        except Exception as e:
            logger.error("Error scanning TW:%s: %s", symbol, e)
            return []
