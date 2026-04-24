"""Alpaca websocket real-time streaming for US stocks.

Requires ALPACA_DATA_KEY and ALPACA_DATA_SECRET.
Uses read-only data keys — never trading keys.
"""
import asyncio
import logging
from typing import Callable, Optional

from trade_forge.secrets import SecretManager

logger = logging.getLogger(__name__)


class AlpacaStream:
    """Real-time US stock data via Alpaca websocket.

    Usage:
        stream = AlpacaStream()
        stream.subscribe_bars(["AAPL", "TSLA"], handler=on_bar)
        await stream.run()  # Blocks, calls handler on each bar
    """

    def __init__(self, paper: bool = True):
        self._paper = paper
        self._symbols: list[str] = []
        self._handler: Optional[Callable] = None
        self._stream = None

    def is_configured(self) -> bool:
        """Check if Alpaca data keys are available."""
        return (
            SecretManager.is_configured("ALPACA_DATA_KEY")
            and SecretManager.is_configured("ALPACA_DATA_SECRET")
        )

    def subscribe_bars(self, symbols: list[str], handler: Callable):
        """Set symbols to stream and the callback handler.

        Args:
            symbols: List of US ticker symbols
            handler: async function called with bar data dict
        """
        self._symbols = symbols
        self._handler = handler
        logger.info("Subscribed to %d symbols for real-time bars", len(symbols))

    async def run(self):
        """Start the websocket stream. Blocks until disconnected."""
        if not self.is_configured():
            logger.warning("Alpaca not configured — streaming disabled")
            return

        if not self._symbols or not self._handler:
            logger.warning("No symbols or handler set — call subscribe_bars() first")
            return

        try:
            from alpaca.data.live import StockDataStream

            key = SecretManager.get("ALPACA_DATA_KEY")
            secret = SecretManager.get("ALPACA_DATA_SECRET")

            self._stream = StockDataStream(key, secret)

            async def _on_bar(bar):
                bar_data = {
                    "symbol": bar.symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "timestamp": str(bar.timestamp),
                }
                try:
                    await self._handler(bar_data)
                except Exception as e:
                    logger.error("Bar handler error for %s: %s", bar.symbol, type(e).__name__)

            self._stream.subscribe_bars(_on_bar, *self._symbols)
            logger.info("Starting Alpaca stream for %s", self._symbols)
            await asyncio.to_thread(self._stream.run)

        except ImportError:
            logger.error("alpaca-py package not installed. Install with: pip install alpaca-py")
        except Exception as e:
            # Never log full error — may contain credentials
            logger.error("Alpaca stream error: %s", type(e).__name__)

    async def stop(self):
        """Stop the stream."""
        if self._stream:
            try:
                self._stream.stop()
                logger.info("Alpaca stream stopped")
            except Exception:
                pass
