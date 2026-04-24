"""
TradeForge Background Monitor -- main entry point.

Combines:
- Alpaca websocket streaming (real-time US)
- Scheduled scans (watchlist, TW daily, US daily, morning brief)
- Telegram notifications

Usage:
    python -m trade_forge.monitor.monitor --config ~/.trade-forge/config.yaml
"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

from trade_forge.config import load_config
from trade_forge.monitor.secure_logger import setup_logging
from trade_forge.monitor.scanner import WatchlistScanner
from trade_forge.monitor.alpaca_stream import AlpacaStream
from trade_forge.notify.telegram import TelegramNotifier
from trade_forge.secrets import SecretManager
from trade_forge.safe_exceptions import install_exception_hooks

logger = logging.getLogger(__name__)


class TradeMonitor:
    """Background monitor that combines real-time streaming and scheduled scans."""

    def __init__(self, config: dict):
        self.config = config
        self.scanner = WatchlistScanner(config)
        self.alpaca = AlpacaStream()

        # Set up Telegram notifier
        token = SecretManager.get("TELEGRAM_BOT_TOKEN")
        chat_id = SecretManager.get("TELEGRAM_CHAT_ID")
        sensitivity = config.get("notification", {}).get("sensitivity", "medium")
        self.notifier = TelegramNotifier(token, chat_id, sensitivity)

        self._running = False

    async def run(self):
        """Start the monitor. Runs until stopped."""
        self._running = True
        logger.info("TradeForge Monitor starting...")

        tasks = [self._scheduled_loop()]

        # Add Alpaca streaming if configured
        if self.alpaca.is_configured():
            us_symbols = self.config.get("watchlist", {}).get("US", [])
            if us_symbols:
                self.alpaca.subscribe_bars(us_symbols, handler=self._on_realtime_bar)
                tasks.append(self.alpaca.run())
                logger.info("Alpaca real-time enabled for %d symbols", len(us_symbols))
        else:
            logger.info("Alpaca not configured -- real-time streaming disabled")

        if self.notifier.is_configured():
            self.notifier.send_text("TradeForge Monitor started")
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram not configured -- notifications disabled")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Monitor shutting down...")
        finally:
            self._running = False
            await self.alpaca.stop()

    async def stop(self):
        """Stop the monitor."""
        self._running = False
        await self.alpaca.stop()

    async def _scheduled_loop(self):
        """Run scheduled tasks (scans, briefs) based on time."""
        scan_interval = 1800  # 30 minutes between full scans
        last_scan = 0

        while self._running:
            now = datetime.now()
            timestamp = now.timestamp()

            # Full watchlist scan every 30 minutes
            if timestamp - last_scan >= scan_interval:
                await self._run_scheduled_scan()
                last_scan = timestamp

            await asyncio.sleep(60)  # Check every minute

    async def _run_scheduled_scan(self):
        """Run a full watchlist scan and push any signals."""
        logger.info("Running scheduled watchlist scan...")
        try:
            signals = self.scanner.scan_all()
            if signals:
                logger.info("Found %d signals", len(signals))
                for signal in signals:
                    self.notifier.send_signal(signal)
                    logger.info("[%s] %s: %s", signal.priority, signal.symbol, signal.message)
            else:
                logger.info("No signals detected")
        except Exception as e:
            logger.error("Scan error: %s", type(e).__name__)

    async def _on_realtime_bar(self, bar: dict):
        """Handle real-time bar from Alpaca stream."""
        symbol = bar.get("symbol", "")
        try:
            signals = self.scanner.scan_symbol(symbol, market="US")
            for signal in signals:
                self.notifier.send_signal(signal)
                logger.info("RT [%s] %s: %s", signal.priority, symbol, signal.message)
        except Exception as e:
            logger.error("Real-time scan error for %s: %s", symbol, type(e).__name__)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TradeForge Background Monitor")
    parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.trade-forge/config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log-dir",
        default=os.path.expanduser("~/.trade-forge/logs"),
        help="Directory for log files",
    )
    args = parser.parse_args()

    # Install security hooks
    install_exception_hooks()

    # Setup logging
    setup_logging(log_dir=args.log_dir)

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error("Config not found: %s -- run install.sh first", args.config)
        sys.exit(1)

    # Run monitor
    monitor = TradeMonitor(config)
    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")


if __name__ == "__main__":
    main()
