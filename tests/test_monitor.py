import pytest
from trade_forge.monitor.monitor import TradeMonitor


@pytest.fixture
def config():
    return {
        "watchlist": {"US": ["AAPL"], "TW": []},
        "signals": {
            "ma_crossover": [20, 60],
            "volume_spike_ratio": 2.0,
            "rsi_overbought": 80,
            "rsi_oversold": 20,
        },
        "notification": {"sensitivity": "medium"},
    }


class TestTradeMonitor:
    def test_init(self, config):
        monitor = TradeMonitor(config)
        assert monitor.scanner is not None
        assert monitor.alpaca is not None
        assert monitor.notifier is not None
        assert monitor._running is False

    def test_scanner_has_watchlist(self, config):
        monitor = TradeMonitor(config)
        assert "AAPL" in monitor.scanner.watchlist_us

    @pytest.mark.asyncio
    async def test_scheduled_scan_does_not_crash(self, config):
        """Scheduled scan should run without error even if no signals found."""
        monitor = TradeMonitor(config)
        await monitor._run_scheduled_scan()

    @pytest.mark.asyncio
    async def test_stop(self, config):
        monitor = TradeMonitor(config)
        monitor._running = True
        await monitor.stop()
        assert monitor._running is False
