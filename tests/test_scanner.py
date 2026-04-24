import pytest
from trade_forge.monitor.scanner import WatchlistScanner
from trade_forge.signals.engine import Signal


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
    }


class TestWatchlistScanner:
    def test_scan_all_returns_list(self, config):
        scanner = WatchlistScanner(config)
        signals = scanner.scan_all()
        assert isinstance(signals, list)
        assert all(isinstance(s, Signal) for s in signals)

    def test_scan_symbol_us(self, config):
        scanner = WatchlistScanner(config)
        signals = scanner.scan_symbol("AAPL", market="US")
        assert isinstance(signals, list)

    def test_scan_invalid_symbol_returns_empty(self, config):
        scanner = WatchlistScanner(config)
        signals = scanner.scan_symbol("ZZZZNOTREAL", market="US")
        assert signals == []

    def test_empty_watchlist(self):
        scanner = WatchlistScanner({"watchlist": {"US": [], "TW": []}, "signals": {}})
        signals = scanner.scan_all()
        assert signals == []
