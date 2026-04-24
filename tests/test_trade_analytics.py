import pytest
from trade_forge.portfolio import PortfolioDB, Trade
from trade_forge.trade_analytics import TradeAnalytics, AnalyticsReport


@pytest.fixture
def db(tmp_path):
    portfolio = PortfolioDB(str(tmp_path / "test.db"))
    yield portfolio
    portfolio.close()


@pytest.fixture
def db_with_trades(db):
    """DB with a mix of winning and losing trades."""
    trades = [
        Trade("TSLA", "US", "BUY", 10, 240.0, "2026-01-01", "reason", "ma20_crossover_up"),
        Trade("TSLA", "US", "SELL", 10, 260.0, "2026-01-15", "take profit", ""),
        Trade("AAPL", "US", "BUY", 50, 180.0, "2026-01-05", "reason", "rsi_oversold"),
        Trade("AAPL", "US", "SELL", 50, 170.0, "2026-01-20", "stop loss", ""),
        Trade("NVDA", "US", "BUY", 20, 500.0, "2026-02-01", "reason", "ma20_crossover_up"),
        Trade("NVDA", "US", "SELL", 20, 550.0, "2026-02-15", "target", ""),
    ]
    for t in trades:
        db.record_trade(t)
    return db


class TestTradeAnalytics:
    def test_empty_db(self, db):
        analytics = TradeAnalytics(db)
        report = analytics.analyze()
        assert isinstance(report, AnalyticsReport)
        assert report.total_trades == 0

    def test_basic_metrics(self, db_with_trades):
        analytics = TradeAnalytics(db_with_trades)
        report = analytics.analyze()
        assert report.total_trades == 3  # 3 buy-sell pairs
        assert report.win_rate > 0

    def test_win_rate_correct(self, db_with_trades):
        analytics = TradeAnalytics(db_with_trades)
        report = analytics.analyze()
        # 2 wins (TSLA +8.3%, NVDA +10%), 1 loss (AAPL -5.6%)
        assert report.win_rate == pytest.approx(66.7, abs=0.1)

    def test_best_worst_signal(self, db_with_trades):
        analytics = TradeAnalytics(db_with_trades)
        report = analytics.analyze()
        # ma20_crossover_up: 2 wins / 2 total = 100%
        # rsi_oversold: 0 wins / 1 total = 0%
        assert report.best_signal == "ma20_crossover_up"
        assert report.worst_signal == "rsi_oversold"

    def test_flags_list(self, db_with_trades):
        analytics = TradeAnalytics(db_with_trades)
        report = analytics.analyze()
        assert isinstance(report.flags, list)
