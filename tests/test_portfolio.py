import os
import pytest
from trade_forge.portfolio import PortfolioDB, Position, Trade


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_portfolio.db")
    portfolio = PortfolioDB(db_path)
    yield portfolio
    portfolio.close()


class TestPositions:
    def test_add_and_get_position(self, db):
        pos = Position(symbol="TSLA", market="US", qty=10, avg_cost=248.0)
        pid = db.add_position(pos)
        assert pid > 0
        positions = db.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "TSLA"
        assert positions[0].qty == 10

    def test_close_position(self, db):
        pos = Position(symbol="AAPL", market="US", qty=50, avg_cost=180.0)
        pid = db.add_position(pos)
        db.close_position(pid)
        open_pos = db.get_open_positions()
        assert len(open_pos) == 0
        all_pos = db.get_all_positions()
        assert len(all_pos) == 1
        assert all_pos[0].status == "closed"

    def test_multiple_positions(self, db):
        db.add_position(Position("TSLA", "US", 10, 248.0))
        db.add_position(Position("AAPL", "US", 50, 180.0))
        db.add_position(Position("2330", "TW", 1000, 580.0))
        positions = db.get_open_positions()
        assert len(positions) == 3


class TestTrades:
    def test_record_and_get_trade(self, db):
        trade = Trade(symbol="TSLA", market="US", action="BUY", qty=10, price=248.0,
                      reason="Tech bullish", signal_type="ma20_crossover_up")
        tid = db.record_trade(trade)
        assert tid > 0
        trades = db.get_trades()
        assert len(trades) == 1
        assert trades[0].symbol == "TSLA"
        assert trades[0].action == "BUY"

    def test_filter_by_symbol(self, db):
        db.record_trade(Trade("TSLA", "US", "BUY", 10, 248.0))
        db.record_trade(Trade("AAPL", "US", "BUY", 50, 180.0))
        tsla_trades = db.get_trades(symbol="TSLA")
        assert len(tsla_trades) == 1
        assert tsla_trades[0].symbol == "TSLA"

    def test_auto_timestamp(self, db):
        trade = Trade("TSLA", "US", "BUY", 10, 248.0)
        db.record_trade(trade)
        trades = db.get_trades()
        assert trades[0].timestamp != ""


class TestPerformance:
    def test_empty_performance(self, db):
        perf = db.get_performance()
        assert perf["total_trades"] == 0

    def test_performance_with_trades(self, db):
        db.record_trade(Trade("TSLA", "US", "BUY", 10, 248.0))
        db.record_trade(Trade("TSLA", "US", "SELL", 10, 265.0))
        perf = db.get_performance()
        assert perf["total_trades"] == 2
        assert perf["buy_count"] == 1
        assert perf["sell_count"] == 1
