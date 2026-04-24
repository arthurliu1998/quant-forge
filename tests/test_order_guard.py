import pytest
from trade_forge.order_guard import OrderGuard


@pytest.fixture
def guard():
    return OrderGuard({"max_single_order_pct": 10, "max_daily_orders": 3, "max_daily_volume_pct": 30})


class TestOrderGuard:
    def test_approve_normal_order(self, guard):
        order = {"action": "BUY", "symbol": "TSLA", "value_pct": 5, "type": "LIMIT"}
        approved, reason = guard.validate(order)
        assert approved is True
        assert reason == "OK"

    def test_reject_oversized_order(self, guard):
        order = {"action": "BUY", "symbol": "TSLA", "value_pct": 15, "type": "LIMIT"}
        approved, reason = guard.validate(order)
        assert approved is False
        assert "exceeds" in reason

    def test_reject_daily_order_limit(self, guard):
        for i in range(3):
            guard.validate({"action": "BUY", "symbol": f"SYM{i}", "value_pct": 5, "type": "LIMIT"})
        approved, reason = guard.validate({"action": "BUY", "symbol": "EXTRA", "value_pct": 5, "type": "LIMIT"})
        assert approved is False
        assert "Daily order limit" in reason

    def test_reject_daily_volume_limit(self):
        g = OrderGuard({"max_single_order_pct": 20, "max_daily_orders": 10, "max_daily_volume_pct": 25})
        g.validate({"action": "BUY", "symbol": "A", "value_pct": 10, "type": "LIMIT"})
        g.validate({"action": "BUY", "symbol": "B", "value_pct": 10, "type": "LIMIT"})
        approved, reason = g.validate({"action": "BUY", "symbol": "C", "value_pct": 10, "type": "LIMIT"})
        assert approved is False
        assert "Daily volume" in reason

    def test_reject_sell_all(self, guard):
        order = {"action": "SELL", "symbol": "TSLA", "value_pct": 5, "sell_all": True, "type": "LIMIT"}
        approved, reason = guard.validate(order)
        assert approved is False
        assert "SELL ALL" in reason

    def test_reject_market_order_outside_hours(self, guard):
        order = {"action": "BUY", "symbol": "TSLA", "value_pct": 5, "type": "MARKET", "market_open": False}
        approved, reason = guard.validate(order)
        assert approved is False
        assert "outside market hours" in reason
