"""Order validation gate. All trade orders must pass through OrderGuard before execution."""
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class OrderGuard:
    """Validates trade orders against safety rules."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_single_order_pct = config.get("max_single_order_pct", 10)
        self.max_daily_orders = config.get("max_daily_orders", 20)
        self.max_daily_volume_pct = config.get("max_daily_volume_pct", 30)
        self._daily_orders: list[dict] = []
        self._current_date: date = date.today()

    def _reset_daily_if_needed(self):
        today = date.today()
        if today != self._current_date:
            self._daily_orders = []
            self._current_date = today

    def validate(self, order: dict) -> tuple[bool, str]:
        """Validate an order against safety rules.

        Args:
            order: dict with keys: action, symbol, qty, value_pct, market, type
                   value_pct = order value as % of portfolio

        Returns:
            (approved, reason) tuple
        """
        self._reset_daily_if_needed()

        # 1. Single order size limit
        value_pct = order.get("value_pct", 0)
        if value_pct > self.max_single_order_pct:
            return False, f"Single order {value_pct}% exceeds {self.max_single_order_pct}% limit"

        # 2. Daily order count limit
        if len(self._daily_orders) >= self.max_daily_orders:
            return False, f"Daily order limit ({self.max_daily_orders}) reached"

        # 3. Daily cumulative volume limit
        daily_total = sum(o.get("value_pct", 0) for o in self._daily_orders)
        if daily_total + value_pct > self.max_daily_volume_pct:
            return False, f"Daily volume {daily_total + value_pct:.1f}% exceeds {self.max_daily_volume_pct}% limit"

        # 4. Block sell-all
        if order.get("action") == "SELL" and order.get("sell_all", False):
            return False, "SELL ALL is blocked — must sell specific quantities"

        # 5. Non-limit orders blocked outside market hours
        if not order.get("market_open", True):
            if order.get("type", "LIMIT") != "LIMIT":
                return False, "Non-limit orders blocked outside market hours"

        # Record order for daily tracking
        self._daily_orders.append(order)
        return True, "OK"

    def reset(self):
        """Reset daily counters."""
        self._daily_orders = []
        self._current_date = date.today()
