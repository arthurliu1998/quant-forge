"""Edge Layer 2: Chipflow factor scoring (Taiwan stocks only)."""
from typing import Optional
from quantforge.factors.base import Factor
from quantforge.core.models import FactorScore


class ChipflowFactor(Factor):
    def __init__(self, weight: float = 0.30):
        super().__init__("chipflow", weight)

    def compute(self, symbol: str, data: dict) -> Optional[FactorScore]:
        if data.get("market") == "US":
            return None

        flow = data.get("institutional_flow")
        if not flow:
            return FactorScore(self.name, 0.5, 0.5, self.weight)

        points = 0.0

        # C1: Foreign investor consecutive days
        fd = flow.get("foreign_net_consecutive_days", 0)
        if fd >= 3: points += 1.0
        elif fd == 2: points += 0.5
        elif fd <= -3: points -= 1.0

        # C2: Investment trust same direction
        if flow.get("trust_same_direction", False):
            points += 1.0

        # C3: Foreign net buy volume ratio
        ratio = flow.get("foreign_net_volume_ratio", 0.0)
        if ratio > 0.10: points += 1.0
        elif ratio > 0.05: points += 0.5

        # C4: Margin balance change
        mc = flow.get("margin_change_pct", 0.0)
        if mc < -5.0: points += 1.0
        elif mc < -2.0: points += 0.5
        elif mc > 10.0: points -= 1.0

        # C5: Short-selling balance change
        sc = flow.get("short_change_pct", 0.0)
        if sc < -10.0: points += 1.0
        elif sc > 20.0: points -= 1.0

        normalized = (points + 5.0) / 10.0
        normalized = max(0.0, min(1.0, normalized))
        return FactorScore(self.name, points, normalized, self.weight)
