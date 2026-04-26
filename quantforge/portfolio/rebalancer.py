"""Portfolio rebalancing engine.

Spec (Section 3.2): Triggers on drift > 5%, regime change, monthly, or new signal.
"""
from dataclasses import dataclass
from quantforge.portfolio.manager import PortfolioManager, AllocationPlan, PortfolioSnapshot


@dataclass
class RebalanceAction:
    action_type: str  # "reduce", "increase", "new", "close"
    symbol: str
    market: str
    current_weight: float
    target_weight: float
    delta_pct: float
    reason: str


class Rebalancer:
    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold

    def check_drift(self, snapshot: PortfolioSnapshot,
                    target: AllocationPlan) -> list[RebalanceAction]:
        """Compare current allocation vs target, return rebalance actions if drift > threshold."""
        actions = []

        # Market-level drift
        tw_drift = snapshot.tw_weight - target.tw_pct
        us_drift = snapshot.us_weight - target.us_pct

        if abs(tw_drift) > self.drift_threshold:
            action = "reduce" if tw_drift > 0 else "increase"
            actions.append(RebalanceAction(
                action_type=action, symbol="TW_MARKET", market="TW",
                current_weight=snapshot.tw_weight, target_weight=target.tw_pct,
                delta_pct=round(tw_drift, 4),
                reason=f"TW allocation drifted {tw_drift:+.1%} from target {target.tw_pct:.0%}",
            ))

        if abs(us_drift) > self.drift_threshold:
            action = "reduce" if us_drift > 0 else "increase"
            actions.append(RebalanceAction(
                action_type=action, symbol="US_MARKET", market="US",
                current_weight=snapshot.us_weight, target_weight=target.us_pct,
                delta_pct=round(us_drift, 4),
                reason=f"US allocation drifted {us_drift:+.1%} from target {target.us_pct:.0%}",
            ))

        # Stock-level drift (single stock > 20%)
        for symbol, weight in snapshot.stock_weights.items():
            if weight > 0.20:
                actions.append(RebalanceAction(
                    action_type="reduce", symbol=symbol,
                    market="TW" if any(h["symbol"] == symbol and h["market"] == "TW"
                                       for h in snapshot.holdings) else "US",
                    current_weight=weight, target_weight=0.20,
                    delta_pct=round(weight - 0.20, 4),
                    reason=f"{symbol} at {weight:.1%} exceeds 20% single-stock limit",
                ))

        return actions

    def needs_rebalance(self, snapshot: PortfolioSnapshot,
                        target: AllocationPlan) -> bool:
        """Quick check if any rebalancing is needed."""
        return len(self.check_drift(snapshot, target)) > 0
