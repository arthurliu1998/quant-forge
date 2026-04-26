"""3-layer portfolio allocation manager.

Spec (Section 3):
- Layer 1: Market allocation (TW/US/Cash) based on regime
- Layer 2: Sector concentration limits
- Layer 3: Individual position sizing (delegated to PositionSizer)
"""
from dataclasses import dataclass
from quantforge.core.models import Regime


# Regime-based market allocation table (spec Section 3.1)
ALLOCATION_TABLE = {
    "both_bull":    {"TW": 0.40, "US": 0.40, "cash": 0.20},
    "tw_strong":    {"TW": 0.50, "US": 0.20, "cash": 0.30},
    "us_strong":    {"TW": 0.20, "US": 0.50, "cash": 0.30},
    "both_bear":    {"TW": 0.15, "US": 0.15, "cash": 0.70},
    "crisis":       {"TW": 0.10, "US": 0.10, "cash": 0.80},
}


@dataclass
class AllocationPlan:
    tw_pct: float
    us_pct: float
    cash_pct: float
    regime_label: str


@dataclass
class PortfolioSnapshot:
    total_capital: float
    holdings: list  # list of dicts: {symbol, market, qty, price, value, sector, weight}
    tw_weight: float
    us_weight: float
    cash_weight: float
    sector_weights: dict  # {sector: weight}
    stock_weights: dict  # {symbol: weight}
    total_exposure: float


class PortfolioManager:
    def __init__(self, sector_max_tw: float = 0.40, sector_max_us: float = 0.35,
                 correlated_max: float = 0.50):
        self.sector_max_tw = sector_max_tw
        self.sector_max_us = sector_max_us
        self.correlated_max = correlated_max

    def get_target_allocation(self, tw_regime: Regime, us_regime: Regime) -> AllocationPlan:
        """Determine target market allocation based on regime pair."""
        if tw_regime == Regime.CRISIS or us_regime == Regime.CRISIS:
            alloc = ALLOCATION_TABLE["crisis"]
            label = "crisis"
        elif tw_regime in (Regime.BEAR_TREND,) and us_regime in (Regime.BEAR_TREND,):
            alloc = ALLOCATION_TABLE["both_bear"]
            label = "both_bear"
        elif tw_regime in (Regime.BULL_TREND, Regime.NEUTRAL) and us_regime in (Regime.BEAR_TREND, Regime.CONSOLIDATION):
            alloc = ALLOCATION_TABLE["tw_strong"]
            label = "tw_strong"
        elif us_regime in (Regime.BULL_TREND, Regime.NEUTRAL) and tw_regime in (Regime.BEAR_TREND, Regime.CONSOLIDATION):
            alloc = ALLOCATION_TABLE["us_strong"]
            label = "us_strong"
        else:
            alloc = ALLOCATION_TABLE["both_bull"]
            label = "both_bull"

        return AllocationPlan(alloc["TW"], alloc["US"], alloc["cash"], label)

    def compute_snapshot(self, total_capital: float,
                         holdings: list[dict]) -> PortfolioSnapshot:
        """Compute current portfolio state from holdings list.

        Each holding dict: {symbol, market, qty, price, sector}
        """
        stock_weights = {}
        sector_weights = {}
        tw_value = 0.0
        us_value = 0.0

        for h in holdings:
            value = h["qty"] * h["price"]
            weight = value / total_capital if total_capital > 0 else 0
            stock_weights[h["symbol"]] = weight

            sector = h.get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

            if h["market"] == "TW":
                tw_value += value
            else:
                us_value += value

        total_invested = tw_value + us_value
        tw_weight = tw_value / total_capital if total_capital > 0 else 0
        us_weight = us_value / total_capital if total_capital > 0 else 0
        cash_weight = 1.0 - tw_weight - us_weight

        return PortfolioSnapshot(
            total_capital=total_capital,
            holdings=holdings,
            tw_weight=round(tw_weight, 4),
            us_weight=round(us_weight, 4),
            cash_weight=round(max(0, cash_weight), 4),
            sector_weights={k: round(v, 4) for k, v in sector_weights.items()},
            stock_weights={k: round(v, 4) for k, v in stock_weights.items()},
            total_exposure=round(total_invested / total_capital if total_capital > 0 else 0, 4),
        )

    def check_sector_limits(self, snapshot: PortfolioSnapshot) -> list[str]:
        """Return warnings for sectors exceeding limits."""
        warnings = []
        for sector, weight in snapshot.sector_weights.items():
            # Use the more restrictive limit
            limit = min(self.sector_max_tw, self.sector_max_us)
            if weight > limit:
                warnings.append(f"Sector '{sector}' at {weight:.1%} exceeds {limit:.0%} limit")
        return warnings
