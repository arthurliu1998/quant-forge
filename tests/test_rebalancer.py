from quantforge.portfolio.rebalancer import Rebalancer, RebalanceAction
from quantforge.portfolio.manager import PortfolioManager, AllocationPlan, PortfolioSnapshot
from quantforge.core.models import Regime


def _snapshot(tw=0.40, us=0.40, stocks=None):
    total = 100000
    holdings = []
    stock_weights = stocks or {}
    return PortfolioSnapshot(
        total_capital=total, holdings=holdings,
        tw_weight=tw, us_weight=us, cash_weight=1-tw-us,
        sector_weights={}, stock_weights=stock_weights,
        total_exposure=tw+us,
    )


def test_no_drift():
    rb = Rebalancer()
    target = AllocationPlan(tw_pct=0.40, us_pct=0.40, cash_pct=0.20, regime_label="both_bull")
    actions = rb.check_drift(_snapshot(0.40, 0.40), target)
    assert len(actions) == 0


def test_tw_over_allocated():
    rb = Rebalancer(drift_threshold=0.05)
    target = AllocationPlan(0.40, 0.40, 0.20, "both_bull")
    actions = rb.check_drift(_snapshot(0.50, 0.38), target)
    assert len(actions) == 1
    assert actions[0].action_type == "reduce"
    assert actions[0].market == "TW"


def test_us_under_allocated():
    rb = Rebalancer(drift_threshold=0.05)
    target = AllocationPlan(0.40, 0.40, 0.20, "both_bull")
    actions = rb.check_drift(_snapshot(0.38, 0.30), target)
    us_actions = [a for a in actions if a.market == "US"]
    assert len(us_actions) == 1
    assert us_actions[0].action_type == "increase"


def test_stock_exceeds_limit():
    rb = Rebalancer()
    target = AllocationPlan(0.40, 0.40, 0.20, "both_bull")
    snap = _snapshot(0.40, 0.40, stocks={"NVDA": 0.25})
    actions = rb.check_drift(snap, target)
    stock_actions = [a for a in actions if a.symbol == "NVDA"]
    assert len(stock_actions) == 1
    assert stock_actions[0].action_type == "reduce"


def test_needs_rebalance():
    rb = Rebalancer()
    target = AllocationPlan(0.40, 0.40, 0.20, "both_bull")
    assert not rb.needs_rebalance(_snapshot(0.40, 0.40), target)
    assert rb.needs_rebalance(_snapshot(0.50, 0.40), target)
