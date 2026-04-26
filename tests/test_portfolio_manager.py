from quantforge.portfolio.manager import PortfolioManager, AllocationPlan, PortfolioSnapshot
from quantforge.core.models import Regime


def test_both_bull_allocation():
    pm = PortfolioManager()
    alloc = pm.get_target_allocation(Regime.BULL_TREND, Regime.BULL_TREND)
    assert alloc.tw_pct == 0.40
    assert alloc.us_pct == 0.40
    assert alloc.cash_pct == 0.20


def test_crisis_allocation():
    pm = PortfolioManager()
    alloc = pm.get_target_allocation(Regime.CRISIS, Regime.BULL_TREND)
    assert alloc.cash_pct == 0.80


def test_tw_strong_allocation():
    pm = PortfolioManager()
    alloc = pm.get_target_allocation(Regime.BULL_TREND, Regime.BEAR_TREND)
    assert alloc.tw_pct == 0.50
    assert alloc.us_pct == 0.20


def test_compute_snapshot():
    pm = PortfolioManager()
    holdings = [
        {"symbol": "AAPL", "market": "US", "qty": 10, "price": 200, "sector": "Tech"},
        {"symbol": "2330", "market": "TW", "qty": 5, "price": 900, "sector": "Semis"},
    ]
    snap = pm.compute_snapshot(total_capital=50000, holdings=holdings)
    assert snap.total_capital == 50000
    assert snap.stock_weights["AAPL"] == 0.04  # 2000/50000
    assert snap.stock_weights["2330"] == 0.09  # 4500/50000
    assert snap.us_weight == 0.04
    assert snap.tw_weight == 0.09
    assert snap.total_exposure == 0.13


def test_sector_limit_warning():
    pm = PortfolioManager()
    holdings = [
        {"symbol": "AAPL", "market": "US", "qty": 100, "price": 200, "sector": "Tech"},
        {"symbol": "NVDA", "market": "US", "qty": 50, "price": 400, "sector": "Tech"},
    ]
    snap = pm.compute_snapshot(total_capital=100000, holdings=holdings)
    warnings = pm.check_sector_limits(snap)
    assert len(warnings) > 0  # Tech = 40% exceeds 35% US limit


def test_empty_portfolio():
    pm = PortfolioManager()
    snap = pm.compute_snapshot(total_capital=50000, holdings=[])
    assert snap.total_exposure == 0
    assert snap.cash_weight == 1.0
