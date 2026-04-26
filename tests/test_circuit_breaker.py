from quantforge.risk.circuit_breaker import CircuitBreaker, BreakerStatus


def test_normal_conditions():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.05, "daily_pnl_pct": -0.01,
                        "consecutive_losses": 2, "negative_months": 0, "vix": 15})
    assert status.trading_allowed
    assert status.new_positions_allowed
    assert status.position_scale == 1.0
    assert len(status.active_breakers) == 0


def test_drawdown_halt():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.18, "daily_pnl_pct": 0,
                        "consecutive_losses": 0, "negative_months": 0, "vix": 15})
    assert not status.trading_allowed
    assert not status.new_positions_allowed
    assert status.position_scale == 0.0


def test_daily_loss_blocks_new_positions():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.05, "daily_pnl_pct": -0.04,
                        "consecutive_losses": 0, "negative_months": 0, "vix": 15})
    assert status.trading_allowed  # can still manage existing
    assert not status.new_positions_allowed


def test_consecutive_losses():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.05, "daily_pnl_pct": 0,
                        "consecutive_losses": 9, "negative_months": 0, "vix": 15})
    assert not status.new_positions_allowed
    assert any("consecutive" in b for b in status.active_breakers)


def test_negative_months():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.05, "daily_pnl_pct": 0,
                        "consecutive_losses": 0, "negative_months": 3, "vix": 15})
    assert not status.new_positions_allowed


def test_vix_crisis():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.05, "daily_pnl_pct": 0,
                        "consecutive_losses": 0, "negative_months": 0, "vix": 35})
    assert status.trading_allowed
    assert not status.new_positions_allowed
    assert status.position_scale == 0.5


def test_multiple_breakers():
    cb = CircuitBreaker()
    status = cb.check({"drawdown_from_peak": 0.05, "daily_pnl_pct": -0.04,
                        "consecutive_losses": 10, "negative_months": 0, "vix": 32})
    assert not status.new_positions_allowed
    assert len(status.active_breakers) == 3  # daily + consecutive + vix
