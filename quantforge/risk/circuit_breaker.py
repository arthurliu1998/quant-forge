"""Circuit breaker system -- halts trading on extreme conditions.

Spec (Section 5.3):
- Drawdown halt: portfolio drawdown from peak > 15% -> stop 1 month
- Daily loss limit: > 3% -> no new positions for rest of day
- Strategy failure: 8 consecutive losses OR 3 months negative -> stop
- VIX filter: VIX > 30 -> reduce positions 50%, no new entries
"""
from dataclasses import dataclass
from datetime import datetime, date, timedelta


@dataclass
class BreakerStatus:
    trading_allowed: bool
    new_positions_allowed: bool
    position_scale: float  # 1.0 = normal, 0.5 = reduced
    active_breakers: list[str]


class CircuitBreaker:
    def __init__(self, config: dict = None):
        config = config or {}
        self.max_drawdown = config.get("max_drawdown", 0.15)
        self.daily_loss_limit = config.get("daily_loss_limit", 0.03)
        self.max_consecutive_losses = config.get("max_consecutive_losses", 8)
        self.max_negative_months = config.get("max_negative_months", 3)
        self.vix_crisis = config.get("vix_crisis", 30.0)
        self._halt_until: date | None = None
        self._daily_loss_triggered: date | None = None

    def check(self, portfolio_state: dict) -> BreakerStatus:
        """Check all circuit breakers.

        Args:
            portfolio_state: dict with keys:
                - drawdown_from_peak: float (e.g., 0.12 = 12% drawdown)
                - daily_pnl_pct: float (e.g., -0.025 = -2.5% today)
                - consecutive_losses: int
                - negative_months: int (consecutive months with negative return)
                - vix: float
        """
        active = []
        trading = True
        new_pos = True
        scale = 1.0

        # Check halt period
        today = date.today()
        if self._halt_until and today < self._halt_until:
            return BreakerStatus(False, False, 0.0,
                                 [f"Trading halted until {self._halt_until}"])

        # 1. Drawdown halt
        dd = portfolio_state.get("drawdown_from_peak", 0.0)
        if dd > self.max_drawdown:
            self._halt_until = today + timedelta(days=30)
            active.append(f"Drawdown {dd:.1%} > {self.max_drawdown:.0%} -- halted 30 days")
            trading = False
            new_pos = False
            scale = 0.0

        # 2. Daily loss limit
        daily = portfolio_state.get("daily_pnl_pct", 0.0)
        if daily < -self.daily_loss_limit:
            active.append(f"Daily loss {daily:.1%} > {self.daily_loss_limit:.0%} limit")
            new_pos = False
            self._daily_loss_triggered = today

        # Reset daily loss trigger on new day
        if self._daily_loss_triggered and self._daily_loss_triggered != today:
            self._daily_loss_triggered = None

        # 3. Strategy failure
        consec = portfolio_state.get("consecutive_losses", 0)
        neg_months = portfolio_state.get("negative_months", 0)
        if consec >= self.max_consecutive_losses:
            active.append(f"{consec} consecutive losses -- strategy review needed")
            new_pos = False
        if neg_months >= self.max_negative_months:
            active.append(f"{neg_months} negative months -- strategy review needed")
            new_pos = False

        # 4. VIX filter
        vix = portfolio_state.get("vix", 0.0)
        if vix > self.vix_crisis:
            active.append(f"VIX {vix:.1f} > {self.vix_crisis} -- crisis mode")
            new_pos = False
            scale = 0.5

        return BreakerStatus(trading, new_pos, scale, active)

    def reset_halt(self):
        self._halt_until = None
        self._daily_loss_triggered = None
