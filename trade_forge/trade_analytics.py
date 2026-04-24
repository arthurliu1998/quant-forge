"""Trade journal analytics — analyze completed trades for behavioral patterns.

Answers questions like:
- Which signal types have the best actual performance?
- Am I cutting winners too early?
- Am I holding losers too long?
- Performance by market, time period?
"""
import logging
from dataclasses import dataclass
from trade_forge.portfolio import PortfolioDB, Trade

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsReport:
    """Summary of trading behavior analysis."""
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_rr: float                    # Actual risk/reward ratio
    best_signal: str                 # Highest win-rate signal type
    worst_signal: str                # Lowest win-rate signal type
    flags: list[str]                 # Behavioral flags


class TradeAnalytics:
    """Analyze trading history for behavioral patterns."""

    def __init__(self, db: PortfolioDB):
        self.db = db

    def analyze(self, limit: int = 200) -> AnalyticsReport:
        """Analyze recent trade history."""
        trades = self.db.get_trades(limit=limit)
        if not trades:
            return AnalyticsReport(0, 0, 0, 0, 0, "", "", [])

        # Pair buys and sells for P&L calculation
        pairs = self._pair_trades(trades)
        if not pairs:
            return AnalyticsReport(len(trades), 0, 0, 0, 0, "", "", ["Not enough sell trades for P&L analysis"])

        # Calculate metrics
        returns = [(sell.price - buy.price) / buy.price * 100 for buy, sell in pairs]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = len(wins) / len(returns) * 100 if returns else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Signal type analysis
        signal_stats = self._analyze_by_signal(pairs)
        best = max(signal_stats, key=lambda x: signal_stats[x]["win_rate"], default="")
        worst = min(signal_stats, key=lambda x: signal_stats[x]["win_rate"], default="")

        # Behavioral flags
        flags = self._detect_flags(returns, wins, losses, avg_win, avg_loss)

        return AnalyticsReport(
            total_trades=len(returns),
            win_rate=round(win_rate, 1),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            avg_rr=round(avg_rr, 2),
            best_signal=best,
            worst_signal=worst,
            flags=flags,
        )

    def _pair_trades(self, trades: list[Trade]) -> list[tuple[Trade, Trade]]:
        """Pair BUY trades with their corresponding SELL trades (FIFO)."""
        buys = {}  # symbol -> list of buy trades
        pairs = []
        for trade in sorted(trades, key=lambda t: t.timestamp or ""):
            if trade.action == "BUY":
                buys.setdefault(trade.symbol, []).append(trade)
            elif trade.action == "SELL" and trade.symbol in buys and buys[trade.symbol]:
                buy = buys[trade.symbol].pop(0)
                pairs.append((buy, trade))
        return pairs

    def _analyze_by_signal(self, pairs: list[tuple[Trade, Trade]]) -> dict:
        """Group trades by signal type and compute win rate per type."""
        stats = {}
        for buy, sell in pairs:
            sig = buy.signal_type or "unknown"
            if sig not in stats:
                stats[sig] = {"wins": 0, "total": 0, "win_rate": 0}
            stats[sig]["total"] += 1
            if sell.price > buy.price:
                stats[sig]["wins"] += 1
            stats[sig]["win_rate"] = stats[sig]["wins"] / stats[sig]["total"] * 100
        return stats

    def _detect_flags(self, returns, wins, losses, avg_win, avg_loss) -> list[str]:
        """Detect behavioral patterns that might need attention."""
        flags = []
        if len(returns) < 10:
            flags.append("Too few trades for reliable analysis")
            return flags

        # Check if cutting winners too early
        if avg_win > 0 and avg_win < abs(avg_loss) * 0.8:
            flags.append(f"Cutting winners early: avg win {avg_win:.1f}% vs avg loss {avg_loss:.1f}%")

        # Check if holding losers too long
        if avg_loss < 0 and abs(avg_loss) > avg_win * 1.5:
            flags.append(f"Holding losers too long: avg loss {avg_loss:.1f}% vs avg win {avg_win:.1f}%")

        # Check for losing streaks
        streak = 0
        max_streak = 0
        for r in returns:
            if r <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        if max_streak >= 5:
            flags.append(f"Max losing streak: {max_streak} consecutive losses")

        # Win rate warning
        win_rate = len(wins) / len(returns) * 100
        if win_rate < 40:
            flags.append(f"Low win rate: {win_rate:.0f}% — review signal quality")

        return flags
