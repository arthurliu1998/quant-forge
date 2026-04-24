"""Portfolio tracking with SQLite storage.

Tracks open/closed positions, trade history, and P&L metrics.
Uses standard sqlite3 (sqlcipher upgrade is future work).
"""
import logging
import os
import sqlite3
from dataclasses import dataclass, fields
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/.trade-forge/portfolio.db")


@dataclass
class Position:
    symbol: str
    market: str          # US or TW
    qty: int
    avg_cost: float
    stop_loss: float = 0.0
    target: float = 0.0
    entry_date: str = ""
    entry_reason: str = ""
    status: str = "open"  # open, closed


@dataclass
class Trade:
    symbol: str
    market: str
    action: str          # BUY or SELL
    qty: int
    price: float
    timestamp: str = ""
    reason: str = ""
    signal_type: str = ""
    scores: str = ""     # JSON string of analyst scores


class PortfolioDB:
    """SQLite-backed portfolio tracker."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                qty INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                stop_loss REAL DEFAULT 0,
                target REAL DEFAULT 0,
                entry_date TEXT,
                entry_reason TEXT,
                status TEXT DEFAULT 'open',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                action TEXT NOT NULL,
                qty INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT,
                reason TEXT,
                signal_type TEXT,
                scores TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def add_position(self, pos: Position) -> int:
        """Add a new position. Returns position ID."""
        cur = self._conn.execute(
            "INSERT INTO positions (symbol, market, qty, avg_cost, stop_loss, target, entry_date, entry_reason, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pos.symbol, pos.market, pos.qty, pos.avg_cost, pos.stop_loss,
             pos.target, pos.entry_date, pos.entry_reason, pos.status),
        )
        self._conn.commit()
        return cur.lastrowid

    def close_position(self, position_id: int):
        """Mark a position as closed."""
        self._conn.execute(
            "UPDATE positions SET status = 'closed' WHERE id = ?", (position_id,)
        )
        self._conn.commit()

    @staticmethod
    def _rows_to(cls, rows):
        keys = [f.name for f in fields(cls)]
        return [cls(**{k: row[k] for k in keys}) for row in rows]

    def get_open_positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions WHERE status = 'open'").fetchall()
        return self._rows_to(Position, rows)

    def get_all_positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return self._rows_to(Position, rows)

    def record_trade(self, trade: Trade) -> int:
        """Record a trade. Returns trade ID."""
        if not trade.timestamp:
            trade.timestamp = datetime.utcnow().isoformat()
        cur = self._conn.execute(
            "INSERT INTO trades (symbol, market, action, qty, price, timestamp, reason, signal_type, scores) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trade.symbol, trade.market, trade.action, trade.qty, trade.price,
             trade.timestamp, trade.reason, trade.signal_type, trade.scores),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_trades(self, symbol: str = None, limit: int = 50) -> list[Trade]:
        """Get trade history, optionally filtered by symbol."""
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return self._rows_to(Trade, rows)

    def get_performance(self) -> dict:
        """Calculate overall performance metrics."""
        trades = self.get_trades(limit=1000)
        if not trades:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}

        # Group by symbol to calculate P&L
        buys = [t for t in trades if t.action == "BUY"]
        sells = [t for t in trades if t.action == "SELL"]

        total = len(trades)
        wins = sum(1 for t in sells if t.price > 0)  # Simplified
        return {
            "total_trades": total,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "win_rate": round(wins / len(sells) * 100, 1) if sells else 0,
        }

    def close(self):
        self._conn.close()
