"""Trade decision trajectory tracker.

Records each trading decision with scores, reasoning, and outcome.
Adapted from kernel-forge's trajectory.py for trading context.
"""
import json
import os
from datetime import datetime, timezone


def init_trajectory(path: str) -> None:
    """Create a new empty trajectory file."""
    data = {
        "session_id": os.path.basename(os.path.dirname(path)) if os.path.dirname(path) else "default",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decisions": [],
        "stats": {"total": 0, "executed": 0, "cancelled": 0},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_trajectory(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_trajectory(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def record_decision(
    path: str,
    symbol: str,
    action: str,
    reasoning: str,
    scores: dict,
    status: str = "executed",
    confidence: float = 0.0,
) -> None:
    """Record a trading decision.

    Args:
        path: Path to trajectory JSON file
        symbol: Stock symbol
        action: e.g., "BUY TSLA @ $248"
        reasoning: Why this decision was made
        scores: Dict of analyst scores {"tech": 7.5, "flow": 7, "market": 8}
        status: executed, cancelled, partial
        confidence: 0-100 confidence score
    """
    data = load_trajectory(path)
    entry = {
        "id": len(data["decisions"]) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "action": action,
        "reasoning": reasoning,
        "scores": scores,
        "status": status,
        "confidence": confidence,
    }
    data["decisions"].append(entry)
    data["stats"]["total"] += 1
    data["stats"][status] = data["stats"].get(status, 0) + 1
    save_trajectory(path, data)


def get_summary(path: str) -> dict:
    """Get trajectory summary."""
    data = load_trajectory(path)
    return {
        "session_id": data["session_id"],
        "total_decisions": data["stats"]["total"],
        "executed": data["stats"].get("executed", 0),
        "cancelled": data["stats"].get("cancelled", 0),
        "decisions": data["decisions"],
    }


def get_symbol_history(path: str, symbol: str) -> list[dict]:
    """Get all decisions for a specific symbol."""
    data = load_trajectory(path)
    return [d for d in data["decisions"] if d["symbol"] == symbol]
