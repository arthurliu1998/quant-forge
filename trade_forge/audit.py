"""Append-only audit log for all trading operations."""
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

AUDIT_PATH = os.path.expanduser("~/.trade-forge/audit.log")


class AuditLog:
    """Append-only audit trail for trading operations."""

    def __init__(self, path: str = AUDIT_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, event_type: str, details: dict) -> None:
        """Record an audit event.

        Args:
            event_type: One of: key_access, order_submit, order_confirm,
                        order_execute, order_block, guard_override
            details: Event-specific details dict
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "details": details,
            "pid": os.getpid(),
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to write audit log: %s", type(e).__name__)

    def get_recent(self, n: int = 20) -> list[dict]:
        """Read the last N audit entries."""
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                lines = f.readlines()
            entries = []
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception:
            return []

    def count_by_type(self, event_type: str) -> int:
        """Count entries of a specific type."""
        entries = self.get_recent(n=10000)
        return sum(1 for e in entries if e.get("type") == event_type)
