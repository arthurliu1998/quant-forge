import os
import pytest
from trade_forge.trajectory import (
    init_trajectory, load_trajectory, record_decision, get_summary, get_symbol_history,
)


@pytest.fixture
def traj_path(tmp_path):
    path = str(tmp_path / "session" / "trajectory.json")
    init_trajectory(path)
    return path


class TestTrajectory:
    def test_init_creates_file(self, traj_path):
        assert os.path.exists(traj_path)
        data = load_trajectory(traj_path)
        assert data["stats"]["total"] == 0
        assert data["decisions"] == []

    def test_record_decision(self, traj_path):
        record_decision(
            traj_path, symbol="TSLA", action="BUY @ $248",
            reasoning="Tech bullish + flow", scores={"tech": 7.5, "flow": 7},
            status="executed", confidence=72.0,
        )
        data = load_trajectory(traj_path)
        assert data["stats"]["total"] == 1
        assert data["decisions"][0]["symbol"] == "TSLA"
        assert data["decisions"][0]["confidence"] == 72.0

    def test_multiple_decisions(self, traj_path):
        record_decision(traj_path, "TSLA", "BUY", "reason1", {"tech": 7}, "executed", 70)
        record_decision(traj_path, "AAPL", "SELL", "reason2", {"tech": 4}, "cancelled", 30)
        summary = get_summary(traj_path)
        assert summary["total_decisions"] == 2
        assert summary["executed"] == 1
        assert summary["cancelled"] == 1

    def test_get_symbol_history(self, traj_path):
        record_decision(traj_path, "TSLA", "BUY", "r1", {}, "executed", 70)
        record_decision(traj_path, "AAPL", "BUY", "r2", {}, "executed", 60)
        record_decision(traj_path, "TSLA", "SELL", "r3", {}, "executed", 50)
        history = get_symbol_history(traj_path, "TSLA")
        assert len(history) == 2
        assert history[0]["action"] == "BUY"
        assert history[1]["action"] == "SELL"

    def test_decision_has_timestamp(self, traj_path):
        record_decision(traj_path, "TSLA", "BUY", "test", {}, "executed", 50)
        data = load_trajectory(traj_path)
        assert "timestamp" in data["decisions"][0]
        assert len(data["decisions"][0]["timestamp"]) > 0
