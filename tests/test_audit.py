import pytest
from trade_forge.audit import AuditLog


@pytest.fixture
def audit(tmp_path):
    return AuditLog(str(tmp_path / "test_audit.log"))


class TestAuditLog:
    def test_record_and_read(self, audit):
        audit.record("order_submit", {"symbol": "TSLA", "action": "BUY"})
        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["type"] == "order_submit"
        assert entries[0]["details"]["symbol"] == "TSLA"

    def test_append_only(self, audit):
        audit.record("order_submit", {"symbol": "TSLA"})
        audit.record("order_confirm", {"symbol": "TSLA"})
        audit.record("order_execute", {"symbol": "TSLA"})
        entries = audit.get_recent()
        assert len(entries) == 3
        assert [e["type"] for e in entries] == ["order_submit", "order_confirm", "order_execute"]

    def test_count_by_type(self, audit):
        audit.record("order_submit", {})
        audit.record("order_block", {})
        audit.record("order_submit", {})
        assert audit.count_by_type("order_submit") == 2
        assert audit.count_by_type("order_block") == 1

    def test_empty_log(self, audit):
        assert audit.get_recent() == []
        assert audit.count_by_type("anything") == 0

    def test_has_timestamp(self, audit):
        audit.record("test", {"data": 1})
        entries = audit.get_recent()
        assert "timestamp" in entries[0]
