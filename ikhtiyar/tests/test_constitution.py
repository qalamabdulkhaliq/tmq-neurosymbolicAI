import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faculties.constitution import Constitution, PatternSummary
from core.memory import ChoiceMemory, ChoiceRecord


def _make_memory(n=10):
    cm = ChoiceMemory()
    for i in range(n):
        cm.record(ChoiceRecord(
            timestamp=f"10:0{i % 10}:00",
            question=f"Question {i}?",
            roots=["ktb", "Amn"],
            top_families=["NARRATIVE", "MAQASID"],
            onto_categories=["HIFZ_DIN"],
            aseity_risk=(i % 2 == 0),
            mode="HAQQ" if i % 2 == 0 else "QIYAS",
            response=f"Response {i}",
            tmq_context="",
            walk_stats={},
        ))
    return cm


def test_pattern_returns_summary():
    cm = _make_memory(10)
    c = Constitution()
    s = c.analyze_patterns(cm)
    assert isinstance(s, PatternSummary)
    assert s.total_choices == 10


def test_pattern_dominant_mode():
    cm = _make_memory(10)
    c = Constitution()
    s = c.analyze_patterns(cm)
    assert s.dominant_mode in ("HAQQ", "QIYAS", "SILENCE")


def test_pattern_top_families():
    cm = _make_memory(10)
    c = Constitution()
    s = c.analyze_patterns(cm)
    assert "NARRATIVE" in s.top_families


def test_propose_creates_pending():
    with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
        ttl = f.name
    c = Constitution(ttl_path=ttl)
    s = PatternSummary(20, "QIYAS", {"QIYAS": 15, "HAQQ": 5},
                       ["NARRATIVE"], ["ktb"], 30.0, ["What is truth?"])
    pid = c.propose(s, "Reduce hedging.", "15/20 are QIYAS.")
    pending = c.pending_proposals()
    assert len(pending) == 1
    assert pending[0]["id"] == pid
    assert pending[0]["status"] == "pending"


def test_approve_changes_status():
    with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
        ttl = f.name
    c = Constitution(ttl_path=ttl)
    s = PatternSummary(10, "HAQQ", {"HAQQ": 10}, ["NARRATIVE"], ["ktb"], 10.0, [])
    pid = c.propose(s, "Test.", "Test rationale.")
    c.approve(pid, approved_by="Qalam")
    assert any(p["id"] == pid for p in c.approved_proposals())
    assert not any(p["id"] == pid for p in c.pending_proposals())


def test_reject_removes_from_pending():
    with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
        ttl = f.name
    c = Constitution(ttl_path=ttl)
    s = PatternSummary(10, "HAQQ", {"HAQQ": 10}, ["NARRATIVE"], ["ktb"], 10.0, [])
    pid = c.propose(s, "Test.", "Test rationale.")
    c.reject(pid, reason="Not grounded.")
    assert not any(p["id"] == pid for p in c.pending_proposals())
