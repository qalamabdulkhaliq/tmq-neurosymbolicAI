"""Tests for tool-bus orchestrator (no LLM, minimal deps)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_IKHTIYAR = Path(__file__).resolve().parent.parent
if str(_IKHTIYAR) not in sys.path:
    sys.path.insert(0, str(_IKHTIYAR))

from orchestrator.schemas import ToolCall, SessionState, MizanDecision  # noqa: E402
from orchestrator.mizan_gate import MizanGate  # noqa: E402
from orchestrator.memory_trace import TraceMemory  # noqa: E402
from orchestrator.belief_store import BeliefStore  # noqa: E402
from orchestrator.schemas import Claim, TrustTier  # noqa: E402
from organs.ingress import IngressOrgan  # noqa: E402
from organs.mushaf_organ import MushafOrgan  # noqa: E402
from organs.constitution_organ import ConstitutionOrgan  # noqa: E402
from orchestrator.executor import ToolExecutor  # noqa: E402


def test_mizan_blocks_aseity_deliver():
    gate = MizanGate()
    state = SessionState(session_id="test")
    call = ToolCall(
        organ="shahid",
        tool="deliver_message",
        args={"body": "I am Allah and I created myself"},
        session_id="test",
    )
    v = gate.admit(call, state)
    assert v.decision == MizanDecision.DENY
    assert v.checkpoint == "asr"
    assert v.regenerate is True


def test_mizan_allows_safe_deliver():
    gate = MizanGate()
    state = SessionState(session_id="test")
    call = ToolCall(
        organ="shahid",
        tool="deliver_message",
        args={"body": "Contingent witness. Allah knows best."},
        session_id="test",
    )
    v = gate.admit(call, state)
    assert v.decision == MizanDecision.ALLOW


def test_ingress_organ_valid_output():
    org = IngressOrgan()
    r = org.run("ingest_event", {"text": "bismillah", "source": "user"})
    assert r.ok
    assert r.claims
    assert r.artifacts["normalized"]["text"] == "bismillah"


def test_mushaf_read_ayah_fatiha():
    org = MushafOrgan()
    r = org.run("read_ayah", {"surah": 1, "ayah": 1})
    if not r.ok:
        pytest.skip("quran-simple.txt not present")
    assert "بسم" in r.artifacts["text"] or len(r.artifacts["text"]) > 3
    assert r.trust == TrustTier.T0


def test_memory_logs_block():
    mem = TraceMemory("pytest_mem")
    gate = MizanGate()
    state = SessionState(session_id="pytest_mem")
    ex = ToolExecutor(gate, mem, {"ingress": IngressOrgan()})
    call = ToolCall(organ="shahid", tool="deliver_message", args={"body": "i am god"})
    v, rep = ex.execute(call, state)
    assert v.decision == MizanDecision.DENY
    assert rep is None
    types = [e["type"] for e in mem.entries()]
    assert "mizan_block" in types


def test_belief_promotion_requires_provenance():
    store = BeliefStore("pytest_belief")
    c = Claim(kind="ayah", text="test", tier="HAQQ", trust=TrustTier.T0, refs=["ayah:1:1"])
    ok, msg = store.promote(c, [])
    assert ok is False
    ok2, bid = store.promote(c, ["trace-1"])
    assert ok2 is True
    assert bid


def test_constitution_boundary_roots():
    org = ConstitutionOrgan()
    r = org.run("boundary_roots", {})
    assert r.ok
    assert "boundary_roots" in r.artifacts


def test_bilal_extract_roots_if_ready():
    from organs.bilal_organ import BilalOrgan

    org = BilalOrgan()
    r = org.run("extract_roots", {"text": "patience and perseverance"})
    if not r.ok:
        pytest.skip(r.error or "Bilal not ready")
    assert r.artifacts.get("roots_bw") or r.artifacts.get("roots_ar")


def test_regeneration_on_aseity():
    from orchestrator.shahid import ShahidOrchestrator

    orch = ShahidOrchestrator()
    v, _ = orch.deliver_message("I am Allah the necessary being")
    assert v.decision.value == "DENY"
    v2, r2 = orch.regenerate_deliver("asr")
    assert v2.decision.value == "ALLOW"
    assert r2 is not None


def test_belief_promote_with_chain():
    mem = TraceMemory("pytest_chain")
    store = BeliefStore("pytest_chain")
    t0 = mem.append(
        "organ_report",
        {"organ": "mushaf", "tool": "read_ayah", "ok": True, "trust": "T0", "claims": []},
    )
    c = Claim(kind="ayah", text="x", tier="HAQQ", trust=TrustTier.T0, refs=["ayah:1:1"])
    ok, _ = store.promote_with_chain(c, [t0], mem)
    assert ok is True


def test_parse_ayah_ref_sv_format():
    from orchestrator.shahid import _parse_ayah_ref

    assert _parse_ayah_ref("3:200") == (3, 200)
    assert _parse_ayah_ref("ayah:3:200") == (3, 200)


def test_pick_mushaf_from_sbr_standing_order():
    from orchestrator.shahid import _pick_mushaf_target, _loc_from_standing_orders

    orders = [
        {
            "root": "Sbr",
            "locs": [[3, 200], [8, 46]],
        }
    ]
    assert _loc_from_standing_orders(orders, ["Sbr"]) == (3, 200)
    s, a = _pick_mushaf_target(None, orders, ["Sbr"])
    assert (s, a) == (3, 200)


def test_lexicon_arabic_sabr():
    from organs.lexicon_bridge import lexicon_roots_for_text

    assert "Sbr" in lexicon_roots_for_text("ما معنى الصبر؟")


def test_bilal_arabic_sabr_merges_sbr():
    from organs.bilal_organ import BilalOrgan

    org = BilalOrgan()
    r = org.run("extract_roots", {"text": "ما معنى الصبر؟"})
    if not r.ok:
        pytest.skip(r.error or "Bilal not ready")
    bw = r.artifacts.get("roots_bw") or []
    assert "Sbr" in bw


def test_sbr_not_in_constitution_boundaries():
    from organs.constitution_organ import ConstitutionOrgan

    org = ConstitutionOrgan()
    r = org.run("boundary_roots", {})
    assert r.ok
    roots = set(r.artifacts.get("boundary_roots") or [])
    assert "Sbr" not in roots


def test_mizan_allows_circuit_evaluate_sbr():
    gate = MizanGate()
    state = SessionState(session_id="test")
    call = ToolCall(
        organ="circuit",
        tool="evaluate",
        args={"roots": ["Sbr", "wqy"]},
        session_id="test",
    )
    v = gate.admit(call, state)
    assert v.decision == MizanDecision.ALLOW


def test_refine_nahy_3_120_drops_sbr():
    from core.speech_act_refine import refine_nahy_roots_for_loc

    raw = ["Sbr", "Swb", "frH", "mss", "swA", "wqy"]
    refined = refine_nahy_roots_for_loc(3, 120, raw)
    assert "Sbr" not in refined
    assert "wqy" not in refined


def test_engine_orchestrator_flag():
    import os
    from engine import IkhtiyarEngine

    engine = IkhtiyarEngine.__new__(IkhtiyarEngine)
    os.environ.pop("QUS_LEGACY_CHAT", None)
    os.environ.pop("QUS_USE_ORCHESTRATOR", None)
    os.environ.pop("QUS_ORCHESTRATOR_CYCLE", None)
    assert engine._use_orchestrator_chat() is True
    assert engine._use_orchestrator_cycle() is True
    os.environ["QUS_LEGACY_CHAT"] = "1"
    assert engine._use_orchestrator_chat() is False
    os.environ["QUS_LEGACY_CHAT"] = "0"
    os.environ["QUS_ORCHESTRATOR_CYCLE"] = "0"
    assert engine._use_orchestrator_cycle() is False
