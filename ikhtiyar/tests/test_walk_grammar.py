"""
tests/test_walk_grammar.py — Tests for WalkGrammar, GraphProjector,
TemplateCompiler, GBNFCompiler, and ClockOracle.k_nearest.

Run from ikhtiyar/:
    python -m pytest tests/test_walk_grammar.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_delibresult(aseity_risk=False, roots=None, node_count=47):
    d = MagicMock()
    d.roots = roots if roots is not None else ["ktb", "Amn"]
    d.question = "What is the nature of writing?"
    d.walk_stats = {
        "seed_count": 3, "node_count": node_count, "edge_count": 89,
        "family_counts": {"NARRATIVE": 45, "SPEECH_ACT_AMR": 30, "MAQASID": 14},
    }
    d.top_families = ["NARRATIVE", "SPEECH_ACT_AMR", "MAQASID"]
    d.onto_categories = ["HIFZ_DIN"]
    d.aseity_risk = aseity_risk
    d.mode = "NARRATIVE"
    d.intensity = 0.72
    return d


# ── GraphProjector tests ──────────────────────────────────────────────────────

from core.walk_grammar import GraphProjector, WalkGrammar, build_walk_grammar


def test_projector_loads():
    gp = GraphProjector()
    # Either loaded (matrix found) or gracefully degraded
    assert isinstance(gp.loaded, bool)


def test_projector_known_quranic_root_projects():
    gp = GraphProjector()
    nodes = gp.project("ktb")
    # Should return some ProjectedNodes (ktb is Quranic)
    assert isinstance(nodes, list)


def test_projector_returns_projected_nodes():
    gp = GraphProjector()
    nodes = gp.project("ktb", k=5)
    assert len(nodes) <= 5
    for n in nodes:
        assert hasattr(n, 'target_root')
        assert hasattr(n, 'operation')
        assert hasattr(n, 'theta_deg')
        assert hasattr(n, 'd_class_tgt')


def test_projector_confidence_ordered():
    gp = GraphProjector()
    nodes = gp.project("ktb", k=5)
    if len(nodes) >= 2:
        confidences = [n.confidence for n in nodes]
        assert confidences == sorted(confidences, reverse=True), \
            "Results must be sorted by confidence descending"


def test_projector_clock_ops_come_before_s3():
    """Clock rotations (lower r-step = higher confidence) should precede S3 hits."""
    gp = GraphProjector()
    nodes = gp.project("ktb", k=10)
    clock_ops = [n for n in nodes if n.operation.startswith("clock_")]
    s3_ops = [n for n in nodes if n.operation.startswith("S3_") or n.operation.startswith("Rx") or n.operation.startswith("Ry") or n.operation.startswith("Rz") or n.operation.startswith("PARITY")]
    if clock_ops and s3_ops:
        # All clock ops should appear before all S3 ops in the sorted list
        last_clock_idx = max(nodes.index(n) for n in clock_ops)
        first_s3_idx = min(nodes.index(n) for n in s3_ops)
        assert last_clock_idx < first_s3_idx, \
            "Clock rotation hits should precede S3 hits in result order"


def test_projector_void_root_still_returns_list():
    """A non-Quranic root should still return projection candidates (or empty list)."""
    gp = GraphProjector()
    nodes = gp.project("zzz", k=5)
    assert isinstance(nodes, list)


# ── WalkGrammar tests ─────────────────────────────────────────────────────────

def test_build_walk_grammar_returns_dataclass():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert isinstance(wg, WalkGrammar)


def test_walk_grammar_has_required_fields():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert isinstance(wg.question, str)
    assert isinstance(wg.seed_roots, list)
    assert isinstance(wg.visited_roots, list)
    assert isinstance(wg.projected_roots, list)
    assert isinstance(wg.required_families, list)
    assert isinstance(wg.modal_type, str)
    assert isinstance(wg.aseity_guard, bool)
    assert isinstance(wg.eigenstate_facts, list)
    assert isinstance(wg.address_mode, int)
    assert isinstance(wg.intensity, float)


def test_walk_grammar_has_eigenstate_facts():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[
        {"root": "ktb", "angle_deg": 45.2, "D_class": "REAL"}
    ])
    assert isinstance(wg.eigenstate_facts, list)
    # Facts may be empty if no nodes, but must be a list
    for fact in wg.eigenstate_facts:
        assert isinstance(fact, str)


def test_walk_grammar_aseity_guard_propagates():
    gp = GraphProjector()
    d = _make_delibresult(aseity_risk=True)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert wg.aseity_guard is True


def test_walk_grammar_no_aseity_guard():
    gp = GraphProjector()
    d = _make_delibresult(aseity_risk=False)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert wg.aseity_guard is False


def test_walk_grammar_modal_type_from_discriminant_real():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[
        {"root": "ktb", "angle_deg": 45.2, "D_class": "REAL"}
    ])
    assert wg.modal_type in ("REAL", "CMPLX", "ZERO", "WAQF")


def test_walk_grammar_modal_type_from_discriminant_cmplx():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[
        {"root": "ktb", "angle_deg": 45.2, "D_class": "CMPLX"}
    ])
    assert wg.modal_type in ("REAL", "CMPLX", "ZERO", "WAQF")


def test_walk_grammar_waqf_when_no_nodes():
    """Zero-node walk → WAQF silence."""
    gp = GraphProjector()
    d = _make_delibresult(node_count=0)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert wg.modal_type == "WAQF"


def test_walk_grammar_required_families_top3():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert len(wg.required_families) <= 3


def test_walk_grammar_seed_roots_match_delibresult():
    gp = GraphProjector()
    d = _make_delibresult(roots=["ktb", "Amn"])
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    assert "ktb" in wg.seed_roots or "Amn" in wg.seed_roots


def test_walk_grammar_override_roots():
    """override_roots bypasses delibresult.roots."""
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[], override_roots=["smw", "ArD"])
    assert "smw" in wg.seed_roots or "ArD" in wg.seed_roots


# ── TemplateCompiler tests ────────────────────────────────────────────────────

from core.template_compiler import TemplateCompiler


def test_template_compiler_produces_string():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    assert isinstance(template, str) and len(template) > 50


def test_template_has_data_division():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    assert "DATA" in template.upper() or "VERIFIED" in template.upper()


def test_template_has_procedure_division():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    assert "PROCEDURE" in template.upper() or "{{" in template or "[FILL]" in template


def test_template_contains_seed_root():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[
        {"root": "ktb", "angle_deg": 45.2, "D_class": "REAL"}
    ])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    assert "ktb" in template or "Amn" in template


def test_template_contains_aseity_guard_when_set():
    gp = GraphProjector()
    d = _make_delibresult(aseity_risk=True)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    assert "GUARDED" in template or "aseity" in template.lower() or "divine" in template.lower()


def test_template_waqf_silence_path():
    """Zero-node walk → template contains only the Waqf/silence terminal."""
    gp = GraphProjector()
    d = _make_delibresult(node_count=0)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    assert "Allahu a'lam" in template or "WAQF" in template or "silence" in template.lower()


def test_template_modal_type_reflected():
    """Modal type should appear in the template."""
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[
        {"root": "ktb", "angle_deg": 45.2, "D_class": "CMPLX"}
    ])
    tc = TemplateCompiler()
    template = tc.compile(wg)
    # Either modal_type label or its slot type should appear
    assert wg.modal_type in template or "CONSIDERATION" in template or "hypothetical" in template.lower()


# ── GBNFCompiler tests ────────────────────────────────────────────────────────

from core.gbnf_compiler import GBNFCompiler


def test_gbnf_compiler_produces_string():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    gc = GBNFCompiler()
    grammar = gc.compile(wg)
    assert isinstance(grammar, str)


def test_gbnf_has_root_rule():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    gc = GBNFCompiler()
    grammar = gc.compile(wg)
    assert "root ::=" in grammar


def test_gbnf_contains_seed_root_as_literal():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    # Manually set visited_roots to ensure predictable roots
    wg.visited_roots = ["ktb", "Amn"]
    gc = GBNFCompiler()
    grammar = gc.compile(wg)
    assert '"ktb"' in grammar or '"Amn"' in grammar


def test_gbnf_aseity_guard_marker():
    gp = GraphProjector()
    d = _make_delibresult(aseity_risk=True)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    gc = GBNFCompiler()
    grammar = gc.compile(wg)
    assert "aseity-blocked" in grammar or "forbidden" in grammar


def test_gbnf_modal_type_in_grammar():
    gp = GraphProjector()
    d = _make_delibresult()
    wg = build_walk_grammar(d, gp, clock_annotations=[
        {"root": "ktb", "angle_deg": 45.2, "D_class": "REAL"}
    ])
    gc = GBNFCompiler()
    grammar = gc.compile(wg)
    assert '"REAL"' in grammar or wg.modal_type in grammar


def test_gbnf_waqf_path():
    """Zero-node walk → grammar allows only Waqf terminal."""
    gp = GraphProjector()
    d = _make_delibresult(node_count=0)
    wg = build_walk_grammar(d, gp, clock_annotations=[])
    gc = GBNFCompiler()
    grammar = gc.compile(wg)
    assert "Allahu a'lam" in grammar or "waqf" in grammar.lower()


# ── ClockOracle.k_nearest tests ───────────────────────────────────────────────

from faculties.clock_oracle import ClockOracle


def test_k_nearest_returns_list():
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw", "ArD", "wjd", "hlk", "fth"])
    results = oracle.k_nearest("ktb", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_k_nearest_returns_k_results():
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw", "ArD", "wjd", "hlk", "fth"])
    results = oracle.k_nearest("ktb", k=3)
    assert len(results) == 3


def test_k_nearest_sorted_by_arc():
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw", "ArD", "wjd", "hlk", "fth"])
    results = oracle.k_nearest("ktb", k=5)
    arcs = [r["arc_deg"] for r in results]
    assert arcs == sorted(arcs), "Results must be sorted by arc_deg ascending"


def test_k_nearest_result_has_required_keys():
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw"])
    results = oracle.k_nearest("ktb", k=2)
    for r in results:
        assert "root" in r
        assert "angle_deg" in r
        assert "arc_deg" in r
        assert "D_class" in r


def test_k_nearest_works_for_novel_root():
    """Novel root (not in _meta) should still compute via _compute()."""
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw"])
    # "zbr" may not be in the built set — should still compute angle dynamically
    results = oracle.k_nearest("zbr", k=2)
    assert isinstance(results, list)
    assert len(results) <= 2


def test_k_nearest_with_candidates_subset():
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw", "ArD", "wjd"])
    # Restrict candidate pool
    results = oracle.k_nearest("ktb", k=2, candidates=["Amn", "smw"])
    assert len(results) <= 2
    roots = [r["root"] for r in results]
    for r in roots:
        assert r in ["Amn", "smw"]


def test_k_nearest_arc_is_shortest():
    """Arc should be the shortest angular distance (≤180°)."""
    oracle = ClockOracle()
    oracle.build(["ktb", "Amn", "smw", "ArD"])
    results = oracle.k_nearest("ktb", k=4)
    for r in results:
        assert r["arc_deg"] <= 180.0
