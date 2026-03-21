import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.react import (
    _parse_args, _extract_thought, _verify,
    ReactResult, TraversalStep, ToolRegistry,
    _ACTION_RE, _ANSWER_RE,
)


# ── _parse_args ──────────────────────────────────────────────────────────────

def test_parse_args_simple():
    result = _parse_args("roots=ktb,Amn | depth=2")
    assert result["roots"] == "ktb,Amn"
    assert result["depth"] == "2"

def test_parse_args_empty():
    assert _parse_args("") == {}

def test_parse_args_single():
    assert _parse_args("root=HQQ") == {"root": "HQQ"}

def test_parse_args_no_value():
    result = _parse_args("node=seg_1_2_3 | families=NARRATIVE,MAQASID")
    assert result["node"] == "seg_1_2_3"
    assert result["families"] == "NARRATIVE,MAQASID"


# ── Regex ────────────────────────────────────────────────────────────────────

def test_action_re_matches_walk():
    text = "Thought: Let me walk.\nAction: walk_roots | roots=ktb | depth=2"
    m = _ACTION_RE.search(text)
    assert m is not None
    assert m.group(1).lower() == "walk_roots"

def test_action_re_matches_no_args():
    text = "Action: find_nodes"
    m = _ACTION_RE.search(text)
    assert m is not None
    assert m.group(1).lower() == "find_nodes"

def test_answer_re_matches():
    text = "Some preamble.\nAnswer: The root H-Q-Q constrains truth claims."
    m = _ANSWER_RE.search(text)
    assert m is not None
    assert "H-Q-Q" in m.group(1)

def test_answer_re_multiline():
    text = "Answer: First line.\nSecond line."
    m = _ANSWER_RE.search(text)
    assert m is not None
    assert "First line" in m.group(1)


# ── _extract_thought ─────────────────────────────────────────────────────────

def test_extract_thought_present():
    text = "Thought: I want to find H-Q-Q edges.\nAction: walk_roots | roots=HQQ"
    assert _extract_thought(text) == "I want to find H-Q-Q edges."

def test_extract_thought_absent():
    text = "Some response without a thought line."
    result = _extract_thought(text)
    assert len(result) > 0   # falls back to first line


# ── ReactResult defaults ─────────────────────────────────────────────────────

def test_react_result_defaults():
    r = ReactResult(question="What is truth?")
    assert r.final_answer == ""
    assert r.traversal_log == []
    assert r.roots_visited == set()
    assert r.confabulation_flags == []
    assert r.mode == "QIYAS"
    assert r.steps_taken == 0


# ── _verify (Mizan closure) ──────────────────────────────────────────────────

def test_verify_no_log_no_flags():
    r = ReactResult(question="test", final_answer="Some answer.")
    _verify(r)
    assert r.confabulation_flags == []

def test_verify_clean_no_flags():
    r = ReactResult(
        question="test",
        final_answer="The root K-T-B appears in NARRATIVE context.",
        roots_visited={"K-T-B"},
    )
    r.traversal_log.append(TraversalStep(0, "t", "walk_roots", {}, "K-T-B observed"))
    _verify(r)
    assert r.confabulation_flags == []

def test_verify_confabulation_flagged():
    r = ReactResult(
        question="test",
        final_answer="The root H-Q-Q constrains all truth, as does W-J-B.",
        roots_visited={"H-Q-Q"},   # W-J-B was NOT walked
    )
    r.traversal_log.append(
        TraversalStep(0, "t", "walk_roots", {}, "H-Q-Q: found 12 edges")
    )
    _verify(r)
    flags = [f for f in r.confabulation_flags if "W-J-B" in f]
    assert len(flags) == 1

def test_verify_observation_roots_count_as_walked():
    # Roots mentioned in observations should be counted as walked
    r = ReactResult(
        question="test",
        final_answer="The root K-T-B co-occurs with A-M-N.",
        roots_visited=set(),   # nothing explicitly added yet
    )
    r.traversal_log.append(
        TraversalStep(0, "t", "walk_roots", {}, "K-T-B and A-M-N co-occur in 47 verses")
    )
    # _verify adds observation roots before checking
    _verify(r)
    # Both roots appear in the observation, so no confabulation
    assert r.confabulation_flags == []


# ── ToolRegistry (with mock graph) ───────────────────────────────────────────

class _MockGraph:
    def walk(self, roots, depth=2, families=None):
        return {"seed_nodes": roots, "visited_nodes": [], "visited_edges": [],
                "family_counts": {}, "modal_summary": {}}

    def describe_walk(self, result):
        return f"Walk: seeds={result['seed_nodes']}"

    def neighbors(self, node_id, families=None):
        return [{"node_id": "seg_2_1_1", "attrs": {"root": "ktb", "form": "kataba"},
                 "family": "NARRATIVE", "modal": "indicative"}]

    def roots_to_nodes(self, roots):
        return [f"seg_1_{i}" for i in range(3)]

    def node(self, node_id):
        return {"root": "ktb", "form": "kataba", "pos": "V", "loc": [1, 1, 1, 1]}

    def edges_for_node(self, node_id):
        return [{"family": "NARRATIVE"}, {"family": "MAQASID"}]


def test_tool_registry_walk_roots():
    reg = ToolRegistry(_MockGraph())
    obs = reg.walk_roots(["ktb", "Amn"], depth=1)
    assert "ktb" in obs or "Walk" in obs

def test_tool_registry_get_neighbors():
    reg = ToolRegistry(_MockGraph())
    obs = reg.get_neighbors("seg_1_1_1")
    assert "NARRATIVE" in obs

def test_tool_registry_find_nodes():
    reg = ToolRegistry(_MockGraph())
    obs = reg.find_nodes("ktb")
    assert "ktb" in obs

def test_tool_registry_describe_node():
    reg = ToolRegistry(_MockGraph())
    obs = reg.describe_node("seg_1_1_1")
    assert "ktb" in obs

def test_tool_registry_unknown_tool():
    reg = ToolRegistry(_MockGraph())
    obs, _, _ = reg.dispatch("nonexistent_tool", {})
    assert "Unknown tool" in obs
