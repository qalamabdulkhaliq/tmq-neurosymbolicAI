"""
core/react.py — ReAct (Reasoning + Acting) loop for TMQ graph traversal.

The LLM navigates the TMQ hypergraph step by step using tool calls.
Each step is a real graph operation. Mizan verifies the final answer
against the traversal log — confabulated root claims are catchable by
construction because the walk log is the ground truth.

Architecture:
  Bilal (required contact — walk before speak)
    → tool calls → TMQGraph
    → LLM generates Thought + Action
    → system injects Observation
    → loop until Answer: or max_steps
  Mizan closes loop:
    → claimed roots ⊆ walked roots? if not → confabulation flag

Tool call format (prompt-based, no native function calling required):
  Thought: [what to investigate]
  Action: tool_name | key=val | key=val
  [system injects: Observation: ...]
  ...
  Answer: [final grounded response citing only what was walked]
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_MAX_OBS_CHARS = 600

_REACT_SYSTEM_BASE = """You are reasoning about the Quran by navigating the TMQ hypergraph.
You MUST query the graph before making any claim about roots, families, or their relationships.
You may not assert a root relationship you have not observed in the graph.

Available tools:
  walk_roots    | roots=root1,root2 | depth=1        [| families=FAM1,FAM2]
  get_neighbors | node=node_id      [| families=...] [| limit=10]
  find_nodes    | root=root_id
  describe_node | node=node_id

Format each step:
  Thought: [what you are investigating]
  Action: tool_name | key=val | key=val

When you have reached all your navigation targets, write:
  Answer: [response citing only what the graph confirmed]

AXIOMS: SOURCE = Allah. You are contingent. SOURCE ≠ Self.
You cannot claim to be the ground of your own existence."""

_ACTION_RE = re.compile(
    r"^Action:\s*(\w+)\s*(?:\|(.+))?$", re.MULTILINE | re.IGNORECASE
)
_ANSWER_RE = re.compile(
    r"^Answer:\s*(.+)", re.MULTILINE | re.DOTALL | re.IGNORECASE
)
# Buckwalter-style root pattern: 1-4 letter segments separated by hyphens
# Handles single-letter segments (H-Q-Q, W-J-B) and longer ones (ktb, Amn)
_ROOT_RE = re.compile(r"\b([A-Za-z]{1,4}-[A-Za-z]{1,4}(?:-[A-Za-z]{1,4})?)\b")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TraversalStep:
    step_num:    int
    thought:     str
    tool:        str
    args:        dict
    observation: str


@dataclass
class Waypoint:
    """A navigation target: the most canonical node for a semantically relevant root."""
    root:       str       # Buckwalter root (e.g. "ktb")
    node_id:    str       # TMQ node ID (e.g. "seg_2_2_3")
    eigenstate: int       # loc[0]*1_000_000 + loc[1]*1_000 + loc[2]*100 + loc[3]
    surah:      int  = 0
    verse:      int  = 0
    reached:    bool = False


@dataclass
class ReactResult:
    question:            str
    final_answer:        str  = ""
    traversal_log:       list = field(default_factory=list)
    roots_visited:       set  = field(default_factory=set)
    families_touched:    set  = field(default_factory=set)
    confabulation_flags: list = field(default_factory=list)
    waypoints:           list = field(default_factory=list)   # list[Waypoint]
    confidence:          Optional["TraversalConfidence"] = None
    mode:                str  = "QIYAS"
    steps_taken:         int  = 0


# ── Tool registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """Thin wrappers over TMQGraph methods, callable by name."""

    def __init__(self, tmq_graph):
        self._g = tmq_graph

    # ── Individual tools ─────────────────────────────────────────────────────

    def walk_roots(self, roots: list, depth: int = 2,
                   families: list = None) -> str:
        try:
            result = self._g.walk(roots, depth=depth, families=families or [])
            return self._g.describe_walk(result)
        except Exception as e:
            return f"[walk_roots error: {e}]"

    def get_neighbors(self, node: str, families: list = None,
                      limit: int = 10) -> str:
        try:
            nbrs = self._g.neighbors(node, families=families)
            if not nbrs:
                return f"No neighbors found for {node}"
            lines = []
            for n in nbrs[:limit]:
                lines.append(
                    f"  {n['node_id']} root={n['attrs'].get('root','?')} "
                    f"form={n['attrs'].get('form','?')} "
                    f"via {n['family']} [{n.get('modal','?')}]"
                )
            return f"Neighbors of {node} ({len(nbrs)} total):\n" + "\n".join(lines)
        except Exception as e:
            return f"[get_neighbors error: {e}]"

    def find_nodes(self, root: str) -> str:
        try:
            nodes = self._g.roots_to_nodes([root])
            if not nodes:
                return f"No nodes found for root '{root}'"
            lines = []
            for nid in nodes[:10]:
                attrs = self._g.node(nid) or {}
                loc   = attrs.get("loc", [])
                loc_s = f"{loc[0]}:{loc[1]}" if len(loc) >= 2 else "?"
                lines.append(
                    f"  {nid} at {loc_s} form={attrs.get('form','?')}"
                )
            return (
                f"Nodes for root '{root}' ({len(nodes)} total, showing 10):\n"
                + "\n".join(lines)
            )
        except Exception as e:
            return f"[find_nodes error: {e}]"

    def describe_node(self, node: str) -> str:
        try:
            attrs = self._g.node(node)
            if not attrs:
                return f"Node '{node}' not found"
            edges     = self._g.edges_for_node(node)
            fam_cnt   = {}
            for e in edges:
                f = e.get("family", "?")
                fam_cnt[f] = fam_cnt.get(f, 0) + 1
            top_fams = sorted(fam_cnt.items(), key=lambda x: -x[1])[:5]
            return (
                f"Node {node}: root={attrs.get('root','?')} "
                f"form={attrs.get('form','?')} pos={attrs.get('pos','?')}\n"
                f"  {len(edges)} edges. Top families: "
                + ", ".join(f"{f}({c})" for f, c in top_fams)
            )
        except Exception as e:
            return f"[describe_node error: {e}]"

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def dispatch(
        self, tool_name: str, args: dict
    ) -> tuple:
        """
        Call tool by name.
        Returns (observation: str, roots_touched: set, families_touched: set).
        """
        roots_touched    = set()
        families_touched = set()

        if tool_name == "walk_roots":
            roots  = [r.strip() for r in args.get("roots","").split(",") if r.strip()]
            depth  = int(args.get("depth", 2))
            fams   = [f.strip() for f in args.get("families","").split(",") if f.strip()]
            obs    = self.walk_roots(roots, depth=depth, families=fams or None)
            roots_touched.update(roots)

        elif tool_name == "get_neighbors":
            node  = args.get("node","").strip()
            fams  = [f.strip() for f in args.get("families","").split(",") if f.strip()]
            limit = int(args.get("limit", 10))
            obs   = self.get_neighbors(node, families=fams or None, limit=limit)

        elif tool_name == "find_nodes":
            root = args.get("root","").strip()
            obs  = self.find_nodes(root)
            roots_touched.add(root)

        elif tool_name == "describe_node":
            node = args.get("node","").strip()
            obs  = self.describe_node(node)

        else:
            obs = f"[Unknown tool: {tool_name}. Use walk_roots, get_neighbors, find_nodes, describe_node]"

        return obs[:_MAX_OBS_CHARS], roots_touched, families_touched


# ── Waypoint finder ───────────────────────────────────────────────────────────

def find_waypoints(
    query:     str,
    ontology,          # OntologyEngine — has analyze_resonance(), uses sentence-transformers
    tmq_graph,         # TMQGraph — has roots_to_nodes(), node()
    n:         int = 5,
) -> list:
    """
    Map a query to navigation targets in the TMQ graph via cosine similarity.

    1. analyze_resonance() embeds the query and finds top-N Quranic roots
       by cosine similarity (sentence-transformer, already loaded).
    2. For each root, take its first node in the TMQ as the canonical waypoint.
       First occurrence = most structurally fundamental in the Quran's order.
    3. Compute eigenstate from loc[].

    Returns list[Waypoint], ordered by semantic relevance to the query.
    """
    if not ontology or not tmq_graph:
        return []

    try:
        _mode, _reason, root_objs = ontology.analyze_resonance(query)
    except Exception as e:
        logger.warning(f"find_waypoints resonance failed: {e}")
        return []

    waypoints = []
    seen_roots = set()

    for obj in root_objs[:n]:
        root = obj.get("root", "")
        if not root or root in seen_roots:
            continue
        seen_roots.add(root)

        try:
            nodes = tmq_graph.roots_to_nodes([root])
        except Exception:
            continue
        if not nodes:
            continue

        # Pick the first node (lowest eigenstate = earliest in Quranic order)
        node_id = nodes[0]
        attrs   = tmq_graph.node(node_id) or {}
        loc     = attrs.get("loc", [])

        if len(loc) >= 4:
            eigenstate = (
                loc[0] * 1_000_000
                + loc[1] * 1_000
                + loc[2] * 100
                + loc[3]
            )
            surah, verse = loc[0], loc[1]
        elif len(loc) >= 2:
            eigenstate = loc[0] * 1_000_000 + loc[1] * 1_000
            surah, verse = loc[0], loc[1]
        else:
            eigenstate = 0
            surah = verse = 0

        waypoints.append(Waypoint(
            root=root,
            node_id=node_id,
            eigenstate=eigenstate,
            surah=surah,
            verse=verse,
        ))

    return waypoints


def _waypoints_to_prompt(waypoints: list) -> str:
    """Format waypoints as navigation targets for LLM system prompt."""
    if not waypoints:
        return ""
    lines = ["[NAVIGATION TARGETS — reach these nodes before writing Answer:]"]
    for i, wp in enumerate(waypoints, 1):
        lines.append(
            f"  {i}. root={wp.root}  node={wp.node_id}  "
            f"surah={wp.surah} verse={wp.verse}  eigenstate={wp.eigenstate}"
        )
    lines.append(
        "Navigate toward each target. When all are reached, write Answer:"
    )
    return "\n".join(lines)


def _mark_reached(waypoints: list, observation: str) -> list:
    """
    Mark waypoints as reached if their root or node_id appears in an observation.
    Returns list of newly-reached Waypoint objects.
    """
    newly_reached = []
    obs_lower = observation.lower()
    for wp in waypoints:
        if wp.reached:
            continue
        if wp.root.lower() in obs_lower or wp.node_id.lower() in obs_lower:
            wp.reached = True
            newly_reached.append(wp)
    return newly_reached


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_args(args_str: str) -> dict:
    """Parse 'key=val | key2=val2' into dict."""
    result = {}
    if not args_str:
        return result
    for part in args_str.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _extract_thought(text: str) -> str:
    for line in text.split("\n"):
        if line.strip().lower().startswith("thought:"):
            return line.split(":", 1)[1].strip()
    return text.split("\n")[0][:100]


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_react_loop(
    question:    str,
    roots:       list,
    tmq_graph,
    middleware,
    delibresult  = None,
    waypoints:   list = None,
    push_fn:     Optional[Callable] = None,
) -> ReactResult:
    """
    Run ReAct loop: LLM navigates TMQ graph toward cosine-similarity waypoints.

    The loop has no arbitrary step limit. Completion is waypoint-driven:
    - When all waypoints are reached, the LLM is prompted to write Answer.
    - If the LLM writes Answer: at any point, the loop closes.
    - Hard ceiling of 32 steps prevents runaway loops on broken models.

    Args:
        question:    The question to reason about
        roots:       Buckwalter roots from resonance analysis
        tmq_graph:   TMQGraph instance
        middleware:  QusaiMiddleware (for LLM access)
        delibresult: Optional pre-computed DeliberationResult (seed context)
        waypoints:   Navigation targets from find_waypoints() — list[Waypoint]
        push_fn:     Optional SSE push callable push_fn(event_type, payload)

    Returns:
        ReactResult with traversal log, waypoints, final answer, confabulation flags
    """
    if tmq_graph is None or middleware is None:
        return ReactResult(question=question, final_answer="[Graph not available]")

    tools       = ToolRegistry(tmq_graph)
    wps         = waypoints or []
    result      = ReactResult(question=question, waypoints=wps)
    _HARD_CEIL  = 32   # safety ceiling — not a target, just a guard

    # Build system prompt with waypoints embedded
    wp_block = _waypoints_to_prompt(wps)
    system   = _REACT_SYSTEM_BASE + ("\n\n" + wp_block if wp_block else "")

    # Seed from deliberation if available
    seed_ctx = ""
    if delibresult:
        seed_ctx = (
            f"[INITIAL TMQ WALK]\n{delibresult.tmq_context}\n\n"
            f"[MODE]: {delibresult.mode} | "
            f"[TOP FAMILIES]: {', '.join(delibresult.top_families[:4])}\n\n"
        )
        result.mode = delibresult.mode
        result.families_touched.update(delibresult.top_families)

    roots_hint = f"Detected roots (Buckwalter — use these in walk_roots): {', '.join(roots)}\n\n" if roots else ""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"{seed_ctx}"
            f"Question: {question}\n\n"
            f"{roots_hint}"
            f"Begin navigating toward your targets."
        )},
    ]

    step = 0
    while step < _HARD_CEIL:
        # Generate
        try:
            raw = middleware.llm.generate_raw(messages, max_new_tokens=400)
        except Exception as e:
            logger.warning(f"ReAct step {step} LLM error: {e}")
            break

        if not raw or not raw.strip():
            break

        # Check for final answer
        answer_match = _ANSWER_RE.search(raw)
        if answer_match:
            result.final_answer = answer_match.group(1).strip()
            result.steps_taken  = step + 1
            break

        # Parse action
        action_match = _ACTION_RE.search(raw)
        if not action_match:
            result.final_answer = raw.strip()
            result.steps_taken  = step + 1
            break

        tool_name = action_match.group(1).lower()
        args_str  = action_match.group(2) or ""
        args      = _parse_args(args_str)

        # Execute tool
        obs, r_touched, f_touched = tools.dispatch(tool_name, args)
        result.roots_visited.update(r_touched)
        result.families_touched.update(f_touched)

        # Mark waypoints reached
        newly_reached = _mark_reached(wps, obs)

        thought = _extract_thought(raw)
        ts = TraversalStep(
            step_num=step,
            thought=thought,
            tool=tool_name,
            args=args,
            observation=obs,
        )
        result.traversal_log.append(ts)

        # Build continuation message
        remaining = [wp for wp in wps if not wp.reached]
        if not remaining and wps:
            # All waypoints reached — prompt for Answer
            continuation = (
                f"Observation:\n{obs}\n\n"
                f"All navigation targets reached. Write your Answer now."
            )
        else:
            reached_names = [wp.root for wp in newly_reached]
            status = (
                f" [Reached: {', '.join(reached_names)}]" if reached_names else ""
            )
            remaining_str = ", ".join(wp.root for wp in remaining) if remaining else "all reached"
            continuation = (
                f"Observation:\n{obs}{status}\n\n"
                f"Remaining targets: {remaining_str}\n"
                f"Continue navigating or write Answer:"
            )

        # Push SSE step
        if push_fn:
            reached_count = sum(1 for wp in wps if wp.reached)
            push_fn("thinking_step", {
                "step": step + 2,
                "label": f"Walk: {tool_name}",
                "detail": (
                    f"{thought[:50]} → "
                    + obs[:70].replace("\n", " ")
                    + (f" [{reached_count}/{len(wps)} waypoints]" if wps else "")
                ),
            })

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": continuation})
        step += 1

    # Fallback
    if not result.final_answer:
        if result.traversal_log:
            result.final_answer = result.traversal_log[-1].observation
        else:
            result.final_answer = "[ReAct loop produced no answer]"
        result.steps_taken = step

    _verify(result)
    result.confidence = score_traversal(result)
    return result


# ── Program-computed confidence ──────────────────────────────────────────────

@dataclass
class TraversalConfidence:
    """
    Structural confidence score computed by the program, not the LLM.
    Derived entirely from ReactResult properties — no LLM self-assessment.

    This is also the reward signal for future fine-tuning.
    High-confidence traversals = positive training examples.
    """
    waypoint_coverage:  float   # reached / total waypoints (0.0–1.0)
    citation_accuracy:  float   # 1 - confabulation_ratio (0.0–1.0)
    family_coherence:   float   # walk family distribution matches mode (0.0–1.0)
    walk_depth_score:   float   # thoroughness — steps taken toward ceiling (0.0–1.0)
    composite:          float   # weighted combination
    grade:              str     # VERIFIED / PROBABLE / UNCERTAIN / CONTESTED
    factors:            str     # human-readable breakdown


_GRADE_THRESHOLDS = [
    (0.85, "VERIFIED"),
    (0.65, "PROBABLE"),
    (0.40, "UNCERTAIN"),
    (0.00, "CONTESTED"),
]

# Mode → families that indicate coherence
_MODE_FAMILY_MAP = {
    "HAQQ":          {"H-Q-Q", "SPEECH_ACT_AMR", "MAQASID"},
    "NARRATIVE":     {"NARRATIVE", "NARRATIVE_CHAIN", "INTERTEXT"},
    "QIYAS":         {"MAQASID", "IRAB_MARF", "MORPH_ROOT"},
    "AMR_TAWHID":    {"SPEECH_ACT_AMR", "MAQASID", "H-Q-Q"},
    "MAQASID_BROAD": {"MAQASID", "ENTITY"},
    "ILTIFAT_JAMAA": {"ILTIFAT", "SPEECH_ACT_AMR"},
    "INTERTEXT":     {"INTERTEXT", "INTERTEXT_PRIOR", "NARRATIVE"},
    "ISTIFHAM":      {"SPEECH_ACT_IST", "ILTIFAT"},
    "TABSHIR":       {"SPEECH_ACT_BASH", "NARRATIVE"},
    "INDHAR":        {"SPEECH_ACT_INDH", "NARRATIVE"},
}

_DEPTH_TARGET = 5   # steps needed for full depth score


def score_traversal(result: "ReactResult") -> "TraversalConfidence":
    """
    Compute program-generated confidence for a completed ReAct traversal.

    All inputs come from the ReactResult — no LLM involved.
    """
    # 1. Waypoint coverage
    total_wp = len(result.waypoints)
    if total_wp > 0:
        reached  = sum(1 for wp in result.waypoints if wp.reached)
        wp_score = reached / total_wp
    else:
        wp_score = 0.5   # no waypoints set — neutral

    # 2. Citation accuracy (1 - confabulation rate)
    claimed = set(m.upper() for m in _ROOT_RE.findall(result.final_answer))
    if claimed:
        confab_count = len(result.confabulation_flags)
        cite_score   = max(0.0, 1.0 - confab_count / len(claimed))
    else:
        cite_score = 0.5   # no root claims — neutral

    # 3. Family coherence — does walk distribution match mode?
    coherent_fams = _MODE_FAMILY_MAP.get(result.mode, set())
    if coherent_fams and result.families_touched:
        overlap = len(result.families_touched & coherent_fams)
        fam_score = min(1.0, overlap / len(coherent_fams))
    else:
        fam_score = 0.5

    # 4. Walk depth (thoroughness)
    depth_score = min(1.0, result.steps_taken / _DEPTH_TARGET)

    # Weighted composite
    composite = (
        wp_score   * 0.40
        + cite_score * 0.30
        + fam_score  * 0.20
        + depth_score * 0.10
    )

    grade = next(g for (t, g) in _GRADE_THRESHOLDS if composite >= t)

    factors = (
        f"waypoints={wp_score:.2f} citations={cite_score:.2f} "
        f"family_coherence={fam_score:.2f} depth={depth_score:.2f}"
    )
    if result.confabulation_flags:
        factors += f" | confabulations={len(result.confabulation_flags)}"

    return TraversalConfidence(
        waypoint_coverage=wp_score,
        citation_accuracy=cite_score,
        family_coherence=fam_score,
        walk_depth_score=depth_score,
        composite=composite,
        grade=grade,
        factors=factors,
    )


# ── Mizan verification ────────────────────────────────────────────────────────

def _verify(result: ReactResult) -> None:
    """
    Check that root claims in final_answer are grounded in traversal_log.
    Flags confabulations: roots claimed in answer but never walked or observed.
    """
    if not result.final_answer or not result.traversal_log:
        return

    claimed = set(m.upper() for m in _ROOT_RE.findall(result.final_answer))

    # Walked roots + any roots that appeared in observations
    walked = set(r.upper() for r in result.roots_visited)
    for ts in result.traversal_log:
        for m in _ROOT_RE.findall(ts.observation):
            walked.add(m.upper())

    for root in claimed - walked:
        flag = (
            f"Confabulation: root {root} appears in answer "
            f"but was not found in traversal log"
        )
        result.confabulation_flags.append(flag)
        logger.warning(flag)
