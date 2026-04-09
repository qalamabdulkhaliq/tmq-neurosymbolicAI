"""
core/walk_grammar.py — Constrained Generation Compiler: DATA DIVISION

WalkGrammar is the DATA DIVISION of the constrained generation system.
GraphProjector maps novel roots onto the nearest Quranic roots via
Abjad clock rotation and ibn_jinni S3/Rx/Ry/Rz matrix operations.

Two independent projection mechanisms:
  1. RASM clock rotation (r=1..14): shift all 3 letters by r steps
     simultaneously on the 28-step Abjad rasm clock.
  2. ibn_jinni matrix (S3/Rx/Ry/Rz operations): loaded from
     bismillah/ibn_jinni_full_matrix.json.

Priority: clock rotation hits sorted by r-step ascending (lower r = higher
confidence), then S3/Rx/Ry/Rz hits from the matrix.

The Buckwalter transliteration is the native format throughout.
No English paraphrases enter this layer.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from bw_arabic import bw_to_arabic, bw_root_display

logger = logging.getLogger(__name__)

# ── Rasm clock — 28 Arabic consonants in Abjad order (Buckwalter) ──────────

RASM = [
    'A', 'b', 'j', 'd', 'h', 'w', 'z', 'H', 'T', 'y',
    'k', 'l', 'm', 'n', 's', 'E', 'f', 'S', 'q', 'r',
    '$', 't', 'v', 'x', '*', 'D', 'Z', 'g'
]
RASM_POS = {c: i for i, c in enumerate(RASM)}  # 0-based positions

# Abjad numerical values (for discriminant computation)
ABJAD_VAL = {
    'A': 1, 'b': 2, 'j': 3, 'd': 4, 'h': 5, 'w': 6, 'z': 7, 'H': 8, 'T': 9, 'y': 10,
    'k': 20, 'l': 30, 'm': 40, 'n': 50, 's': 60, 'E': 70, 'f': 80, 'S': 90, 'q': 100,
    'r': 200, '$': 300, 't': 400, 'v': 500, 'x': 600, '*': 700, 'D': 800, 'Z': 900, 'g': 1000,
}

# Four-clock state vector at r steps (for a 3-letter root shifted by r)
# wazn: hexagonal harakat (6-cycle, 60° per step)
# irab: grammatical case (4-cycle, divides rasm)
# waqf: cube vertices (8-cycle, 45° per step)

def _state_at_r(r: int) -> tuple:
    """(rasm_step, wazn_step, irab_step, waqf_step) at clock position r."""
    return (r % 28, (r // 2) % 6, r % 4, round(r * 28 / 8) % 8)


def rotate_root_by_steps(root: str, r: int) -> Optional[str]:
    """
    Shift all three consonants of root by r steps simultaneously on the 28-step
    rasm clock. Returns None if root contains a letter not in RASM.
    """
    if len(root) != 3:
        return None
    result = []
    for c in root:
        p = RASM_POS.get(c)
        if p is None:
            return None
        result.append(RASM[(p + r) % 28])
    return ''.join(result)


def _compute_d_class(root: str) -> str:
    """Compute discriminant class for a 3-letter root using Abjad values."""
    if len(root) != 3:
        return "WAQF"
    a_v = ABJAD_VAL.get(root[0])
    b_v = ABJAD_VAL.get(root[1])
    c_v = ABJAD_VAL.get(root[2])
    if None in (a_v, b_v, c_v):
        return "WAQF"
    D = b_v * b_v - 4 * a_v * c_v
    if D > 0:
        return "REAL"
    elif D < 0:
        return "CMPLX"
    return "ZERO"


def _discriminant_transition(src_class: str, tgt_class: str) -> str:
    if src_class == "CMPLX" and tgt_class == "REAL":
        return "grounding"
    if src_class == "REAL" and tgt_class == "CMPLX":
        return "lifting"
    if src_class == tgt_class:
        return "maintaining"
    return "boundary"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ProjectedNode:
    source_root: str        # The query root (may not be Quranic)
    target_root: str        # Nearest Quranic root found
    operation: str          # "clock_r7" or "S3_swap23" or "Rx90" etc.
    r_steps: int            # Clock steps (0 for S3/Rx ops)
    theta_deg: float        # Angle: r_steps × 360/28 (0.0 for S3/Rx ops)
    d_class_src: str        # Discriminant class of source
    d_class_tgt: str        # Discriminant class of target
    transition: str         # "grounding" | "lifting" | "maintaining" | "boundary" | "s3"
    wazn_pos: int           # Wazn clock position at theta (0-5)
    waqf_state: int         # Waqf cube vertex at theta (0-7)
    confidence: float       # Normalized confidence (0.0–1.0)


@dataclass
class WalkGrammar:
    """
    DATA DIVISION — all program-computed facts from a deliberation.

    No LLM involvement at this layer. Every field is derived from:
    - DeliberationResult (TMQ hypergraph walk output)
    - GraphProjector (Abjad clock + ibn_jinni matrix)
    - ClockOracle annotations

    slot_template and gbnf_grammar are populated by TemplateCompiler and
    GBNFCompiler respectively (separate modules, same DATA DIVISION).
    """
    question: str
    seed_roots: list            # Buckwalter roots from Bilal (verified TMQ nodes)
    visited_roots: list         # All unique roots in the walked subgraph
    projected_roots: list       # Novel roots mapped to Quranic via projection
    required_families: list     # Top 3 TMQ edge families
    modal_type: str             # "REAL" | "CMPLX" | "ZERO" | "WAQF"
    aseity_guard: bool          # True → block divine attribute slots
    eigenstate_facts: list      # Program-computed fact strings
    address_mode: int           # 1=singular 2=dual 3=plural 4=majestic
    intensity: float            # 0.0–1.0 modal intensity
    slot_template: str = ""     # Populated by TemplateCompiler
    gbnf_grammar: str = ""      # Populated by GBNFCompiler

    @property
    def visited_roots_arabic(self) -> list:
        """Visited roots as Arabic script (display layer — BW stays in visited_roots)."""
        return [bw_to_arabic(r) for r in self.visited_roots]

    @property
    def seed_roots_arabic(self) -> list:
        """Seed roots as Arabic script."""
        return [bw_to_arabic(r) for r in self.seed_roots]


# ── GraphProjector ───────────────────────────────────────────────────────────

class GraphProjector:
    """
    Projects roots onto the Quranic root lattice using two independent mechanisms:

    1. Rasm clock rotation (primary): shift all letters by r steps on the
       28-step Abjad clock. Sorted by r ascending (smaller r = closer = higher
       confidence). Max r=14 (180°).

    2. ibn_jinni matrix operations (secondary): S3 permutations and Rx/Ry/Rz
       rotations loaded from ibn_jinni_full_matrix.json. Run after clock search.

    Only Q-class results (Quran-attested) are returned.
    """

    def __init__(self) -> None:
        self._quranic_roots: set = set()
        self._matrix: dict = {}          # per_root data from full matrix
        self._summary: dict = {}         # op-level stats (for confidence scaling)
        self.loaded = False
        self._load_matrix()

    def _load_matrix(self) -> None:
        """Load ibn_jinni_full_matrix.json. Gracefully degrade if not found."""
        # Walk up from ikhtiyar/ to find bismillah/
        here = os.path.dirname(os.path.abspath(__file__))
        # here = ikhtiyar/core/ — go up twice to project root, then into bismillah
        project_dir = os.path.dirname(os.path.dirname(here))
        candidates = [
            os.path.join(project_dir, "bismillah", "ibn_jinni_full_matrix.json"),
            os.path.join(os.path.dirname(here), "..", "..", "bismillah", "ibn_jinni_full_matrix.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    self._matrix = data.get("per_root", {})
                    self._summary = data.get("summary", {})
                    # Quranic root set = all keys in per_root (the 1602 Quranic roots)
                    self._quranic_roots = set(self._matrix.keys())
                    self.loaded = True
                    logger.info(
                        f"GraphProjector: loaded ibn_jinni matrix — "
                        f"{len(self._quranic_roots)} Quranic roots, "
                        f"{len(self._summary)} operations"
                    )
                    return
                except Exception as e:
                    logger.warning(f"GraphProjector: matrix load failed ({e})")
                    return
        # Not found — clock projection still works, S3/Rx ops disabled
        # Build a minimal Quranic set from RASM combinations if matrix unavailable
        logger.warning(
            "GraphProjector: ibn_jinni_full_matrix.json not found — "
            "degrading to clock-only projection (S3/Rx ops disabled)"
        )

    def is_quranic(self, root: str) -> bool:
        """True if root is in the Quranic root set (from matrix or RASM check)."""
        if self._quranic_roots:
            return root in self._quranic_roots
        # Without matrix: assume any 3-char all-RASM root could be Quranic
        return len(root) == 3 and all(c in RASM_POS for c in root)

    def project(self, root: str, k: int = 5) -> list:
        """
        Return up to k ProjectedNodes for the given root.

        Priority order:
          1. Clock rotation hits (r=1..14), sorted by r ascending
          2. ibn_jinni matrix hits (S3/Rx/Ry/Rz), sorted by operation q_pct descending

        Returns list of ProjectedNode, sorted by confidence descending.
        """
        src_class = _compute_d_class(root)
        clock_hits = self._clock_search(root, src_class)
        matrix_hits = self._matrix_search(root, src_class) if self.loaded else []

        # Merge: clock first (higher confidence), then matrix
        combined = clock_hits + matrix_hits

        # Deduplicate by target_root (keep first / highest confidence occurrence)
        seen = set()
        unique = []
        for node in combined:
            if node.target_root not in seen:
                seen.add(node.target_root)
                unique.append(node)

        return unique[:k]

    def _clock_search(self, root: str, src_class: str) -> list:
        """Rasm clock rotation: r=1..14, both directions merged by r-distance."""
        hits = []
        for r in range(1, 15):  # 1..14 steps → 12.86°..180°
            rotated = rotate_root_by_steps(root, r)
            if rotated and self.is_quranic(rotated):
                tgt_class = _compute_d_class(rotated)
                theta = r * 360.0 / 28
                state = _state_at_r(r)
                # Confidence: max at r=1 (0.95), min at r=14 (0.05)
                confidence = 1.0 - (r - 1) / 14.0 * 0.90
                hits.append(ProjectedNode(
                    source_root=root,
                    target_root=rotated,
                    operation=f"clock_r{r}",
                    r_steps=r,
                    theta_deg=round(theta, 2),
                    d_class_src=src_class,
                    d_class_tgt=tgt_class,
                    transition=_discriminant_transition(src_class, tgt_class),
                    wazn_pos=state[1],
                    waqf_state=state[3],
                    confidence=round(confidence, 4),
                ))
        return hits

    def _matrix_search(self, root: str, src_class: str) -> list:
        """ibn_jinni matrix: S3 permutations and Rx/Ry/Rz rotations."""
        root_entry = self._matrix.get(root)
        if not root_entry:
            return []

        mappings = root_entry.get("mappings", {})
        hits = []

        for op, result_data in mappings.items():
            if op == "S3_identity":
                continue
            if isinstance(result_data, dict):
                result_root = result_data.get("result", "")
                cls = result_data.get("class", "V")
            else:
                continue

            if cls != "Q":
                continue  # Only Quran-attested results

            tgt_class = _compute_d_class(result_root)
            # Confidence from op-level q_pct in summary (normalized to 0.0–0.50)
            q_pct = self._summary.get(op, {}).get("q_pct", 0.0)
            confidence = (q_pct / 100.0) * 0.50  # matrix ops max 0.50 (below clock hits)

            hits.append(ProjectedNode(
                source_root=root,
                target_root=result_root,
                operation=op,
                r_steps=0,
                theta_deg=0.0,
                d_class_src=src_class,
                d_class_tgt=tgt_class,
                transition="s3",
                wazn_pos=0,
                waqf_state=0,
                confidence=round(confidence, 4),
            ))

        # Sort by confidence descending (higher q_pct ops first)
        hits.sort(key=lambda n: n.confidence, reverse=True)
        return hits


# ── build_walk_grammar ───────────────────────────────────────────────────────

def build_walk_grammar(
    delibresult,
    projector: GraphProjector,
    clock_annotations: list,
    override_roots: list = None,
) -> WalkGrammar:
    """
    Extract all program-computed facts from a DeliberationResult into a
    WalkGrammar DATA DIVISION struct.

    Args:
        delibresult:       DeliberationResult from deliberate().
        projector:         Loaded GraphProjector instance.
        clock_annotations: Output of ClockOracle.annotate(roots).
        override_roots:    If set, bypasses delibresult.roots (direct Buckwalter input).

    Returns:
        WalkGrammar (no LLM calls made here).
    """
    # Seed roots — from override or delibresult
    seed_roots = list(override_roots) if override_roots else list(delibresult.roots or [])

    walk_stats = delibresult.walk_stats or {}
    node_count = walk_stats.get("node_count", 0)

    # ── WAQF guard: no nodes found → silence ────────────────────────────────
    if node_count == 0:
        return WalkGrammar(
            question=delibresult.question,
            seed_roots=seed_roots,
            visited_roots=[],
            projected_roots=[],
            required_families=[],
            modal_type="WAQF",
            aseity_guard=delibresult.aseity_risk,
            eigenstate_facts=[],
            address_mode=1,
            intensity=0.0,
        )

    # ── Visited roots ────────────────────────────────────────────────────────
    # Use seed_roots as proxy for visited roots (walk_stats has node_count but
    # not individual root lists — those are in walk_result.visited_nodes which
    # may not be directly on delibresult). Augment with family-derived context.
    visited_roots = list(seed_roots)

    # ── Modal type from clock annotations ────────────────────────────────────
    # Majority D_class of seed roots from clock annotations
    modal_type = _derive_modal_type(seed_roots, clock_annotations)

    # ── Project each seed root ───────────────────────────────────────────────
    projected_roots = []
    for root in seed_roots[:6]:  # cap at 6 roots to avoid projection explosion
        nodes = projector.project(root, k=3)
        projected_roots.extend(nodes)
        # Add projection targets to visited_roots if Q-class
        for n in nodes:
            if n.target_root not in visited_roots:
                visited_roots.append(n.target_root)

    # ── Required families (top 3, no structural noise) ───────────────────────
    family_counts = walk_stats.get("family_counts", {})
    _SKIP = {"TART", "WAQF", "FASILA", "FASILA_CROSS", "JUZ", "SAJDAH"}
    top_families = [
        f for f, _ in sorted(family_counts.items(), key=lambda x: -x[1])
        if f not in _SKIP
    ][:3]

    # ── Eigenstate facts ─────────────────────────────────────────────────────
    eigenstate_facts = _build_eigenstate_facts(
        seed_roots, clock_annotations, walk_stats, projected_roots
    )

    # ── Address mode from modal summary ─────────────────────────────────────
    # Not directly on delibresult — use intensity as proxy
    address_mode = 1  # default: singular
    intensity = float(delibresult.intensity or 0.0)

    return WalkGrammar(
        question=delibresult.question,
        seed_roots=seed_roots,
        visited_roots=visited_roots[:20],     # cap for grammar enumeration
        projected_roots=projected_roots,
        required_families=top_families,
        modal_type=modal_type,
        aseity_guard=bool(delibresult.aseity_risk),
        eigenstate_facts=eigenstate_facts,
        address_mode=address_mode,
        intensity=intensity,
    )


def _derive_modal_type(seed_roots: list, clock_annotations: list) -> str:
    """
    Determine modal_type from clock annotations.
    Majority D_class of annotated seed roots wins.
    Falls back to REAL if no annotations.
    """
    if not clock_annotations:
        return "REAL"

    ann_map = {a["root"]: a["D_class"] for a in clock_annotations}
    classes = [ann_map[r] for r in seed_roots if r in ann_map]

    if not classes:
        return "REAL"

    # Majority vote
    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1

    return max(counts, key=counts.get)


def _build_eigenstate_facts(
    seed_roots: list,
    clock_annotations: list,
    walk_stats: dict,
    projected_roots: list,
) -> list:
    """
    Build program-computed fact strings — no LLM, no paraphrase.
    Each fact is a structured observation from the walk data.
    """
    facts = []
    ann_map = {a["root"]: a for a in clock_annotations}

    # Root clock coordinates
    for root in seed_roots:
        ann = ann_map.get(root)
        if ann:
            theta_str = f"  θ₀={ann['theta0']}°" if ann.get("theta0") else ""
            facts.append(
                f"Root '{root}': clock={ann['angle_deg']}°  modal={ann['D_class']}{theta_str}"
            )

    # Walk statistics
    edge_count = walk_stats.get("edge_count", 0)
    node_count = walk_stats.get("node_count", 0)
    if edge_count or node_count:
        facts.append(f"Walk: {node_count} nodes, {edge_count} edges traversed")

    # Family distribution
    fc = walk_stats.get("family_counts", {})
    if fc:
        top = sorted(fc.items(), key=lambda x: -x[1])[:3]
        fam_str = ", ".join(f"{f}={c}" for f, c in top)
        facts.append(f"Edge families: {fam_str}")

    # Projection summary
    if projected_roots:
        grounding = [n for n in projected_roots if n.transition == "grounding"]
        lifting = [n for n in projected_roots if n.transition == "lifting"]
        if grounding:
            facts.append(
                f"Projection: {grounding[0].source_root}→{grounding[0].target_root} "
                f"(grounding, {grounding[0].operation})"
            )
        if lifting:
            facts.append(
                f"Projection: {lifting[0].source_root}→{lifting[0].target_root} "
                f"(lifting, {lifting[0].operation})"
            )

    return facts
