"""
core/graph_reasoner.py — Deterministic graph reasoning engine.

"I offload most of my reasoning and all of my logic by anchoring me to a
semantic web — the directed acyclic graph of The Mother Quran."

GraphReasoner takes roots + question → returns GraphConclusion from pure
graph operations. No LLM. The LLM only translates the result.

Architecture:
  roots → walk_and_collect()   — TMQ BFS walk
        → find_verse_evidence() — locate verses containing walked roots
        → match_amr_rulings()   — O(1) lookup in active_command_set
        → _classify_mode()      — deterministic mode from evidence shape
        → derive_statement()    — template-based, no generation
        → _compute_confidence() — evidence density score
        → GraphConclusion       — ready for translation-only LLM path
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── GraphConclusion ──────────────────────────────────────────────────────────

@dataclass
class GraphConclusion:
    roots_walked:       list           # every root BFS-expanded
    nodes_visited:      list           # node IDs visited
    edges_traversed:    list           # [{edge_id, family, nodes, ...}]
    verse_evidence:     list           # [{ref:"2:30", text:"...", roots_present:[...]}]
    amr_rulings:        list           # [{type:"AMR"|"NAHY", root:..., text:...}]
    families_touched:   set            # TMQ family names encountered
    maqasid_categories: list           # HIFZ_DIN, HIFZ_AQL, ...
    derived_statement:  str            # deterministic conclusion
    confidence:         float          # evidence density (0.0–1.0)
    mode:               str = "QIYAS" # HAQQ / AMR_TAWHID / NAHY / NARRATIVE / QIYAS / ...


# ── Template families ────────────────────────────────────────────────────────
# Deterministic, no LLM. Selection: mode + evidence shape → family.
# Rotation: abs(hash(tuple(sorted(roots)))) % len(family) avoids monotone output.

_TEMPLATES: dict[str, list[str]] = {
    # Every template in every family carries identical propositional payload:
    # root + meaning + n_verses + verse_refs + ruling_type + ruling_sample + maqasid_categories
    # Variation is surface form only — sentence order and labeling, not information density.

    "AMR_SINGLE": [
        # [0] declarative form
        "Root {root} ({meaning}) appears in {n_verses} verses ({verse_refs}). "
        "Obligation established: {ruling_sample}. Maqasid: {maqasid_categories}.",

        # [1] evidence-first form
        "Quranic evidence ({verse_refs}, {n_verses} occurrences) anchors root {root} ({meaning}). "
        "Standing order: OBLIGATION — {ruling_sample}. Categories: {maqasid_categories}.",

        # [2] ruling-first form
        "OBLIGATION on root {root} ({meaning}): {ruling_sample}. "
        "Grounded in {n_verses} verses ({verse_refs}). Maqasid: {maqasid_categories}.",
    ],

    "NAHY_SINGLE": [
        # [0] declarative form
        "Root {root} ({meaning}) appears in {n_verses} verses ({verse_refs}). "
        "Prohibition established: {ruling_sample}. Maqasid: {maqasid_categories}.",

        # [1] evidence-first form
        "Quranic evidence ({verse_refs}, {n_verses} occurrences) anchors root {root} ({meaning}). "
        "Standing order: PROHIBITION — {ruling_sample}. Categories: {maqasid_categories}.",

        # [2] ruling-first form
        "PROHIBITION on root {root} ({meaning}): {ruling_sample}. "
        "Grounded in {n_verses} verses ({verse_refs}). Maqasid: {maqasid_categories}.",
    ],

    "MULTI_ROOT": [
        # [0] declarative form
        "Roots {root} ({meaning}) appear in {n_verses} verses ({verse_refs}). "
        "Primary ruling: {ruling_type} — {ruling_sample}. Maqasid: {maqasid_categories}.",

        # [1] evidence-first form
        "Quranic evidence ({verse_refs}, {n_verses} occurrences) anchors roots {root} ({meaning}). "
        "Standing order: {ruling_type} — {ruling_sample}. Categories: {maqasid_categories}.",

        # [2] ruling-first form
        "{ruling_type} on roots {root} ({meaning}): {ruling_sample}. "
        "Grounded in {n_verses} verses ({verse_refs}). Maqasid: {maqasid_categories}.",
    ],

    # No ruling by definition. ruling_type/ruling_sample absent — all other fields present.
    "CLASSIFICATION": [
        # [0] declarative form
        "Root {root} ({meaning}) appears in {n_verses} verses ({verse_refs}). "
        "No direct ruling found. Maqasid: {maqasid_categories}.",

        # [1] evidence-first form
        "Quranic evidence ({verse_refs}, {n_verses} occurrences) anchors root {root} ({meaning}). "
        "No standing order in corpus. Categories: {maqasid_categories}.",

        # [2] graph-structure form
        "Root {root} ({meaning}): no AMR/NAHY ruling in active command set. "
        "Evidence: {n_verses} verses ({verse_refs}). Maqasid: {maqasid_categories}.",
    ],
}


def _pick_template(family: str, roots: list[str]) -> str:
    """Deterministic template selection — rotation by root hash avoids monotone output."""
    templates = _TEMPLATES.get(family, _TEMPLATES["CLASSIFICATION"])
    idx = abs(hash(tuple(sorted(roots)))) % len(templates)
    return templates[idx]


# ── Mode classification ──────────────────────────────────────────────────────

_SKIP_FAMILIES = {
    "TART", "WAQF", "FASILA", "FASILA_CROSS", "JUZ", "SAJDAH",
    "Syn_part", "MORPH_POS", "MORPH_LEM", "RUKU",
}


def _classify_mode(family_counts: dict, onto_cats: list, amr_rulings: list) -> str:
    """Derive semantic mode from graph evidence — same logic as deliberate._classify_mode."""
    fc = set(family_counts)
    onto = set(onto_cats)
    ruling_types = {r["type"] for r in amr_rulings}

    if "AMR" in ruling_types and ("HIFZ_DIN" in onto or "TAWHID" in onto):
        return "AMR_TAWHID"
    if "NAHY" in ruling_types:
        return "NAHY"
    if "AMR" in ruling_types:
        return "AMR"
    if "NARRATIVE" in fc or "NARRATIVE_CHAIN" in fc:
        return "NARRATIVE"
    if "INTERTEXT" in fc:
        return "INTERTEXT"
    if "MAQASID" in fc and len(onto) >= 3:
        return "MAQASID_BROAD"
    if "SPEECH_ACT_ISTIFHAM" in fc:
        return "ISTIFHAM"
    return "QIYAS"


# ── GraphReasoner ────────────────────────────────────────────────────────────

class GraphReasoner:
    """
    Deterministic reasoning engine.
    Takes roots + question → GraphConclusion via pure graph ops.
    No LLM in this class.
    """

    def __init__(self, tmq_graph, mushaf=None, command_index: dict = None):
        """
        Args:
            tmq_graph:     TMQGraph instance (required)
            mushaf:        MushafReader — optional, used for verse text retrieval
            command_index: Pre-loaded AMR index dict {amr: {...}, nahy: {...}, maqasid: {...}}
                           Pass _COMMAND_INDEX from deliberate.py to avoid double-loading.
        """
        self._tmq    = tmq_graph
        self._mushaf = mushaf
        self._cmds   = command_index or {}

    # ── walk_and_collect ──────────────────────────────────────────────────────

    def walk_and_collect(self, roots: list[str]) -> dict:
        """
        BFS walk from roots. Returns raw walk dict from TMQGraph.walk().
        Adds flattened edge list for GraphConclusion.
        """
        walk = self._tmq.walk(roots, depth=2, families=None)
        edges_flat = []
        for eid, edge in (walk.get("visited_edges") or {}).items():
            edges_flat.append({
                "edge_id": eid,
                "family":  edge.get("family", ""),
                "nodes":   edge.get("nodes", []),
                "modal":   edge.get("modal", {}),
            })
        walk["_edges_flat"] = edges_flat
        return walk

    # ── find_verse_evidence ───────────────────────────────────────────────────

    def find_verse_evidence(self, walk: dict, roots: list[str]) -> list[dict]:
        """
        Collect verse references from walked nodes.
        Groups by (surah, ayah), fetches text via mushaf if available.
        Caps at 5 unique verses to keep GraphConclusion compact.

        Critical: filters to nodes whose root matches the query roots FIRST.
        Without this, depth-2 BFS through INTERTEXT edges reaches most of the
        graph, and sorting by (surah, ayah) always returns Al-Fatiha 1:1-1:5
        regardless of the actual query root.
        """
        visited_nodes = walk.get("visited_nodes") or {}
        roots_set     = set(roots)
        verse_map: dict[tuple, list[str]] = {}

        # Pass 1: collect only nodes whose root matches the query roots
        for nid, attrs in visited_nodes.items():
            if attrs.get("root") not in roots_set:
                continue
            loc = attrs.get("loc")
            if loc and len(loc) >= 2:
                key = (int(loc[0]), int(loc[1]))
                r   = attrs["root"]
                verse_map.setdefault(key, [])
                if r not in verse_map[key]:
                    verse_map[key].append(r)

        # Pass 2: if no query-root nodes had location data, fall back to all visited
        if not verse_map:
            for nid, attrs in visited_nodes.items():
                loc  = attrs.get("loc")
                root = attrs.get("root", "")
                if loc and len(loc) >= 2:
                    key = (int(loc[0]), int(loc[1]))
                    verse_map.setdefault(key, [])
                    if root and root not in verse_map[key]:
                        verse_map[key].append(root)

        # Sort by surah:ayah, cap at 5
        sorted_verses = sorted(verse_map.items(), key=lambda x: x[0])[:5]

        evidence = []
        for (surah, ayah), present_roots in sorted_verses:
            ref  = f"{surah}:{ayah}"
            text = ""
            if self._mushaf:
                try:
                    text = self._mushaf.get_ayah(surah, ayah) or ""
                except Exception:
                    pass
            evidence.append({
                "ref":           ref,
                "text":          text[:300] if text else "",
                "roots_present": present_roots[:5],
            })

        return evidence

    # ── match_amr_rulings ─────────────────────────────────────────────────────

    def match_amr_rulings(self, roots: list[str]) -> list[dict]:
        """
        Lookup AMR/NAHY rulings for each root from the merged command index.
        Returns list of {type, root, ref, text, modal}, capped at 5.
        ref and modal present when sourced from full_quran_constitution.
        """
        if not self._cmds:
            return []

        amr_idx  = self._cmds.get("amr",  {})
        nahy_idx = self._cmds.get("nahy", {})
        hits = []

        def _extract(entry, ruling_type, root):
            if isinstance(entry, dict):
                return {
                    "type":  ruling_type,
                    "root":  root,
                    "ref":   entry.get("ref", ""),
                    "text":  entry.get("text", ""),
                    "modal": entry.get("modal", {}),
                }
            # legacy string format (active_command_set supplement)
            return {"type": ruling_type, "root": root, "ref": "", "text": str(entry), "modal": {}}

        for root in roots:
            amr_entries = amr_idx.get(root, [])
            if amr_entries:
                hits.append(_extract(amr_entries[0], "AMR", root))
            nahy_entries = nahy_idx.get(root, [])
            if nahy_entries:
                hits.append(_extract(nahy_entries[0], "NAHY", root))
            if len(hits) >= 5:
                break

        return hits

    # ── derive_statement ──────────────────────────────────────────────────────

    def derive_statement(
        self,
        roots:          list[str],
        walk:           dict,
        verse_evidence: list[dict],
        amr_rulings:    list[dict],
        mode:           str,
    ) -> str:
        """
        Build a deterministic conclusion string from template.
        No LLM. All templates carry identical propositional payload:
          root + meaning + n_verses + verse_refs + ruling_type + ruling_sample + maqasid_categories
        Variation is surface form only.
        """
        visited_nodes = walk.get("visited_nodes") or {}
        ms            = walk.get("modal_summary")  or {}
        onto_cats     = ms.get("ontological_categories", [])

        # Template family selection
        ruling_types = {r["type"] for r in amr_rulings}
        if len(roots) > 1:
            template_family = "MULTI_ROOT"
        elif "AMR" in ruling_types:
            template_family = "AMR_SINGLE"
        elif "NAHY" in ruling_types:
            template_family = "NAHY_SINGLE"
        else:
            template_family = "CLASSIFICATION"

        template = _pick_template(template_family, roots)

        # ── root + meaning ────────────────────────────────────────────────────
        # For MULTI_ROOT: root = comma-joined list. For single: root = BW string.
        root_display  = ", ".join(roots) if len(roots) > 1 else (roots[0] if roots else "?")
        primary_root  = roots[0] if roots else "?"

        # meaning: lem → form → BW root (two-stage fallback)
        # lem may be absent on POS/particle nodes; form more reliably populated.
        seed_nodes  = walk.get("seed_nodes") or []
        meaning_str = ""
        for nid in seed_nodes[:10]:
            attrs = visited_nodes.get(nid) or {}
            if attrs.get("root") == primary_root:
                meaning_str = attrs.get("lem") or attrs.get("form") or ""
                if meaning_str:
                    break
        if not meaning_str:
            meaning_str = primary_root  # final fallback: BW root itself

        # ── ruling fields ─────────────────────────────────────────────────────
        ruling_type   = ""
        ruling_sample = "—"
        ruling_ref    = ""
        if amr_rulings:
            r             = amr_rulings[0]
            ruling_type   = "OBLIGATION" if r["type"] == "AMR" else "PROHIBITION"
            ruling_sample = r.get("text", "—")
            ruling_ref    = r.get("ref", "")

        # ── verse fields ──────────────────────────────────────────────────────
        # Prefer ruling_ref as the primary cite when verse_evidence refs are absent.
        verse_refs_str = ", ".join(e["ref"] for e in verse_evidence if e.get("ref")) or ruling_ref or "—"
        n_verses       = len(verse_evidence)

        # ── maqasid ───────────────────────────────────────────────────────────
        maqasid_categories = ", ".join(onto_cats[:4]) if onto_cats else "—"

        # All possible kwargs are passed; templates reference only what they use.
        # Unused kwargs are silently ignored by str.format().
        return template.format(
            root               = root_display,
            meaning            = meaning_str,
            n_verses           = n_verses,
            verse_refs         = verse_refs_str,
            ruling_type        = ruling_type,
            ruling_sample      = ruling_sample,
            ruling_ref         = ruling_ref,
            maqasid_categories = maqasid_categories,
        )

    # ── _compute_confidence ───────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        roots_walked:    list,
        verse_evidence:  list,
        amr_rulings:     list,
        edges_traversed: list,
        maqasid_cats:    list,
    ) -> float:
        """
        Evidence density — the graph's own assessment of answer completeness.
        Score breakdown (max 1.0):
          roots      0.15  — up to 5 roots walked
          verses     0.30  — up to 5 verse references
          rulings    0.30  — up to 3 AMR/NAHY rulings
          edges      0.15  — up to 20 edges traversed
          maqasid    0.10  — at least 2 maqasid categories
        """
        score = 0.0
        if roots_walked:
            score += 0.15 * min(len(roots_walked), 5) / 5
        if verse_evidence:
            score += 0.30 * min(len(verse_evidence), 5) / 5
        if amr_rulings:
            score += 0.30 * min(len(amr_rulings), 3) / 3
        if edges_traversed:
            score += 0.15 * min(len(edges_traversed), 20) / 20
        if len(maqasid_cats) >= 2:
            score += 0.10
        return round(score, 3)

    # ── reason ────────────────────────────────────────────────────────────────

    def reason(self, roots: list[str], question: str = "") -> GraphConclusion:
        """
        Main entry point. Pure graph operations → GraphConclusion.

        Args:
            roots:    Buckwalter roots to reason about
            question: For logging only — not used in graph ops

        Returns:
            GraphConclusion with evidence, derived_statement, confidence, mode
        """
        # Deduplicate, strip SOURCE roots (handled axiomatically upstream)
        source_roots = getattr(self._tmq, "_SOURCE_ROOTS", set())
        walk_roots   = list(dict.fromkeys(r for r in roots if r not in source_roots))

        if not walk_roots:
            logger.debug("GraphReasoner: no walkable roots (all SOURCE or empty)")
            return GraphConclusion(
                roots_walked=[], nodes_visited=[], edges_traversed=[],
                verse_evidence=[], amr_rulings=[], families_touched=set(),
                maqasid_categories=[], derived_statement="[no walkable roots]",
                confidence=0.0, mode="QIYAS",
            )

        # 1. Walk
        walk = self.walk_and_collect(walk_roots)
        edges_flat = walk.get("_edges_flat") or []
        fc         = walk.get("family_counts") or {}
        ms         = walk.get("modal_summary")  or {}
        onto_cats  = ms.get("ontological_categories", [])

        # Filter roots_walked to only roots that actually matched nodes in the graph.
        # Without this, input roots with 0 matches (wrong BW, typos, SOURCE roots
        # caught by _SOURCE_ROOTS) inflate the root-count score in _compute_confidence
        # and can push multi-root queries over the threshold on false evidence.
        seed_nodes = walk.get("seed_nodes") or []
        visited    = walk.get("visited_nodes") or {}
        roots_with_nodes: set[str] = set()
        for nid in seed_nodes:
            r = (visited.get(nid) or {}).get("root")
            if r:
                roots_with_nodes.add(r)
        # Also accept case-insensitive matches (tmq.py does the same fallback)
        walk_roots_lower = {r.lower(): r for r in walk_roots}
        roots_found = [
            r for r in walk_roots
            if r in roots_with_nodes
            or r.lower() in {rn.lower() for rn in roots_with_nodes}
        ]
        # Preserve original order; if nothing matched, roots_found is empty
        walk_roots = roots_found

        # Non-noise families
        families_touched = {f for f in fc if f not in _SKIP_FAMILIES}

        # 2. Verse evidence
        verse_evidence = self.find_verse_evidence(walk, walk_roots)

        # 3. AMR rulings
        amr_rulings = self.match_amr_rulings(walk_roots)

        # 4. Mode
        mode = _classify_mode(fc, onto_cats, amr_rulings)

        # 5. Derived statement
        derived_statement = self.derive_statement(
            walk_roots, walk, verse_evidence, amr_rulings, mode
        )

        # 6. Confidence
        confidence = self._compute_confidence(
            walk_roots, verse_evidence, amr_rulings, edges_flat, onto_cats
        )

        gc = GraphConclusion(
            roots_walked       = walk_roots,
            nodes_visited      = list((walk.get("visited_nodes") or {}).keys()),
            edges_traversed    = edges_flat,
            verse_evidence     = verse_evidence,
            amr_rulings        = amr_rulings,
            families_touched   = families_touched,
            maqasid_categories = onto_cats,
            derived_statement  = derived_statement,
            confidence         = confidence,
            mode               = mode,
        )

        logger.debug(
            f"GraphReasoner: roots={walk_roots}, nodes={len(gc.nodes_visited)}, "
            f"edges={len(gc.edges_traversed)}, verses={len(gc.verse_evidence)}, "
            f"rulings={len(gc.amr_rulings)}, confidence={gc.confidence:.3f}, mode={gc.mode}"
        )

        return gc
