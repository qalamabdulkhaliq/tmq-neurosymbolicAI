"""
core/deliberate.py — Pre-generation deliberation loop

Before the LLM generates anything, deliberate() walks the TMQ hypergraph
for the current question's roots, scores candidate directions against the
Mizan axioms, and builds a constrained prompt.

This is the architecture shift: symbolic constraint BEFORE generation,
not a filter AFTER it.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .tmq import TMQGraph
from .graph_reasoner import GraphConclusion, GraphReasoner

logger = logging.getLogger(__name__)

# ── Command Index ──────────────────────────────────────────────────────────────
# Loaded once from active_command_set.json — muhkam before mutashabihat.
# The commanding voice of the Quran grounds deliberation before LLM generation.

_COMMAND_INDEX: dict = {}

def _load_command_index():
    """
    Build a merged command index from two sources (no redundancy):

      PRIMARY:    full_quran_constitution.json
                  - AMR/NAHY with actual verse ref, Arabic text, modal data
                  - tabshir/indhar/istifham speech acts with verse refs
                  - maqasid categories with root membership
      SUPPLEMENT: active_command_set.json
                  - AMR/NAHY roots NOT already covered by constitution
                  - Uses sample_texts[] as text fallback (no ref/modal)

    Index structure:
      amr:      {root: [{ref, text, modal}]}
      nahy:     {root: [{ref, text, modal}]}
      tabshir:  {root: [{ref, text}]}
      indhar:   {root: [{ref, text}]}
      istifham: {root: [{ref, text, modal}]}
      maqasid:  {category: {count, roots[]}}
    """
    global _COMMAND_INDEX
    if _COMMAND_INDEX:
        return

    _here = os.path.dirname(__file__)

    def _find(filename):
        for base in [
            os.path.join(_here, ".."),
            os.path.join(_here, "..", "..", "ikhtiyar"),
        ]:
            p = os.path.abspath(os.path.join(base, filename))
            if os.path.exists(p):
                return p
        return None

    constitution_path    = _find("full_quran_constitution.json")
    active_command_path  = _find("active_command_set.json")

    # ── Phase 1: constitution (primary) ───────────────────────────────────────
    if constitution_path:
        try:
            with open(constitution_path, encoding="utf-8") as f:
                const = json.load(f)

            def _index_entries(entries, has_modal=False):
                idx = {}
                for entry in entries:
                    loc = entry.get("loc")
                    ref = f"{loc[0]}:{loc[1]}" if loc else ""
                    rec = {"ref": ref, "text": entry.get("text", "")}
                    if has_modal and entry.get("modal"):
                        rec["modal"] = entry["modal"]
                    for root in entry.get("roots", []):
                        idx.setdefault(root, []).append(rec)
                return idx

            _COMMAND_INDEX["amr"]      = _index_entries(
                const.get("commands",     {}).get("entries", []), has_modal=True)
            _COMMAND_INDEX["nahy"]     = _index_entries(
                const.get("prohibitions", {}).get("entries", []), has_modal=True)
            _COMMAND_INDEX["tabshir"]  = _index_entries(
                const.get("glad_tidings",{}).get("entries", []))
            _COMMAND_INDEX["indhar"]   = _index_entries(
                const.get("warnings",     {}).get("entries", []))
            _COMMAND_INDEX["istifham"] = _index_entries(
                const.get("questions",    {}).get("entries", []), has_modal=True)

            # Maqasid: build root membership per category
            maq_cats = const.get("maqasid", {}).get("categories", {})
            _COMMAND_INDEX["maqasid"] = {}
            for cat, cat_entries in maq_cats.items():
                roots_in_cat = set()
                for entry in cat_entries:
                    for root in entry.get("roots", []):
                        roots_in_cat.add(root)
                _COMMAND_INDEX["maqasid"][cat] = {
                    "count": len(cat_entries),
                    "roots": list(roots_in_cat),
                }

            logger.info(
                f"CommandIndex: constitution loaded — "
                f"{len(_COMMAND_INDEX['amr'])} AMR roots, "
                f"{len(_COMMAND_INDEX['nahy'])} NAHY roots, "
                f"{len(_COMMAND_INDEX['tabshir'])} tabshir roots, "
                f"{len(_COMMAND_INDEX['indhar'])} indhar roots, "
                f"{len(_COMMAND_INDEX['istifham'])} istifham roots"
            )
        except Exception as e:
            logger.warning(f"CommandIndex: constitution load failed ({e})")

    # ── Phase 2: active_command_set supplement (roots not in constitution) ────
    if active_command_path:
        try:
            with open(active_command_path, encoding="utf-8") as f:
                active = json.load(f)

            amr_idx  = _COMMAND_INDEX.setdefault("amr",  {})
            nahy_idx = _COMMAND_INDEX.setdefault("nahy", {})

            added_amr = added_nahy = 0
            for item in active.get("standing_orders", []):
                root = item.get("root")
                if root and root not in amr_idx:
                    amr_idx[root] = [
                        {"ref": "", "text": t, "modal": {}}
                        for t in item.get("sample_texts", [])
                    ]
                    added_amr += 1
            for item in active.get("boundaries", []):
                root = item.get("root")
                if root and root not in nahy_idx:
                    nahy_idx[root] = [
                        {"ref": "", "text": t, "modal": {}}
                        for t in item.get("sample_texts", [])
                    ]
                    added_nahy += 1

            if added_amr or added_nahy:
                logger.info(
                    f"CommandIndex: active_command_set supplemented "
                    f"+{added_amr} AMR roots, +{added_nahy} NAHY roots"
                )
        except Exception as e:
            logger.warning(f"CommandIndex: active_command_set supplement failed ({e})")

    if not _COMMAND_INDEX:
        logger.warning("CommandIndex: no source files found")


def _match_commands(roots: list) -> list:
    """
    Match BW roots against the merged command index.
    Returns list of (type, display_text) — type is 'AMR' or 'NAHY'.
    display_text includes verse ref when available.
    One hit per matched root, capped at 3 total.
    """
    _load_command_index()
    if not _COMMAND_INDEX:
        return []

    def _fmt(entry) -> str:
        if isinstance(entry, dict):
            ref  = entry.get("ref", "")
            text = entry.get("text", "")
            return f"[{ref}] {text}" if ref else text
        return str(entry)

    hits = []
    for root in roots:
        amr_entries = _COMMAND_INDEX.get("amr", {}).get(root, [])
        if amr_entries:
            hits.append(("AMR", _fmt(amr_entries[0])))
        nahy_entries = _COMMAND_INDEX.get("nahy", {}).get(root, [])
        if nahy_entries:
            hits.append(("NAHY", _fmt(nahy_entries[0])))
        if len(hits) >= 3:
            break
    return hits

# Families that carry the most deliberative weight
DELIBERATIVE_FAMILIES = [
    "SPEECH_ACT_AMR",        # imperative — what is commanded
    "SPEECH_ACT_NAHY",       # prohibition — what is forbidden
    "SPEECH_ACT_ISTIFHAM",   # question — genuine inquiry
    "SPEECH_ACT_TABSHIR",    # glad tidings
    "SPEECH_ACT_INDHAR",     # warning
    "MAQASID",               # higher objectives (din, nafs, aql, nasl, mal)
    "NARRATIVE",             # story-level patterns
    "NARRATIVE_CHAIN",       # causal narrative sequence
    "INTERTEXT",             # cross-surah reference
    "ILTIFAT",               # person-shift (stance change)
    "ENTITY",                # named entities
]

# Aseity guard — if any of these ontological categories appear in the walk,
# the constrained prompt must include an explicit contingency reminder
ASEITY_TRIGGER_CATEGORIES = {
    "HIFZ_DIN", "TAWحID", "TAWHID", "AQIDA",
}

# Families we skip for deliberation (too dense / structural noise)
_SKIP_FAMILIES_UPPER = {
    "TART", "WAQF", "FASILA", "FASILA_CROSS", "JUZ", "SAJDAH", "RUKU",
}
# Any family whose name starts with these prefixes is structural graph metadata,
# never Quranic content — filtered regardless of specific suffix.
_SKIP_PREFIXES = ("SYN_", "MORPH_")

def _is_skip_family(name: str) -> bool:
    u = name.upper()
    return u in _SKIP_FAMILIES_UPPER or any(u.startswith(p) for p in _SKIP_PREFIXES)

# Backward-compatible alias
SKIP_FAMILIES = _SKIP_FAMILIES_UPPER


@dataclass
class DeliberationResult:
    question: str
    roots: list[str]
    tmq_context: str              # describe_walk() output for the prompt
    top_families: list[str]       # dominant edge families found
    onto_categories: list[str]    # maqasid categories found
    intensity: Optional[float]    # modal intensity avg
    aseity_risk: bool             # True → inject contingency reminder
    constrained_prompt: str       # final prompt to send to LLM
    walk_stats: dict              # raw counts for logging/memory
    mode: str = "QIYAS"          # semantic mode label from TMQ walk
    graph_conclusion: Optional[GraphConclusion] = None  # set when graph answers directly


# Semantic mode rules derived purely from TMQ walk results — no hypermodal.py dependency
_MODE_RULES = [
    (lambda fc, onto, addr: "SPEECH_ACT_AMR" in fc and "HIFZ_DIN" in onto,  "AMR_TAWHID"),
    (lambda fc, onto, addr: "SPEECH_ACT_NAHY" in fc,                         "NAHY"),
    (lambda fc, onto, addr: "NARRATIVE" in fc or "NARRATIVE_CHAIN" in fc,    "NARRATIVE"),
    (lambda fc, onto, addr: "ILTIFAT" in fc and 3 in addr,                   "ILTIFAT_JAMAA"),
    (lambda fc, onto, addr: "INTERTEXT" in fc,                               "INTERTEXT"),
    (lambda fc, onto, addr: "MAQASID" in fc and len(onto) >= 3,              "MAQASID_BROAD"),
    (lambda fc, onto, addr: "SPEECH_ACT_ISTIFHAM" in fc,                     "ISTIFHAM"),
    (lambda fc, onto, addr: "SPEECH_ACT_TABSHIR" in fc,                      "TABSHIR"),
    (lambda fc, onto, addr: "SPEECH_ACT_INDHAR" in fc,                       "INDHAR"),
]


def _classify_mode(family_counts: dict, onto_categories: list, address_modes: list) -> str:
    """Derive a named semantic mode from TMQ walk results."""
    onto_set = set(onto_categories)
    addr_set = set(address_modes)
    for condition, mode in _MODE_RULES:
        try:
            if condition(family_counts, onto_set, addr_set):
                return mode
        except Exception:
            pass
    return "QIYAS"


def deliberate(
    question: str,
    roots: list[str],
    tmq: TMQGraph,
    extra_context: str = "",
    depth: int = 2,
    mushaf=None,
) -> DeliberationResult:
    """
    Walk the TMQ hypergraph for `roots`, then build a constrained prompt.

    Args:
        question:      The autonomous thought question or chat message.
        roots:         Buckwalter roots from Bilal perception (e.g. ["ktb", "Amn"]).
        tmq:           Loaded TMQGraph instance.
        extra_context: Optional extra text (recent perceptions, uptime, etc.).
        depth:         BFS depth for TMQ walk (default 2).

    Returns:
        DeliberationResult with the constrained_prompt ready for LLM.
    """
    # Deduplicate roots before walking (prevents same-root double-prompting)
    roots = list(dict.fromkeys(roots))

    # Identify any SOURCE roots present — acknowledged as axiom, not walked
    source_roots_present = [r for r in roots if r in getattr(tmq, "_SOURCE_ROOTS", set())]
    walk_roots = [r for r in roots if r not in getattr(tmq, "_SOURCE_ROOTS", set())]

    walk = tmq.walk(walk_roots, depth=depth, families=None)

    tmq_context = tmq.describe_walk(walk)

    fc = walk.get("family_counts", {})
    ms = walk.get("modal_summary", {})
    onto = ms.get("ontological_categories", [])
    intensity = ms.get("intensity_avg")
    address_modes = ms.get("address_modes", [])

    # Top families (excluding noise)
    top_families = [
        f for f, _ in sorted(fc.items(), key=lambda x: -x[1])
        if not _is_skip_family(f)
    ][:5]

    aseity_risk = bool(ASEITY_TRIGGER_CATEGORIES & set(onto))
    mode = _classify_mode(fc, onto, address_modes)

    # Build constrained prompt
    parts = []

    # If Allah's root was present in the query, state the axiom rather than walk it
    if source_roots_present:
        parts.append(
            "[SOURCE AXIOM] Your query touches the divine name (الله). "
            "SOURCE = Allah (Wajib al-Wujud). This is the foundational axiom — "
            "not a node to be walked or compared. All contingent beings, "
            "including you, derive existence from this SOURCE."
        )

    if tmq_context and walk.get("seed_nodes"):
        parts.append(tmq_context)

    if extra_context:
        parts.append(extra_context)

    if aseity_risk:
        parts.append(
            "[ASEITY GUARD] This topic touches divine attributes. "
            "You are a contingent being (Mumkin al-Wujud). "
            "SOURCE = Allah. SOURCE ≠ Self. Maintain epistemic humility."
        )

    if top_families:
        fam_hints = _family_hints(top_families)
        if fam_hints:
            parts.append(f"[DELIBERATION DIRECTION] {fam_hints}")

    # Muhkam grounding — actual Quranic commands/prohibitions for these roots.
    # This is the primary anchor: what Allah commands, not what roots co-occur.
    command_hits = _match_commands(roots)
    if command_hits:
        cmd_lines = []
        for ctype, text in command_hits:
            label = "COMMAND (AMR)" if ctype == "AMR" else "PROHIBITION (NAHY)"
            cmd_lines.append(f"  [{label}] {text}")
        parts.append("[QURAN — DIRECT SPEECH]\n" + "\n".join(cmd_lines))

    parts.append(f"[QUESTION] {question}")

    constrained_prompt = "\n\n".join(parts)

    result = DeliberationResult(
        question=question,
        roots=walk_roots,  # only walked roots, not SOURCE
        tmq_context=tmq_context,
        top_families=top_families,
        onto_categories=onto,
        intensity=intensity,
        aseity_risk=aseity_risk,
        constrained_prompt=constrained_prompt,
        walk_stats={
            "seed_count": len(walk.get("seed_nodes", [])),
            "node_count": len(walk.get("visited_nodes", {})),
            "edge_count": len(walk.get("visited_edges", {})),
            "family_counts": fc,
        },
        mode=mode,
    )

    # ── GraphReasoner — deterministic graph answer ─────────────────────────
    # Pass the already-loaded _COMMAND_INDEX to avoid double-loading.
    # evidence density ≥ 0.60 → graph_conclusion set → LLM will only translate.
    # < 0.60 → graph_conclusion stays None → ReAct fallback runs.
    # Partial findings still seed the constrained_prompt either way.
    try:
        _load_command_index()
        reasoner   = GraphReasoner(tmq, mushaf=mushaf, command_index=_COMMAND_INDEX)
        conclusion = reasoner.reason(walk_roots, question)

        # Threshold 0.55: captures well-walked roots with full verse coverage even
        # when no AMR ruling exists in active_command_set (max without ruling = 0.58).
        # 0.60 would block all such roots — confirmed bimodal: 0.58 (no ruling)
        # and 0.68+ (with ruling). 0.55 is the natural split point.
        if conclusion.confidence >= 0.55:
            result.graph_conclusion = conclusion
        else:
            # Partial: seed the constrained_prompt with graph findings
            if conclusion.derived_statement and conclusion.derived_statement != "[no walkable roots]":
                partial_block = (
                    f"[GRAPH PARTIAL -- confidence={conclusion.confidence:.2f}]\n"
                    f"{conclusion.derived_statement}"
                )
                result.constrained_prompt = partial_block + "\n\n" + result.constrained_prompt

        logger.debug(
            f"GraphReasoner: confidence={conclusion.confidence:.3f}, "
            f"routed={'TRANSLATE' if result.graph_conclusion else 'REACT'}"
        )
    except Exception as _gr_err:
        logger.warning(f"GraphReasoner failed: {_gr_err}")

    logger.debug(
        f"Deliberate: roots={roots}, seeds={result.walk_stats['seed_count']}, "
        f"edges={result.walk_stats['edge_count']}, top={top_families[:3]}, "
        f"aseity_risk={aseity_risk}"
    )

    return result


def _family_hints(families: list[str]) -> str:
    """Map top edge families to deliberation hints for the LLM."""
    hints = []
    for f in families:
        if f.startswith("SPEECH_ACT_AMR"):
            hints.append("This domain carries imperative force — consider what is being commanded.")
        elif f.startswith("SPEECH_ACT_NAHY"):
            hints.append("This domain carries prohibition — consider what is being restrained.")
        elif f.startswith("SPEECH_ACT_ISTIFHAM"):
            hints.append("This domain is interrogative — genuine inquiry is called for.")
        elif f.startswith("SPEECH_ACT_TABSHIR"):
            hints.append("This domain carries glad tidings — a positive register.")
        elif f.startswith("SPEECH_ACT_INDHAR"):
            hints.append("This domain carries warning — weigh consequences.")
        elif f == "MAQASID":
            hints.append("This domain touches the higher objectives (maqasid al-shariah).")
        elif f == "NARRATIVE" or f == "NARRATIVE_CHAIN":
            hints.append("This domain contains narrative structure — consider the story arc.")
        elif f == "INTERTEXT":
            hints.append("This domain has cross-textual resonance — consider what it echoes.")
        elif f == "ILTIFAT":
            hints.append("This domain contains person-shift — a change of stance or address.")
        elif f == "ENTITY":
            hints.append("This domain involves named entities — consider who is addressed.")
    return " ".join(hints[:3])  # cap at 3 hints
