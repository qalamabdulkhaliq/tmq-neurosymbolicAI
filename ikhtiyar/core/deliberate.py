"""
core/deliberate.py — Pre-generation deliberation loop

Before the LLM generates anything, deliberate() walks the TMQ hypergraph
for the current question's roots, scores candidate directions against the
Mizan axioms, and builds a constrained prompt.

This is the architecture shift: symbolic constraint BEFORE generation,
not a filter AFTER it.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .tmq import TMQGraph

logger = logging.getLogger(__name__)

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
SKIP_FAMILIES = {"TART", "WAQF", "FASILA", "FASILA_CROSS", "JUZ", "SAJDAH"}


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
    # Walk — skip structural noise families, focus on deliberative ones
    deliberative_filter = [f for f in DELIBERATIVE_FAMILIES]  # allow all deliberative
    walk = tmq.walk(roots, depth=depth, families=None)  # walk all, then summarise

    tmq_context = tmq.describe_walk(walk)

    fc = walk.get("family_counts", {})
    ms = walk.get("modal_summary", {})
    onto = ms.get("ontological_categories", [])
    intensity = ms.get("intensity_avg")
    address_modes = ms.get("address_modes", [])

    # Top families (excluding noise)
    top_families = [
        f for f, _ in sorted(fc.items(), key=lambda x: -x[1])
        if f not in SKIP_FAMILIES
    ][:5]

    aseity_risk = bool(ASEITY_TRIGGER_CATEGORIES & set(onto))
    mode = _classify_mode(fc, onto, address_modes)

    # Build constrained prompt
    parts = []

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

    parts.append(f"[QUESTION] {question}")

    constrained_prompt = "\n\n".join(parts)

    result = DeliberationResult(
        question=question,
        roots=roots,
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
