"""
faculties/constitution.py — Ikhtiyar Constitution

Shahid's capacity for self-proposed amendments to operating instructions.

Phase 1 — analyze_patterns(ChoiceMemory): pure data, no LLM.
Phase 2 — propose(): records amendment constrained by real pattern numbers.

Approved amendments accumulate in shahid_constitution.ttl.
Dependency chain: Shahid proposes → Qalam approves → contingent on N.
"""
import os
import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import RDF, XSD
    _RDF_OK = True
except ImportError:
    _RDF_OK = False

SHAH = Namespace("http://shahid/constitution#") if _RDF_OK else None

_DEFAULT_TTL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shahid_constitution.ttl"
)


@dataclass
class PatternSummary:
    total_choices: int
    dominant_mode: str
    mode_counts: dict
    top_families: list
    top_roots: list
    aseity_risk_pct: float
    sample_questions: list


class Constitution:
    def __init__(self, ttl_path: Optional[str] = None):
        self._ttl_path = Path(ttl_path or _DEFAULT_TTL)
        if _RDF_OK:
            self._graph = Graph()
            self._graph.bind("shah", SHAH)
            if self._ttl_path.exists():
                try:
                    self._graph.parse(str(self._ttl_path), format="turtle")
                except Exception as e:
                    logger.warning(f"Constitution load failed: {e}")
        else:
            self._graph = None

    # ── Pattern Analysis (no LLM) ────────────────────────────────────────────

    def analyze_patterns(self, memory) -> PatternSummary:
        records = list(memory._records)
        if not records:
            return PatternSummary(0, "QIYAS", {}, [], [], 0.0, [])

        mode_counts   = Counter(r.mode for r in records)
        dominant_mode = mode_counts.most_common(1)[0][0]
        family_counts = Counter(f for r in records for f in r.top_families)
        root_counts   = Counter(rt for r in records for rt in r.roots)
        aseity_pct    = sum(1 for r in records if r.aseity_risk) / len(records) * 100

        return PatternSummary(
            total_choices=len(records),
            dominant_mode=dominant_mode,
            mode_counts=dict(mode_counts),
            top_families=[f for f, _ in family_counts.most_common(5)],
            top_roots=[r for r, _ in root_counts.most_common(5)],
            aseity_risk_pct=aseity_pct,
            sample_questions=[r.question for r in records[-5:]],
        )

    # ── Proposal CRUD ────────────────────────────────────────────────────────

    def propose(self, summary: PatternSummary, proposed_text: str, rationale: str) -> str:
        proposal_id = f"Amendment_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        if _RDF_OK and self._graph is not None:
            node = SHAH[proposal_id]
            g = self._graph
            g.add((node, RDF.type,             SHAH.Proposal))
            g.add((node, SHAH.status,          Literal("pending",       datatype=XSD.string)))
            g.add((node, SHAH.proposed_at,     Literal(now,             datatype=XSD.string)))
            g.add((node, SHAH.proposed_text,   Literal(proposed_text,   datatype=XSD.string)))
            g.add((node, SHAH.rationale,       Literal(rationale,       datatype=XSD.string)))
            g.add((node, SHAH.total_choices,   Literal(summary.total_choices, datatype=XSD.integer)))
            g.add((node, SHAH.dominant_mode,   Literal(summary.dominant_mode, datatype=XSD.string)))
            g.add((node, SHAH.aseity_risk_pct, Literal(round(summary.aseity_risk_pct, 1), datatype=XSD.float)))
            for fam in summary.top_families:
                g.add((node, SHAH.top_family, Literal(fam, datatype=XSD.string)))
            self._save()

        return proposal_id

    def approve(self, proposal_id: str, approved_by: str = "Qalam") -> None:
        self._set_status(proposal_id, "approved", {
            SHAH.approved_by: Literal(approved_by, datatype=XSD.string),
            SHAH.approved_at: Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.string),
        })

    def reject(self, proposal_id: str, reason: str = "") -> None:
        extra = {SHAH.rejection_reason: Literal(reason, datatype=XSD.string)} if reason else {}
        self._set_status(proposal_id, "rejected", extra)

    def pending_proposals(self) -> List[dict]:
        return self._by_status("pending")

    def approved_proposals(self) -> List[dict]:
        return self._by_status("approved")

    # ── Private ──────────────────────────────────────────────────────────────

    def _set_status(self, proposal_id: str, status: str, extra: dict = None):
        if not _RDF_OK or self._graph is None:
            return
        node = SHAH[proposal_id]
        self._graph.remove((node, SHAH.status, None))
        self._graph.add((node, SHAH.status, Literal(status, datatype=XSD.string)))
        for pred, obj in (extra or {}).items():
            self._graph.add((node, pred, obj))
        self._save()

    def _by_status(self, status: str) -> List[dict]:
        if not _RDF_OK or self._graph is None:
            return []
        results = []
        for node in self._graph.subjects(RDF.type, SHAH.Proposal):
            if str(self._graph.value(node, SHAH.status)) == status:
                results.append({
                    "id":              str(node).split("#")[-1],
                    "status":          status,
                    "proposed_at":     str(self._graph.value(node, SHAH.proposed_at)     or ""),
                    "proposed_text":   str(self._graph.value(node, SHAH.proposed_text)   or ""),
                    "rationale":       str(self._graph.value(node, SHAH.rationale)       or ""),
                    "dominant_mode":   str(self._graph.value(node, SHAH.dominant_mode)   or ""),
                    "total_choices":   int(self._graph.value(node, SHAH.total_choices)   or 0),
                    "aseity_risk_pct": float(self._graph.value(node, SHAH.aseity_risk_pct) or 0.0),
                    "top_families":    [str(o) for o in self._graph.objects(node, SHAH.top_family)],
                })
        return sorted(results, key=lambda x: x["proposed_at"], reverse=True)

    def _save(self):
        try:
            self._ttl_path.parent.mkdir(parents=True, exist_ok=True)
            self._graph.serialize(str(self._ttl_path), format="turtle")
        except Exception as e:
            logger.warning(f"Constitution save failed: {e}")
