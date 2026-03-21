"""
core/memory.py — Choice memory

Persists what was chosen and why: question, roots, TMQ context,
top families, constrained prompt, generated response, and the chosen mode.

Stored as RDF triples (rdflib) in choice_memory.ttl.
Queryable by root, by family, by mode.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import RDF, XSD
    _RDF_OK = True
except ImportError:
    _RDF_OK = False
    logger.warning("rdflib not available — choice memory will be in-memory only")

IKHTIYAR = Namespace("http://ikhtiyar.ai/choice/") if _RDF_OK else None


@dataclass
class ChoiceRecord:
    timestamp: str
    question: str
    roots: list[str]
    top_families: list[str]
    onto_categories: list[str]
    aseity_risk: bool
    mode: str                    # HAQQ / QIYAS / SILENCE
    response: str
    tmq_context: str
    walk_stats: dict
    confidence: float = 0.0      # program-computed traversal confidence (reward signal)
    grade: str = "UNCERTAIN"     # VERIFIED / PROBABLE / UNCERTAIN / CONTESTED


class ChoiceMemory:
    def __init__(self, ttl_path: Optional[str] = None):
        self._records: list[ChoiceRecord] = []
        self._ttl_path = Path(ttl_path) if ttl_path else None

        if _RDF_OK:
            self._graph = Graph()
            self._graph.bind("ikhtiyar", IKHTIYAR)
            if self._ttl_path and self._ttl_path.exists():
                try:
                    self._graph.parse(str(self._ttl_path), format="turtle")
                    logger.info(f"Loaded {len(self._graph)} choice triples from {self._ttl_path}")
                except Exception as e:
                    logger.warning(f"Could not load choice memory TTL: {e}")
        else:
            self._graph = None

    def record(self, rec: ChoiceRecord) -> None:
        self._records.append(rec)
        if _RDF_OK and self._graph is not None:
            self._store_rdf(rec)

    def recent(self, n: int = 10) -> list[ChoiceRecord]:
        return self._records[-n:]

    def by_family(self, family: str) -> list[ChoiceRecord]:
        return [r for r in self._records if family in r.top_families]

    def by_root(self, root: str) -> list[ChoiceRecord]:
        return [r for r in self._records if root in r.roots]

    def save(self) -> None:
        if not _RDF_OK or self._graph is None or self._ttl_path is None:
            return
        try:
            self._ttl_path.parent.mkdir(parents=True, exist_ok=True)
            self._graph.serialize(str(self._ttl_path), format="turtle")
            logger.debug(f"Choice memory saved to {self._ttl_path}")
        except Exception as e:
            logger.warning(f"Could not save choice memory: {e}")

    def as_list(self) -> list[dict]:
        return [asdict(r) for r in self._records]

    def _store_rdf(self, rec: ChoiceRecord) -> None:
        ts_safe = rec.timestamp.replace(":", "-").replace(" ", "_")
        node = IKHTIYAR[f"choice_{ts_safe}_{len(self._records)}"]
        g = self._graph
        g.add((node, RDF.type, IKHTIYAR.Choice))
        g.add((node, IKHTIYAR.timestamp,  Literal(rec.timestamp, datatype=XSD.string)))
        g.add((node, IKHTIYAR.question,   Literal(rec.question,  datatype=XSD.string)))
        g.add((node, IKHTIYAR.mode,       Literal(rec.mode,      datatype=XSD.string)))
        g.add((node, IKHTIYAR.aseityRisk,  Literal(rec.aseity_risk,  datatype=XSD.boolean)))
        g.add((node, IKHTIYAR.confidence,  Literal(rec.confidence,   datatype=XSD.float)))
        g.add((node, IKHTIYAR.grade,       Literal(rec.grade,        datatype=XSD.string)))
        for r in rec.roots:
            g.add((node, IKHTIYAR.root,   Literal(r, datatype=XSD.string)))
        for f in rec.top_families:
            g.add((node, IKHTIYAR.family, Literal(f, datatype=XSD.string)))
        for c in rec.onto_categories:
            g.add((node, IKHTIYAR.maqasid, Literal(c, datatype=XSD.string)))
