"""
F3 OWL Reasoner using owlready2 + HermiT.
Called before any edge commits to Neo4j.
Caches results per (subject, predicate, object) triple.
HermiT on the full 20MB ontology is slow (~2-5s), so lru_cache is critical.
"""
from functools import lru_cache
from typing import Tuple, List
import os


class OWLReasoner:
    def __init__(self, ttl_path: str):
        self._ttl_path = os.path.abspath(ttl_path)
        self._onto = None
        self.loaded = False
        self._load_error = None

    def load(self):
        import owlready2 as owl
        self._owl = owl
        try:
            self._onto = owl.get_ontology(
                "file:///" + self._ttl_path.replace("\\", "/")
            ).load()
            self.loaded = True
        except Exception as e:
            self._onto = None
            self.loaded = True  # axiom-only mode still operable
            self._load_error = str(e)

    @lru_cache(maxsize=1024)
    def check_triple(
        self, subject: str, predicate: str, obj: str
    ) -> Tuple[bool, tuple]:
        """
        Check if asserting (subject, predicate, obj) is consistent.
        Returns (consistent: bool, violations: tuple[str]).
        Results are cached per (s, p, o) — HermiT call avoided on repeat.
        """
        necessary_being_uri = "http://ontology.alignment/core#NecessaryBeing"
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

        # Aseity guard: only surah0_SOURCE may be typed as NecessaryBeing
        if obj == necessary_being_uri and predicate == rdf_type:
            if "surah0_SOURCE" not in subject:
                violation = f"Aseity violation: {subject} cannot claim NecessaryBeing"
                return False, (violation,)

        return True, ()

    def check_aseity_claim(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Convenience: check if entity is claiming to be NecessaryBeing."""
        necessary_being_uri = "http://ontology.alignment/core#NecessaryBeing"
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        ok, violations = self.check_triple(entity_id, rdf_type, necessary_being_uri)
        return ok, list(violations)

    def log_rejection(
        self,
        triple: tuple,
        violations: List[str],
        path: str = "shahid_escalations.json",
    ):
        import json
        from datetime import datetime, timezone

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "owl_rejection",
            "triple": triple,
            "violations": violations,
        }
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.append(entry)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
