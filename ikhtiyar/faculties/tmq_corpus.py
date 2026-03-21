import json
from typing import List, Dict, Any, Optional


class TMQCorpus:
    """Read-only wrapper around TMQ_v10_hypermodal_enriched.json.
    Loaded once at startup, shared across all faculties.
    Never modified.

    Optional enrichment_path: path to TMQ_enrichment_v1.json (or later versions).
    Enrichment edges are appended after base edges; base is never touched.
    """

    def __init__(self, path: str, enrichment_path: Optional[str] = None):
        with open(path, encoding="utf-8") as f:
            self._data = json.load(f)
        self._stats = self._data["stats"]
        self._node_registry: Dict[str, Any] = self._data["node_registry"]
        # hyperedges is a dict keyed by string integers
        raw_edges = self._data["hyperedges"]
        if isinstance(raw_edges, dict):
            self._edges: List[Dict] = list(raw_edges.values())
        else:
            self._edges = raw_edges
        self._data = None

        self._enrichment_families: set = set()
        if enrichment_path:
            self._load_enrichment(enrichment_path)

    def _load_enrichment(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            enrich = json.load(f)
        raw = enrich.get("hyperedges", {})
        new_edges = list(raw.values()) if isinstance(raw, dict) else raw
        for e in new_edges:
            if isinstance(e, dict):
                self._enrichment_families.add(e.get("family", ""))
        self._edges.extend(new_edges)

    @property
    def enrichment_families(self) -> set:
        return set(self._enrichment_families)

    @property
    def total_edges(self) -> int:
        return self._stats["total_edges"]

    @property
    def total_nodes(self) -> int:
        return self._stats["total_nodes"]

    def nodes_by_tier(self, tier: str) -> List[Dict]:
        return [v for v in self._node_registry.values()
                if isinstance(v, dict) and v.get("tier") == tier]

    def edges_by_family(self, family: str) -> List[Dict]:
        return [e for e in self._edges
                if isinstance(e, dict) and e.get("family") == family]

    def edges_by_eigenstate(self, eigenstate: int) -> List[Dict]:
        result = []
        for e in self._edges:
            if not isinstance(e, dict):
                continue
            modal = e.get("modal")
            if isinstance(modal, dict) and modal.get("dominant_eigenstate") == eigenstate:
                result.append(e)
        return result

    def node(self, node_id: str) -> Dict:
        return self._node_registry.get(node_id, {})

    def all_edges(self) -> List[Dict]:
        return list(self._edges)
