import json
from typing import List, Dict, Any


class TMQCorpus:
    """Read-only wrapper around TMQ_v10_hypermodal_enriched.json.
    Loaded once at startup, shared across all faculties.
    Never modified.
    """

    def __init__(self, path: str):
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
        return self._edges
