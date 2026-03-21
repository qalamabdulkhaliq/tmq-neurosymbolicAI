from typing import List, Dict, Optional
import chromadb


class SpectralIndex:
    """Indexes TMQ hyperedge modal vectors in Chroma.
    Vectors encode: [intensity, rawi_resonance, address_mode_norm, eigenstate_norm]
    plus one-hot maqasid layers.
    Never re-embeds text — uses TMQ's pre-computed spectral values directly.
    """

    MAQASID_LABELS = [
        "faith", "life", "intellect", "lineage", "wealth",
        "justice", "environment", "dignity"
    ]

    def __init__(self, persist_dir: str = "chroma_spectral"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            "shahid_spectral",
            metadata={"hnsw:space": "cosine"}
        )

    def _edge_to_vector(self, edge: dict) -> List[float]:
        modal = edge.get("modal") or {}
        intensity = float(modal.get("intensity") or 0.0)
        rawi = float(modal.get("rawi_resonance") or 0.0)
        addr = float(modal.get("address_mode") or 0) / 3.0
        eigen = float(modal.get("dominant_eigenstate") or 0) / 6.0
        layers = modal.get("ontological_layers") or []
        layer_vec = [1.0 if lbl in layers else 0.0 for lbl in self.MAQASID_LABELS]
        return [intensity, rawi, addr, eigen] + layer_vec

    def ingest_tmq(self, tmq, families: Optional[List[str]] = None, batch_size: int = 500):
        edges = tmq.all_edges()
        if families:
            edges = [e for e in edges if e.get("family") in families]
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            ids, vecs, metas = [], [], []
            for j, e in enumerate(batch):
                eid = f"edge_{i+j}_{e.get('family', '?')}"
                ids.append(eid)
                vecs.append(self._edge_to_vector(e))
                modal = e.get("modal") or {}
                metas.append({
                    "family": str(e.get("family", "")),
                    "tier": str(e.get("tier", "")),
                    "eigenstate": int(modal.get("dominant_eigenstate") or 0),
                    "emotional_register": str(modal.get("emotional_register") or ""),
                    "intensity": float(modal.get("intensity") or 0.0),
                })
            self._col.upsert(ids=ids, embeddings=vecs, metadatas=metas)

    def count(self) -> int:
        return self._col.count()

    def search_by_eigenstate(self, eigenstate: int, n: int = 10) -> List[Dict]:
        query_vec = [0.5, 0.5, 0.5, eigenstate / 6.0] + [0.0] * len(self.MAQASID_LABELS)
        results = self._col.query(query_embeddings=[query_vec], n_results=min(n, self.count()))
        hits = []
        for meta in results["metadatas"][0]:
            hits.append({"metadata": meta})
        return hits

    def find_modal_neighbors(self, edge: dict, threshold: float = 0.8, n: int = 10) -> List[Dict]:
        vec = self._edge_to_vector(edge)
        results = self._col.query(query_embeddings=[vec], n_results=min(n + 1, self.count()))
        hits = []
        for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
            if dist <= (1.0 - threshold):
                hits.append({"metadata": meta, "distance": dist})
        return hits[:n]
