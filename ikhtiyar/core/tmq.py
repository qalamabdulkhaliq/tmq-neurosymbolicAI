"""
core/tmq.py — TMQ v12 Hypergraph API

Schema (TMQ_v12.json):
  node_registry: {node_id: {tier, form, lem, root, pos, loc:[s,v,w,seg]}}
  hyperedges:    {edge_id: {nodes:[node_ids], family, tier, scope, meta, modal}}

Roots are stored in Buckwalter transliteration (e.g. "smw", "ktb", "Alh").
"""

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TMQGraph:
    def __init__(self, path: str):
        path = Path(path)
        logger.info(f"Loading TMQ from {path} ...")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._nodes: dict = data["node_registry"]          # node_id → attrs
        self._edges: dict = data["hyperedges"]              # edge_id → {nodes, family, ...}

        # Index: root → list of node_ids that carry that root
        self._root_index: dict[str, list[str]] = defaultdict(list)
        for nid, attrs in self._nodes.items():
            r = attrs.get("root")
            if r:
                self._root_index[r].append(nid)

        # Index: node_id → list of edge_ids that contain it
        self._node_to_edges: dict[str, list[str]] = defaultdict(list)
        for eid, edge in self._edges.items():
            for nid in edge.get("nodes", []):
                self._node_to_edges[nid].append(eid)

        logger.info(
            f"TMQ ready — {len(self._nodes):,} nodes, "
            f"{len(self._edges):,} hyperedges, "
            f"{len(self._root_index):,} distinct roots"
        )

    # ── Basic accessors ────────────────────────────────────────────────────────

    def edge_families(self) -> list[str]:
        """All distinct edge family names in the graph."""
        return sorted({e["family"] for e in self._edges.values()})

    def node(self, node_id: str) -> Optional[dict]:
        return self._nodes.get(node_id)

    def edge(self, edge_id: str) -> Optional[dict]:
        return self._edges.get(edge_id)

    # ── Root → nodes ──────────────────────────────────────────────────────────

    def roots_to_nodes(self, roots: list[str]) -> list[str]:
        """Return all node IDs whose root field matches any of the given roots."""
        result = []
        for r in roots:
            result.extend(self._root_index.get(r, []))
        return result

    # ── Node neighbourhood ────────────────────────────────────────────────────

    def edges_for_node(self, node_id: str, families: Optional[list[str]] = None) -> list[dict]:
        """All hyperedges containing node_id, optionally filtered by family."""
        out = []
        for eid in self._node_to_edges.get(node_id, []):
            e = self._edges[eid]
            if families is None or e["family"] in families:
                out.append({"edge_id": eid, **e})
        return out

    def neighbors(self, node_id: str, families: Optional[list[str]] = None) -> list[dict]:
        """
        Return neighbouring nodes reached via any hyperedge from node_id.
        Each result: {node_id, attrs, via_edge, family, modal}
        """
        seen = set()
        result = []
        for e in self.edges_for_node(node_id, families):
            for nid in e["nodes"]:
                if nid != node_id and nid not in seen:
                    seen.add(nid)
                    result.append({
                        "node_id": nid,
                        "attrs":   self._nodes.get(nid, {}),
                        "via_edge": e["edge_id"],
                        "family":   e["family"],
                        "modal":    e.get("modal", {}),
                    })
        return result

    # ── BFS walk ──────────────────────────────────────────────────────────────

    def walk(
        self,
        roots: list[str],
        depth: int = 2,
        families: Optional[list[str]] = None,
    ) -> dict:
        """
        BFS from all nodes matching `roots`, up to `depth` hops.
        Returns:
          {
            "seed_nodes": [...],
            "visited_nodes": {node_id: attrs},
            "visited_edges": {edge_id: edge},
            "family_counts": {family: count},
            "modal_summary": {intensity_avg, address_modes, ontological_categories},
          }
        """
        seed_nodes = self.roots_to_nodes(roots)
        if not seed_nodes:
            return {
                "seed_nodes": [],
                "visited_nodes": {},
                "visited_edges": {},
                "family_counts": {},
                "modal_summary": {},
            }

        visited_nodes: dict[str, dict] = {}
        visited_edges: dict[str, dict] = {}
        family_counts: dict[str, int] = defaultdict(int)
        intensities: list[float] = []
        address_modes: list[int] = []
        onto_cats: list[str] = []

        frontier = deque((nid, 0) for nid in seed_nodes)
        seen_nodes = set(seed_nodes)

        while frontier:
            nid, d = frontier.popleft()
            visited_nodes[nid] = self._nodes.get(nid, {})

            if d >= depth:
                continue

            for e in self.edges_for_node(nid, families):
                eid = e["edge_id"]
                if eid not in visited_edges:
                    visited_edges[eid] = e
                    family_counts[e["family"]] += 1
                    modal = e.get("modal") or {}
                    if modal.get("intensity") is not None:
                        intensities.append(modal["intensity"])
                    if modal.get("address_mode") is not None:
                        address_modes.append(modal["address_mode"])
                    for layer in (modal.get("ontological_layers") or []):
                        if layer.get("category"):
                            onto_cats.append(layer["category"])

                for nb_id in e["nodes"]:
                    if nb_id not in seen_nodes:
                        seen_nodes.add(nb_id)
                        frontier.append((nb_id, d + 1))

        modal_summary = {
            "intensity_avg": round(sum(intensities) / len(intensities), 3) if intensities else None,
            "address_modes": list(set(address_modes)),
            "ontological_categories": list(set(onto_cats)),
        }

        return {
            "seed_nodes":    seed_nodes,
            "visited_nodes": visited_nodes,
            "visited_edges": visited_edges,
            "family_counts": dict(family_counts),
            "modal_summary": modal_summary,
        }

    # ── Semantic summary (for prompt injection) ───────────────────────────────

    def describe_walk(self, walk_result: dict) -> str:
        """
        Convert a walk() result into a compact string suitable for LLM prompt injection.
        """
        if not walk_result.get("seed_nodes"):
            return "[TMQ: no matching nodes for these roots]"

        fc = walk_result["family_counts"]
        ms = walk_result["modal_summary"]
        onto = ms.get("ontological_categories", [])
        addr = ms.get("address_modes", [])
        intensity = ms.get("intensity_avg")

        lines = [
            f"[TMQ HYPERGRAPH CONTEXT]",
            f"Seed nodes: {len(walk_result['seed_nodes'])} | "
            f"Subgraph: {len(walk_result['visited_nodes'])} nodes, "
            f"{len(walk_result['visited_edges'])} hyperedges",
        ]

        if fc:
            top = sorted(fc.items(), key=lambda x: -x[1])[:6]
            lines.append("Edge families: " + ", ".join(f"{f}({n})" for f, n in top))

        if onto:
            lines.append("Maqasid categories: " + ", ".join(onto[:5]))

        addr_labels = {1: "singular", 2: "dual", 3: "plural", 4: "majestic"}
        if addr:
            labels = [addr_labels.get(a, str(a)) for a in addr]
            lines.append("Address modes: " + ", ".join(labels))

        if intensity is not None:
            lines.append(f"Modal intensity: {intensity}")

        return "\n".join(lines)
