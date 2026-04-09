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

    # Roots that resolve to the divine name / SOURCE node.
    # Since SOURCE = Allah is a foundational axiom (not a discoverable proposition),
    # walking FROM these nodes as peers to other roots is ontologically wrong.
    # They are acknowledged in the prompt separately, not BFS-expanded.
    _SOURCE_ROOTS = {"Alh", "ALh", "alh", "allah", "Allah"}

    def roots_to_nodes(self, roots: list[str]) -> list[str]:
        """
        Return deduplicated node IDs whose root field matches any of the given roots.
        SOURCE roots (Alh / Allah) are excluded — SOURCE is an axiom, not a walk target.
        """
        seen: set[str] = set()
        result: list[str] = []
        for r in roots:
            if r in self._SOURCE_ROOTS:
                continue  # SOURCE = Allah is presupposed, not walked
            found = self._root_index.get(r, [])
            if not found:
                # Case-insensitive fallback
                r_lower = r.lower()
                for key in self._root_index:
                    if key.lower() == r_lower:
                        found = self._root_index[key]
                        break
            for nid in found:
                if nid not in seen:
                    seen.add(nid)
                    result.append(nid)
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
            for nid in (e["nodes"] or []):
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

                for nb_id in (e["nodes"] or []):
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
        Surfaces actual ayat segments first — graph stats are secondary navigation.
        """
        if not walk_result.get("seed_nodes"):
            return "[TMQ: no matching nodes for these roots]"

        fc = walk_result["family_counts"]
        ms = walk_result["modal_summary"]
        onto = ms.get("ontological_categories", [])
        addr = ms.get("address_modes", [])
        intensity = ms.get("intensity_avg")

        lines = ["[TMQ — AYAT CONTEXT]"]

        # ── Ayat first: group seed nodes by verse, show Arabic form + root ──
        verse_map: dict[tuple, list[str]] = {}
        for nid in walk_result["seed_nodes"]:
            attrs = walk_result["visited_nodes"].get(nid) or self._nodes.get(nid, {})
            loc = attrs.get("loc")
            form = attrs.get("form", "")
            root = attrs.get("root", "")
            if loc and len(loc) >= 2 and form:
                key = (loc[0], loc[1])   # (surah, ayah)
                verse_map.setdefault(key, []).append(
                    f"{form}[{root}]" if root else form
                )
        for (s, v), segs in sorted(verse_map.items())[:6]:
            lines.append(f"  {s}:{v}  {' '.join(segs)}")

        # ── Graph stats: navigation context, not subject ──
        _skip_exact   = {"TART", "WAQF", "FASILA", "FASILA_CROSS", "JUZ", "SAJDAH", "RUKU"}
        _skip_pfx     = ("SYN_", "MORPH_")
        if fc:
            top = [(f, n) for f, n in sorted(fc.items(), key=lambda x: -x[1])
                   if f.upper() not in _skip_exact
                   and not any(f.upper().startswith(p) for p in _skip_pfx)][:4]
            if top:
                lines.append("Families: " + ", ".join(f"{f}({n})" for f, n in top))

        if onto:
            lines.append("Maqasid: " + ", ".join(onto[:4]))

        addr_labels = {1: "singular", 2: "dual", 3: "plural", 4: "majestic"}
        if addr:
            lines.append("Address: " + ", ".join(addr_labels.get(a, str(a)) for a in addr))

        if intensity is not None:
            lines.append(f"Intensity: {intensity:.2f}")

        return "\n".join(lines)
