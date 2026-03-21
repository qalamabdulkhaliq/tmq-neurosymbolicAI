"""
F4 Gap Daemon.
Scans every interval_minutes for spectral gaps in the TMQ hypergraph.
A gap = two nodes whose dominant_eigenstate values differ by exactly 1
but share no TMQ hyperedge.
Writes findings to shahid_escalations.json for Qalam to review.
"""
import json
import os
from apscheduler.schedulers.background import BackgroundScheduler
from faculties.shahid_clock import ShahidClock
from faculties.tmq_corpus import TMQCorpus
from faculties.semantic_index import SpectralIndex


class GapDaemon:
    def __init__(
        self,
        graph,
        index: SpectralIndex,
        tmq: TMQCorpus,
        clock: ShahidClock,
        escalation_path: str = "shahid_escalations.json",
        interval_minutes: int = 30,
    ):
        self._graph = graph
        self._index = index
        self._tmq = tmq
        self._clock = clock
        self._esc_path = escalation_path
        self._interval = interval_minutes
        self._scheduler = BackgroundScheduler()
        self._last_report = []
        self.running = False

    # Max nodes per edge to consider for adjacency — skip huge structural edges
    _MAX_EDGE_NODES = 20

    def scan_spectral_gaps(self) -> list:
        """Find eigenstate-adjacent node pairs with no TMQ hyperedge between them.

        Uses frozenset pairs for O(1) lookup. Skips edges with > _MAX_EDGE_NODES
        nodes to avoid O(k^2) explosion on large structural hyperedges.
        """
        moment = self._clock.now()

        # Single pass: build connected pairs AND eigenstate membership
        connected_pairs: set = set()  # frozenset({nodeA, nodeB}) that share an edge
        by_state: dict = {}

        for edge in self._tmq.all_edges():
            if not isinstance(edge, dict):
                continue
            nodes = edge.get("nodes") or []
            modal = edge.get("modal") or {}
            es = modal.get("dominant_eigenstate")

            if es is not None:
                for n in nodes:
                    by_state.setdefault(es, set()).add(n)

            # Only build pairs for small edges — large structural edges are
            # connectivity-dense and won't represent meaningful gaps anyway
            if len(nodes) <= self._MAX_EDGE_NODES:
                for i, na in enumerate(nodes):
                    for nb in nodes[i + 1:]:
                        connected_pairs.add(frozenset((na, nb)))

        gaps = []
        seen_pairs: set = set()
        states = sorted(by_state.keys())

        for i in range(len(states) - 1):
            a, b = states[i], states[i + 1]
            if b - a != 1:
                continue
            nodes_a = list(by_state[a])[:20]
            nodes_b = list(by_state[b])[:20]
            for na in nodes_a:
                for nb in nodes_b:
                    key = frozenset((na, nb))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    if key not in connected_pairs:
                        gaps.append({
                            "utc_iso": moment.utc_iso,
                            "hours_online": moment.hours_online,
                            "type": "spectral_gap",
                            "eigenstate_a": a,
                            "eigenstate_b": b,
                            "node_a": na,
                            "node_b": nb,
                        })
            if len(gaps) >= 50:
                break

        self._last_report = gaps
        self._append_to_escalations(gaps)
        return gaps

    def _append_to_escalations(self, gaps: list):
        existing = []
        if os.path.exists(self._esc_path):
            with open(self._esc_path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.extend(gaps)
        with open(self._esc_path, "w") as f:
            json.dump(existing, f, indent=2)

    def last_report(self) -> list:
        return self._last_report

    def start(self):
        self._scheduler.add_job(
            self.scan_spectral_gaps,
            "interval",
            minutes=self._interval,
            id="gap_scan",
        )
        self._scheduler.start()
        self.running = True

    def stop(self):
        self._scheduler.shutdown(wait=False)
        self.running = False
