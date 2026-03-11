"""
faculties/graph_memory.py

F1 — Neo4j graph memory for Shahid.

ShahidGraph wraps the Neo4j driver and provides:
  - add_thought()       — persist a validated thought node
  - get_thought()       — retrieve by element-id
  - add_edge()          — create a typed relationship with provenance
  - get_edges()         — list outgoing relationships
  - clear_test_data()   — wipe :TestNode nodes (test isolation)
  - seed_from_tmq()     — one-time import of TMQ corpus as :TMQNode nodes
  - find_unlinked_adjacent_eigenstates() — gap query for Gap Daemon (F4)
"""

from neo4j import GraphDatabase
from faculties.shahid_clock import Moment
from faculties.provenance import ProvenanceRecord, attach


class ShahidGraph:
    def __init__(self, uri: str, auth: tuple):
        self._uri = uri
        self._auth = auth
        self._driver = None
        self.connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Open the driver and verify connectivity (raises on failure)."""
        self._driver = GraphDatabase.driver(self._uri, auth=self._auth)
        self._driver.verify_connectivity()
        self.connected = True

    def close(self):
        """Close the driver connection pool."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self.connected = False

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def clear_test_data(self):
        """Delete all :TestNode nodes (used in test fixtures only)."""
        with self._driver.session() as s:
            s.run("MATCH (n:TestNode) DETACH DELETE n")

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_thought(
        self,
        text: str,
        mode: str,
        moment: Moment,
        label: str = "Thought",
    ) -> str:
        """
        Create a node labelled :<label>:Thought and return its element-id.

        Parameters
        ----------
        text    : the thought content
        mode    : HAQQ | QIYAS | SILENCE
        moment  : Moment from ShahidClock.now()
        label   : additional label (default 'Thought'); use 'TestNode' in tests
        """
        props = {
            "text": text,
            "mode": mode,
            "utc_iso": moment.utc_iso,
            "hours_online": moment.hours_online,
        }
        with self._driver.session() as s:
            result = s.run(
                f"CREATE (n:{label}:Thought $props) RETURN elementId(n) AS eid",
                props=props,
            )
            return result.single()["eid"]

    def get_thought(self, node_id: str) -> dict:
        """
        Return properties of the node with the given element-id.
        Returns {} if no node found.
        """
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n) WHERE elementId(n) = $eid RETURN properties(n) AS props",
                eid=node_id,
            )
            rec = result.single()
            return dict(rec["props"]) if rec else {}

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        prov: ProvenanceRecord,
        extra_props: dict = None,
    ):
        """
        Create a typed relationship from from_id to to_id with provenance
        fields merged into the relationship properties.
        """
        edge_props = attach(extra_props or {}, prov)
        with self._driver.session() as s:
            s.run(
                f"""MATCH (a) WHERE elementId(a) = $a_id
                    MATCH (b) WHERE elementId(b) = $b_id
                    CREATE (a)-[r:{rel_type} $props]->(b)""",
                a_id=from_id,
                b_id=to_id,
                props=edge_props,
            )

    def get_edges(self, from_id: str) -> list:
        """
        Return a list of property dicts for all outgoing relationships
        from the node with element-id = from_id.
        """
        with self._driver.session() as s:
            result = s.run(
                """MATCH (a)-[r]->(b) WHERE elementId(a) = $eid
                   RETURN properties(r) AS props""",
                eid=from_id,
            )
            return [dict(row["props"]) for row in result]

    # ------------------------------------------------------------------
    # TMQ seeding (one-time corpus import)
    # ------------------------------------------------------------------

    def seed_from_tmq(self, tmq, batch_size: int = 500):
        """
        One-time import of TMQ node_registry as read-only :TMQNode nodes.
        Safe to call multiple times — uses MERGE to avoid duplicates.

        Parameters
        ----------
        tmq        : a TMQCorpus instance (must expose ._node_registry)
        batch_size : how many nodes to MERGE per transaction
        """
        nodes = list(tmq._node_registry.items())
        with self._driver.session() as s:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                params = [
                    {"tmq_id": k, "tier": v.get("tier", "?")}
                    for k, v in batch
                    if isinstance(v, dict)
                ]
                if not params:
                    continue
                s.run(
                    """UNWIND $rows AS row
                       MERGE (n:TMQNode {tmq_id: row.tmq_id})
                       SET n.tier = row.tier""",
                    rows=params,
                )

    # ------------------------------------------------------------------
    # Gap Daemon query (F4)
    # ------------------------------------------------------------------

    def find_unlinked_adjacent_eigenstates(self, limit: int = 50) -> list:
        """
        Find TMQNode pairs whose eigenstates differ by exactly 1 but
        have no edge between them — candidates for the Gap Daemon.

        Returns a list of dicts with keys: a_id, b_id, ea, eb.
        """
        with self._driver.session() as s:
            result = s.run(
                """MATCH (a:TMQNode), (b:TMQNode)
                   WHERE a.eigenstate IS NOT NULL AND b.eigenstate IS NOT NULL
                     AND abs(a.eigenstate - b.eigenstate) = 1
                     AND NOT (a)--(b)
                     AND a.tmq_id < b.tmq_id
                   RETURN a.tmq_id AS a_id, b.tmq_id AS b_id,
                          a.eigenstate AS ea, b.eigenstate AS eb
                   LIMIT $lim""",
                lim=limit,
            )
            return [dict(r) for r in result]
