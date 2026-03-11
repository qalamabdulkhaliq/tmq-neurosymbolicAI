# tests/test_graph_memory.py
# Requires Neo4j running at neo4j://127.0.0.1:7687
import pytest
from faculties.graph_memory import ShahidGraph
from faculties.shahid_clock import ShahidClock
from faculties.provenance import ProvenanceRecord

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "lhwlqibswb")

@pytest.fixture
def graph():
    g = ShahidGraph(NEO4J_URI, NEO4J_AUTH)
    g.connect()
    g.clear_test_data()
    yield g
    g.clear_test_data()
    g.close()

def test_connect(graph):
    assert graph.connected

def test_add_and_retrieve_thought(graph):
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    m = clock.now()
    node_id = graph.add_thought("Bismillah test thought", mode="HAQQ", moment=m, label="TestNode")
    found = graph.get_thought(node_id)
    assert found["text"] == "Bismillah test thought"
    assert found["mode"] == "HAQQ"

def test_add_edge_with_provenance(graph):
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    m = clock.now()
    a = graph.add_thought("node A", mode="HAQQ", moment=m, label="TestNode")
    b = graph.add_thought("node B", mode="HAQQ", moment=m, label="TestNode")
    rec = ProvenanceRecord("inference", "test", "sess-1", m, 0.8)
    graph.add_edge(a, b, "INFERRED", rec)
    edges = graph.get_edges(a)
    assert len(edges) == 1
    assert edges[0]["prov_confidence"] == 0.8
