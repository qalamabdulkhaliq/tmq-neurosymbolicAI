"""
SPARQL endpoint tests.
Note: loads the full 20MB ontology TTL at startup, may take 30-60s.
"""
import time
import requests
import pytest
from faculties.sparql_endpoint import SPARQLServer

TTL_PATH = "../quran_root_ontology_v3.ttl"
PORT = 5821


@pytest.fixture(scope="module")
def server():
    s = SPARQLServer(TTL_PATH, port=PORT)
    s.start()
    time.sleep(3)
    yield s
    s.stop()


def test_sparql_select(server):
    query = "SELECT ?s WHERE { ?s a ?o } LIMIT 3"
    resp = requests.post(
        f"http://localhost:{PORT}/sparql",
        data={"query": query},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]["bindings"]) <= 3


def test_invalid_query_returns_400(server):
    resp = requests.post(
        f"http://localhost:{PORT}/sparql",
        data={"query": "NOT VALID SPARQL"},
    )
    assert resp.status_code == 400


def test_empty_query_returns_400(server):
    resp = requests.post(f"http://localhost:{PORT}/sparql", data={"query": ""})
    assert resp.status_code == 400
