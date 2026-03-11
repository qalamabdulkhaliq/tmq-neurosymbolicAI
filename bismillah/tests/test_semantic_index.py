import pytest
from faculties.semantic_index import SpectralIndex
from faculties.tmq_corpus import TMQCorpus

TMQ_PATH = "TMQ_v10_hypermodal_enriched.json"

@pytest.fixture(scope="module")
def tmq():
    return TMQCorpus(TMQ_PATH)

@pytest.fixture
def index(tmp_path):
    return SpectralIndex(persist_dir=str(tmp_path / "chroma"))

def test_ingest_small_batch(index, tmq):
    # Ingest just MAQASID edges (402 edges — fast)
    index.ingest_tmq(tmq, families=["MAQASID"])
    count = index.count()
    assert count == 402

def test_search_by_eigenstate(index, tmq):
    index.ingest_tmq(tmq, families=["MAQASID"])
    results = index.search_by_eigenstate(0, n=5)
    assert len(results) <= 5
    for r in results:
        assert "family" in r["metadata"]

def test_find_modal_neighbors(index, tmq):
    index.ingest_tmq(tmq, families=["MAQASID"])
    edges = tmq.edges_by_family("MAQASID")
    if edges:
        results = index.find_modal_neighbors(edges[0], threshold=0.5, n=5)
        assert isinstance(results, list)
