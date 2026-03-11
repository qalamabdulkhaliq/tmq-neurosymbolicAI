from faculties.tmq_corpus import TMQCorpus

TMQ_PATH = "TMQ_v10_hypermodal_enriched.json"

def test_loads_without_error():
    tmq = TMQCorpus(TMQ_PATH)
    assert tmq.total_edges > 0
    assert tmq.total_nodes > 0

def test_stats():
    tmq = TMQCorpus(TMQ_PATH)
    assert tmq.total_edges == 45092
    assert tmq.total_nodes == 134629

def test_nodes_by_tier():
    tmq = TMQCorpus(TMQ_PATH)
    surahs = tmq.nodes_by_tier("surah")
    assert len(surahs) == 114

def test_edges_by_family():
    tmq = TMQCorpus(TMQ_PATH)
    iltifat = tmq.edges_by_family("ILTIFAT")
    assert len(iltifat) == 10863

def test_edges_by_eigenstate():
    tmq = TMQCorpus(TMQ_PATH)
    state0 = tmq.edges_by_eigenstate(0)
    assert len(state0) == 16056

def test_edge_has_modal_fields():
    tmq = TMQCorpus(TMQ_PATH)
    edges = tmq.edges_by_family("TART")
    e = edges[0]
    modal = e["modal"]
    assert "intensity" in modal
    assert "dominant_eigenstate" in modal
    assert "nodes" in e
