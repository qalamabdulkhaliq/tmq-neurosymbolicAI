import os
import pytest
from faculties.reasoner import OWLReasoner

TTL_PATH = "QUS-AI HF/quran_root_ontology_v3.ttl"


@pytest.fixture(scope="module")
def reasoner():
    r = OWLReasoner(TTL_PATH)
    r.load()
    return r


def test_loads_without_error(reasoner):
    assert reasoner.loaded


def test_consistent_triple_passes(reasoner):
    ok, violations = reasoner.check_triple(
        "http://ontology.quran/surah0_SOURCE",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#Thing",
    )
    assert ok
    assert violations == ()


def test_aseity_claim_is_blocked(reasoner):
    ok, violations = reasoner.check_aseity_claim("shahid_local_instance")
    assert not ok
    assert len(violations) > 0


def test_source_aseity_is_allowed(reasoner):
    ok, violations = reasoner.check_aseity_claim(
        "http://ontology.quran/surah0_SOURCE"
    )
    assert ok


def test_rejection_logged(tmp_path, reasoner):
    log_path = str(tmp_path / "test_esc.json")
    ok, violations = reasoner.check_aseity_claim("bad_actor")
    reasoner.log_rejection(("bad_actor", "rdf:type", "NecessaryBeing"), violations, log_path)
    import json
    with open(log_path) as f:
        entries = json.load(f)
    assert len(entries) == 1
    assert entries[0]["type"] == "owl_rejection"
