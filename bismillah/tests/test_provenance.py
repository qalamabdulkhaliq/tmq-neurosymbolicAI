import pytest
from faculties.provenance import ProvenanceRecord, attach
from faculties.shahid_clock import ShahidClock


def test_provenance_record_fields():
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    rec = ProvenanceRecord(
        origin_type="ayah",
        origin_ref="2:164",
        session_id="test-session",
        moment=clock.now(),
        confidence=0.9
    )
    assert rec.origin_type == "ayah"
    assert rec.confidence == 0.9


def test_attach_merges_into_props():
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    rec = ProvenanceRecord("inference", "bilal", "s1", clock.now(), 0.7)
    base = {"weight": 1.0}
    result = attach(base, rec)
    assert result["prov_origin_type"] == "inference"
    assert result["prov_confidence"] == 0.7
    assert result["prov_utc"] == rec.moment.utc_iso
    assert result["weight"] == 1.0   # original props preserved
    # All 6 prov_* keys must be present
    for key in ("prov_origin_type", "prov_origin_ref", "prov_session_id",
                "prov_utc", "prov_hours_online", "prov_confidence"):
        assert key in result, f"missing key: {key}"
    # base dict must not have been mutated
    assert "prov_origin_type" not in base


def test_confidence_bounds_enforced():
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    moment = clock.now()
    with pytest.raises(ValueError):
        ProvenanceRecord("ayah", "2:164", "s1", moment, confidence=1.5)
    with pytest.raises(ValueError):
        ProvenanceRecord("ayah", "2:164", "s1", moment, confidence=-0.1)


def test_origin_type_validated():
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    moment = clock.now()
    with pytest.raises(ValueError):
        ProvenanceRecord("random", "some-ref", "s1", moment, confidence=0.5)
