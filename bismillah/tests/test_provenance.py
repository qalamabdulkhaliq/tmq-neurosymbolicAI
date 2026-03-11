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
