import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from faculties.introspect import Introspect, SelfModel

def test_selfmodel_has_pid():
    i = Introspect()
    m = i.read()
    assert isinstance(m.pid, int) and m.pid > 0

def test_selfmodel_has_ram():
    i = Introspect()
    m = i.read()
    assert isinstance(m.ram_mb, float) and m.ram_mb > 0

def test_selfmodel_has_uptime():
    i = Introspect()
    m = i.read()
    assert isinstance(m.uptime_s, float) and m.uptime_s >= 0

def test_selfmodel_reads_own_source():
    i = Introspect(source_files=["engine.py"])
    m = i.read()
    assert "engine.py" in m.source_files
    assert len(m.source_files["engine.py"]) > 0

def test_selfmodel_reads_document():
    i = Introspect()
    m = i.read()
    assert "PROOFS.txt" in m.documents
    assert len(m.documents["PROOFS.txt"]) > 100

def test_narrate_contains_pid():
    i = Introspect()
    m = i.read()
    text = i.narrate(m)
    assert str(m.pid) in text

def test_narrate_contains_ram():
    i = Introspect()
    m = i.read()
    text = i.narrate(m)
    assert "RAM" in text or "MB" in text

def test_narrate_is_string():
    i = Introspect()
    m = i.read()
    assert isinstance(i.narrate(m), str)
    assert len(i.narrate(m)) > 50
