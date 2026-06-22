import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine as engine_module
from engine import IkhtiyarEngine


class _ThreadStub:
    def __init__(self, target=None, daemon=None, name=None):
        self.target = target
        self.daemon = daemon
        self.name = name

    def start(self):
        pass


def _new_engine(monkeypatch):
    monkeypatch.setattr(engine_module.threading, "Thread", _ThreadStub)
    engine = IkhtiyarEngine(tmq_path="", hvt_path="", ttl_path="")
    engine._hifz_active = False
    engine._hifz_thread = None
    calls = []
    engine.wipe_memory = lambda: calls.append("wipe")
    return engine, calls


def test_start_hifz_resume_preserves_episodic_memory(monkeypatch):
    engine, calls = _new_engine(monkeypatch)

    assert engine.start_hifz(restart=False) is True

    assert calls == []
    assert engine._hifz_active is True


def test_start_hifz_restart_wipes_episodic_memory(monkeypatch):
    engine, calls = _new_engine(monkeypatch)

    assert engine.start_hifz(restart=True) is True

    assert calls == ["wipe"]
    assert engine._hifz_active is True
