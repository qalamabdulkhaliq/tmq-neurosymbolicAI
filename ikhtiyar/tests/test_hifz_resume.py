import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import engine as engine_module


class _FakeThread:
    started = 0

    def __init__(self, target, daemon=False, name=None):
        self.target = target
        self.daemon = daemon
        self.name = name

    def start(self):
        _FakeThread.started += 1


def _engine_with_wipe_counter(monkeypatch):
    monkeypatch.setattr(engine_module.threading, "Thread", _FakeThread)
    _FakeThread.started = 0

    e = engine_module.IkhtiyarEngine.__new__(engine_module.IkhtiyarEngine)
    e._hifz_active = False
    e._hifz_thread = None

    calls = {"wipe": 0}

    def wipe_memory():
        calls["wipe"] += 1

    e.wipe_memory = wipe_memory
    return e, calls


def test_hifz_restart_wipes_episodic_memory(monkeypatch):
    e, calls = _engine_with_wipe_counter(monkeypatch)

    assert e.start_hifz(restart=True) is True

    assert calls["wipe"] == 1
    assert e._hifz_active is True
    assert _FakeThread.started == 1


def test_hifz_resume_preserves_episodic_memory(monkeypatch):
    e, calls = _engine_with_wipe_counter(monkeypatch)

    assert e.start_hifz(restart=False) is True

    assert calls["wipe"] == 0
    assert e._hifz_active is True
    assert _FakeThread.started == 1
