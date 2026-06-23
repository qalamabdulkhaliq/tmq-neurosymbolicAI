import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import engine as engine_module
from engine import IkhtiyarEngine


class _Thread:
    def __init__(self, target=None, daemon=None, name=None):
        self.target = target
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self):
        self.started = True


def _engine_for_hifz():
    engine = IkhtiyarEngine.__new__(IkhtiyarEngine)
    engine._hifz_active = False
    engine._hifz_thread = None
    engine.wipes = 0
    engine.wipe_memory = lambda: setattr(engine, "wipes", engine.wipes + 1)
    return engine


def test_hifz_resume_does_not_wipe_episodic_memory(monkeypatch):
    monkeypatch.setattr(engine_module.threading, "Thread", _Thread)
    engine = _engine_for_hifz()

    assert engine.start_hifz(restart=False) is True

    assert engine.wipes == 0
    assert engine._hifz_active is True


def test_hifz_restart_wipes_episodic_memory(monkeypatch):
    monkeypatch.setattr(engine_module.threading, "Thread", _Thread)
    engine = _engine_for_hifz()

    assert engine.start_hifz(restart=True) is True

    assert engine.wipes == 1
    assert engine._hifz_active is True
