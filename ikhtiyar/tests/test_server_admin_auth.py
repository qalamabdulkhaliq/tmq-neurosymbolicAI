import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import create_app


class _Constitution:
    def __init__(self):
        self.approved = []

    def pending_proposals(self):
        return []

    def approved_proposals(self):
        return []

    def approve(self, pid, approved_by):
        self.approved.append((pid, approved_by))

    def reject(self, pid, reason=""):
        pass


class _Engine:
    def __init__(self):
        self.constitution = _Constitution()
        self.wipes = 0
        self.hifz_starts = []
        self.hadith_starts = []
        self._recent_steps = []

    def subscribe(self):
        return iter(())

    def chat(self, message):
        return message

    def get_memories(self):
        return []

    def get_status(self):
        return {}

    def start_hifz(self, restart=True):
        self.hifz_starts.append(restart)
        return True

    def hifz_status(self):
        return {}

    def wipe_memory(self):
        self.wipes += 1

    def start_hadith_hifz(self, restart=False):
        self.hadith_starts.append(restart)
        return True

    def hadith_hifz_status(self):
        return {}


def test_destructive_endpoint_requires_configured_admin_token(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post("/hifz/wipe")

    assert response.status_code == 503
    assert engine.wipes == 0


def test_destructive_endpoint_rejects_missing_admin_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post("/hifz/wipe")

    assert response.status_code == 403
    assert engine.wipes == 0


def test_destructive_endpoint_accepts_admin_header(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post("/hifz/wipe", headers={"X-Ikhtiyar-Admin-Token": "secret"})

    assert response.status_code == 200
    assert engine.wipes == 1


def test_constitution_approval_requires_admin_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    engine = _Engine()
    client = TestClient(create_app(engine))

    blocked = client.post("/constitution/approve", json={"id": "proposal-1"})
    allowed = client.post(
        "/constitution/approve",
        json={"id": "proposal-1"},
        headers={"Authorization": "Bearer secret"},
    )

    assert blocked.status_code == 403
    assert allowed.status_code == 200
    assert engine.constitution.approved == [("proposal-1", "Qalam")]
