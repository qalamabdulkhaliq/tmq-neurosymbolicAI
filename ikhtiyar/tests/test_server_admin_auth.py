import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from server import create_app


class _DummyConstitution:
    def __init__(self):
        self.approved = []
        self.rejected = []

    def pending_proposals(self):
        return []

    def approved_proposals(self):
        return []

    def approve(self, pid, approved_by=""):
        self.approved.append((pid, approved_by))

    def reject(self, pid, reason=""):
        self.rejected.append((pid, reason))


class _DummyEngine:
    def __init__(self):
        self.wiped = False
        self.hifz_started = False
        self.hadith_started = False
        self.constitution = _DummyConstitution()
        self._recent_steps = []
        self._shahid_memory = None

    def chat(self, message):
        return f"reply: {message}"

    def get_memories(self):
        return []

    def get_status(self):
        return {}

    def subscribe(self):
        return iter(())

    def start_hifz(self, restart=True):
        self.hifz_started = restart
        return True

    def hifz_status(self):
        return {}

    def wipe_memory(self):
        self.wiped = True

    def start_hadith_hifz(self, restart=False):
        self.hadith_started = restart
        return True

    def hadith_hifz_status(self):
        return {}


def _client():
    engine = _DummyEngine()
    return TestClient(create_app(engine)), engine


def test_admin_routes_fail_closed_without_configured_token(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    client, engine = _client()

    response = client.post("/hifz/wipe")

    assert response.status_code == 503
    assert not engine.wiped


def test_admin_routes_reject_missing_or_invalid_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    client, engine = _client()

    assert client.post("/hifz/wipe").status_code == 403
    assert client.post("/hifz/wipe", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert not engine.wiped


def test_admin_route_accepts_valid_x_admin_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    client, engine = _client()

    response = client.post("/hifz/wipe", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    assert engine.wiped


def test_admin_route_accepts_valid_bearer_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    client, engine = _client()

    response = client.post(
        "/constitution/approve",
        json={"id": "proposal-1"},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert engine.constitution.approved == [("proposal-1", "Qalam")]


def test_public_read_routes_do_not_require_admin_token(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    client, _ = _client()

    response = client.get("/status")

    assert response.status_code == 200
