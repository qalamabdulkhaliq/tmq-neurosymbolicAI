import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import create_app


class FakeConstitution:
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


class FakeEngine:
    def __init__(self):
        self.constitution = FakeConstitution()
        self._recent_steps = []
        self.wipe_count = 0
        self.hifz_starts = []
        self.hadith_hifz_starts = []

    def subscribe(self):
        return iter(())

    def chat(self, message):
        return f"reply: {message}"

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
        self.wipe_count += 1

    def start_hadith_hifz(self, restart=False):
        self.hadith_hifz_starts.append(restart)
        return True

    def hadith_hifz_status(self):
        return {}


def _client(engine=None):
    return TestClient(create_app(engine or FakeEngine()))


def test_mutating_routes_fail_closed_without_admin_token(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    engine = FakeEngine()

    response = _client(engine).post("/hifz/wipe")

    assert response.status_code == 503
    assert engine.wipe_count == 0


def test_mutating_routes_reject_missing_admin_header(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = FakeEngine()

    response = _client(engine).post("/constitution/approve", json={"id": "proposal-1"})

    assert response.status_code == 401
    assert engine.constitution.approved == []


def test_mutating_routes_accept_bearer_admin_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = FakeEngine()

    response = _client(engine).post(
        "/constitution/approve",
        json={"id": "proposal-1"},
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert engine.constitution.approved == [("proposal-1", "Qalam")]


def test_mutating_routes_accept_x_admin_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = FakeEngine()

    response = _client(engine).post(
        "/hifz/wipe",
        headers={"X-Admin-Token": "secret-token"},
    )

    assert response.status_code == 200
    assert engine.wipe_count == 1
