import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient


class _FakeConstitution:
    def __init__(self):
        self.approved = []
        self.rejected = []

    def pending_proposals(self):
        return []

    def approved_proposals(self):
        return []

    def approve(self, pid, approved_by):
        self.approved.append((pid, approved_by))

    def reject(self, pid, reason=""):
        self.rejected.append((pid, reason))


class _FakeEngine:
    def __init__(self):
        self.constitution = _FakeConstitution()
        self.wipes = 0
        self.hifz_starts = []
        self.hadith_starts = []
        self._recent_steps = []
        self._shahid_memory = None

    def start_hifz(self, restart=True):
        self.hifz_starts.append(restart)
        return True

    def hifz_status(self):
        return {"active": False}

    def wipe_memory(self):
        self.wipes += 1

    def start_hadith_hifz(self, restart=False):
        self.hadith_starts.append(restart)
        return True

    def hadith_hifz_status(self):
        return {"active": False}

    def get_memories(self):
        return []

    def get_status(self):
        return {"ok": True}

    def subscribe(self):
        return iter(())

    def chat(self, message):
        return message


def _client(monkeypatch, token=None):
    if token is None:
        monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", token)

    from server import create_app

    engine = _FakeEngine()
    return TestClient(create_app(engine)), engine


def test_launch_and_engine_defaults_point_to_shipped_files():
    import engine
    import launch

    assert Path(launch.TMQ_PATH).is_file()
    assert Path(launch.TTL_PATH).is_file()
    assert Path(engine.DEFAULT_TMQ_PATH).is_file()
    assert Path(engine.DEFAULT_TTL_PATH).is_file()


def test_qusai_middleware_imports_and_instantiates_without_deleted_loader():
    from qusai_core.pipeline.middleware import QusaiMiddleware

    middleware = QusaiMiddleware(lazy_load=True)

    assert middleware.ontology is not None
    assert middleware.validator is not None
    assert middleware.model is not None


def test_admin_routes_fail_closed_when_token_not_configured(monkeypatch):
    client, engine = _client(monkeypatch)

    response = client.post("/hifz/wipe")

    assert response.status_code == 403
    assert engine.wipes == 0


def test_admin_routes_reject_missing_token(monkeypatch):
    client, engine = _client(monkeypatch, token="secret-token")

    response = client.post("/hifz/start", json={"restart": True})

    assert response.status_code == 403
    assert engine.hifz_starts == []


def test_admin_routes_accept_bearer_token(monkeypatch):
    client, engine = _client(monkeypatch, token="secret-token")

    response = client.post(
        "/hifz/wipe",
        headers={"authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert engine.wipes == 1


def test_constitution_approval_requires_admin_token(monkeypatch):
    client, engine = _client(monkeypatch, token="secret-token")

    blocked = client.post("/constitution/approve", json={"id": "proposal-1"})
    allowed = client.post(
        "/constitution/approve",
        json={"id": "proposal-1"},
        headers={"x-ikhtiyar-admin-token": "secret-token"},
    )

    assert blocked.status_code == 403
    assert allowed.status_code == 200
    assert engine.constitution.approved == [("proposal-1", "Qalam")]
