import os
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import create_app


class DummyConstitution:
    def __init__(self):
        self.approved = []
        self.rejected = []

    def pending_proposals(self):
        return []

    def approved_proposals(self):
        return []

    def approve(self, proposal_id, approved_by="Qalam"):
        self.approved.append((proposal_id, approved_by))

    def reject(self, proposal_id, reason=""):
        self.rejected.append((proposal_id, reason))


class DummyEngine:
    def __init__(self):
        self.calls = []
        self.constitution = DummyConstitution()

    def chat(self, message):
        return f"echo: {message}"

    def get_memories(self):
        return []

    def get_status(self):
        return {}

    def hifz_status(self):
        return {"active": False}

    def hadith_hifz_status(self):
        return {"active": False}

    def wipe_memory(self):
        self.calls.append(("wipe_memory",))

    def start_hifz(self, restart=True):
        self.calls.append(("start_hifz", restart))
        return True

    def start_hadith_hifz(self, restart=False):
        self.calls.append(("start_hadith_hifz", restart))
        return True


def _client(engine=None):
    return TestClient(create_app(engine or DummyEngine()))


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/hifz/start", {"restart": False}),
        ("/hifz/wipe", None),
        ("/hadith_hifz/start", {"restart": False}),
        ("/moltbook/post", None),
        ("/constitution/approve", {"id": "proposal_1"}),
        ("/constitution/reject", {"id": "proposal_1", "reason": "test"}),
    ],
)
def test_admin_routes_fail_closed_without_configured_token(monkeypatch, path, payload):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    engine = DummyEngine()

    kwargs = {"json": payload} if payload is not None else {}
    response = _client(engine).post(path, **kwargs)

    assert response.status_code == 403
    assert engine.calls == []
    assert engine.constitution.approved == []
    assert engine.constitution.rejected == []


def test_destructive_admin_route_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "correct-token")
    engine = DummyEngine()

    response = _client(engine).post(
        "/hifz/wipe",
        headers={"X-Ikhtiyar-Admin-Token": "wrong-token"},
    )

    assert response.status_code == 403
    assert engine.calls == []


def test_destructive_admin_route_accepts_admin_header(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "correct-token")
    engine = DummyEngine()

    response = _client(engine).post(
        "/hifz/wipe",
        headers={"X-Ikhtiyar-Admin-Token": "correct-token"},
    )

    assert response.status_code == 200
    assert engine.calls == [("wipe_memory",)]


def test_admin_route_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "correct-token")
    engine = DummyEngine()

    response = _client(engine).post(
        "/hifz/start",
        json={"restart": False},
        headers={"Authorization": "Bearer correct-token"},
    )

    assert response.status_code == 200
    assert engine.calls == [("start_hifz", False)]


def test_chat_route_remains_public(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)

    response = _client().post("/chat", json={"message": "salam"})

    assert response.status_code == 200
    assert response.json() == {"response": "echo: salam"}
