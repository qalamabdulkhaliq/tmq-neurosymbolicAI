import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import create_app


class _Constitution:
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


class _Engine:
    def __init__(self):
        self.calls = []
        self.constitution = _Constitution()
        self._recent_steps = []
        self._shahid_memory = None

    def chat(self, message):
        return f"reply: {message}"

    def get_memories(self):
        return []

    def get_status(self):
        return {"ok": True}

    def start_hifz(self, restart=True):
        self.calls.append(("start_hifz", restart))
        return True

    def hifz_status(self):
        return {"active": False}

    def wipe_memory(self):
        self.calls.append(("wipe_memory",))

    def start_hadith_hifz(self, restart=False):
        self.calls.append(("start_hadith_hifz", restart))
        return True

    def hadith_hifz_status(self):
        return {"active": False}


def _client(engine=None):
    return TestClient(create_app(engine or _Engine()))


def test_admin_endpoints_blocked_when_token_unconfigured(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    engine = _Engine()
    client = _client(engine)

    assert client.post("/hifz/wipe").status_code == 403
    assert client.post("/hifz/start", json={"restart": True}).status_code == 403
    assert client.post("/hadith_hifz/start", json={"restart": True}).status_code == 403
    assert client.post("/constitution/approve", json={"id": "p1"}).status_code == 403
    assert client.post("/constitution/reject", json={"id": "p1"}).status_code == 403

    assert engine.calls == []
    assert engine.constitution.approved == []
    assert engine.constitution.rejected == []


def test_admin_endpoints_reject_missing_or_wrong_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    engine = _Engine()
    client = _client(engine)

    assert client.post("/hifz/wipe").status_code == 403
    assert client.post(
        "/hifz/wipe",
        headers={"Authorization": "Bearer wrong"},
    ).status_code == 403

    assert engine.calls == []


def test_admin_token_allows_state_mutations(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret")
    engine = _Engine()
    client = _client(engine)
    headers = {"Authorization": "Bearer secret"}

    assert client.post("/hifz/wipe", headers=headers).status_code == 200
    assert client.post(
        "/hifz/start",
        json={"restart": False},
        headers=headers,
    ).status_code == 200
    assert client.post(
        "/hadith_hifz/start",
        json={"restart": True},
        headers=headers,
    ).status_code == 200
    assert client.post(
        "/constitution/approve",
        json={"id": "p1"},
        headers=headers,
    ).status_code == 200
    assert client.post(
        "/constitution/reject",
        json={"id": "p2", "reason": "not grounded"},
        headers=headers,
    ).status_code == 200

    assert engine.calls == [
        ("wipe_memory",),
        ("start_hifz", False),
        ("start_hadith_hifz", True),
    ]
    assert engine.constitution.approved == [("p1", "Qalam")]
    assert engine.constitution.rejected == [("p2", "not grounded")]


def test_public_routes_do_not_require_admin_token(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    client = _client()

    assert client.get("/status").status_code == 200
    assert client.post("/chat", json={"message": "salam"}).status_code == 200
