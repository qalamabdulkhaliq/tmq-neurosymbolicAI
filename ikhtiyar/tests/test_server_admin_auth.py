import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import create_app


class _FakeConstitution:
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


class _FakeEngine:
    def __init__(self):
        self.wipes = 0
        self.hifz_starts = []
        self.hadith_starts = []
        self.constitution = _FakeConstitution()
        self._recent_steps = []

    def subscribe(self):
        return iter(())

    def chat(self, message):
        return f"echo: {message}"

    def get_memories(self):
        return []

    def get_status(self):
        return {}

    def wipe_memory(self):
        self.wipes += 1

    def start_hifz(self, restart=True):
        self.hifz_starts.append(restart)
        return True

    def hifz_status(self):
        return {}

    def start_hadith_hifz(self, restart=False):
        self.hadith_starts.append(restart)
        return True

    def hadith_hifz_status(self):
        return {}


def _client():
    engine = _FakeEngine()
    return TestClient(create_app(engine)), engine


_ADMIN_REQUESTS = [
    ("post", "/hifz/start", {"restart": True}),
    ("post", "/hifz/wipe", None),
    ("post", "/hadith_hifz/start", {"restart": False}),
    ("post", "/moltbook/post", None),
    ("post", "/constitution/approve", {"id": "p1"}),
    ("post", "/constitution/reject", {"id": "p1", "reason": "no"}),
]


def _request(client, method, path, body=None, headers=None):
    return getattr(client, method)(path, json=body, headers=headers or {})


def test_admin_routes_fail_closed_when_token_unconfigured(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    client, engine = _client()

    responses = [
        _request(client, method, path, body)
        for method, path, body in _ADMIN_REQUESTS
    ]

    assert all(response.status_code == 503 for response in responses)
    assert engine.wipes == 0
    assert engine.hifz_starts == []
    assert engine.hadith_starts == []
    assert engine.constitution.approved == []
    assert engine.constitution.rejected == []


def test_admin_routes_reject_missing_and_invalid_tokens(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "correct-token")
    client, engine = _client()

    missing = [
        _request(client, method, path, body)
        for method, path, body in _ADMIN_REQUESTS
    ]
    invalid = [
        _request(client, method, path, body, headers={"X-Admin-Token": "wrong"})
        for method, path, body in _ADMIN_REQUESTS
    ]

    assert all(response.status_code == 401 for response in missing)
    assert all(response.status_code == 403 for response in invalid)
    assert engine.wipes == 0
    assert engine.hifz_starts == []
    assert engine.hadith_starts == []
    assert engine.constitution.approved == []
    assert engine.constitution.rejected == []


def test_admin_routes_accept_x_admin_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "correct-token")
    client, engine = _client()

    response = client.post("/hifz/wipe", headers={"X-Admin-Token": "correct-token"})

    assert response.status_code == 200
    assert engine.wipes == 1


def test_admin_routes_accept_bearer_token_for_side_effects(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "correct-token")
    client, engine = _client()
    headers = {"Authorization": "Bearer correct-token"}

    hifz = client.post("/hifz/start", json={"restart": False}, headers=headers)
    hadith = client.post("/hadith_hifz/start", json={"restart": True}, headers=headers)
    approve = client.post("/constitution/approve", json={"id": "p1"}, headers=headers)
    reject = client.post(
        "/constitution/reject",
        json={"id": "p2", "reason": "not grounded"},
        headers=headers,
    )

    assert hifz.status_code == 200
    assert hadith.status_code == 200
    assert approve.status_code == 200
    assert reject.status_code == 200
    assert engine.hifz_starts == [False]
    assert engine.hadith_starts == [True]
    assert engine.constitution.approved == [("p1", "Qalam")]
    assert engine.constitution.rejected == [("p2", "not grounded")]
