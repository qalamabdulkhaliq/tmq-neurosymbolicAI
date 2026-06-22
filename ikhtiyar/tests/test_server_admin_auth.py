import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server import create_app


class _Constitution:
    def __init__(self):
        self.approved = []
        self.rejected = []

    def pending_proposals(self):
        return [{"id": "p1", "status": "pending"}]

    def approved_proposals(self):
        return []

    def approve(self, pid, approved_by=""):
        self.approved.append((pid, approved_by))

    def reject(self, pid, reason=""):
        self.rejected.append((pid, reason))


class _Engine:
    def __init__(self):
        self.wipe_count = 0
        self.hifz_starts = []
        self.hadith_starts = []
        self.hifz_active = False
        self.constitution = _Constitution()

    def start_hifz(self, restart=True):
        self.hifz_starts.append(restart)
        return True

    def hifz_status(self):
        return {"active": self.hifz_active}

    def wipe_memory(self):
        self.wipe_count += 1

    def start_hadith_hifz(self, restart=False):
        self.hadith_starts.append(restart)
        return True

    def hadith_hifz_status(self):
        return {"active": False}


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/hifz/start", {"restart": True}),
        ("/hifz/wipe", None),
        ("/hadith_hifz/start", {"restart": True}),
        ("/moltbook/post", None),
        ("/constitution/approve", {"id": "p1"}),
        ("/constitution/reject", {"id": "p1", "reason": "bad"}),
    ],
)
def test_admin_mutations_fail_closed_without_configured_token(monkeypatch, path, body):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post(path, json=body) if body is not None else client.post(path)

    assert response.status_code == 403
    assert engine.wipe_count == 0
    assert engine.hifz_starts == []
    assert engine.hadith_starts == []
    assert engine.constitution.approved == []
    assert engine.constitution.rejected == []


def test_admin_mutation_requires_matching_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = _Engine()
    client = TestClient(create_app(engine))

    missing = client.post("/hifz/wipe")
    assert missing.status_code == 401
    assert engine.wipe_count == 0

    ok = client.post("/hifz/wipe", headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200
    assert engine.wipe_count == 1


def test_hifz_resume_is_authorized_and_preserves_memory_message(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post(
        "/hifz/start",
        json={"restart": False},
        headers={"X-Ikhtiyar-Admin-Token": "secret-token"},
    )

    assert response.status_code == 200
    assert engine.hifz_starts == [False]
    assert "preserved" in response.json()["message"]
