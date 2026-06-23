import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import create_app


class _Engine:
    def __init__(self):
        self.wiped = False
        self.started = []
        self.constitution = None

    def subscribe(self):
        return iter(())

    def chat(self, message):
        return "ok"

    def get_memories(self):
        return []

    def get_status(self):
        return {}

    def hifz_status(self):
        return {}

    def hadith_hifz_status(self):
        return {}

    def wipe_memory(self):
        self.wiped = True

    def start_hifz(self, restart=True):
        self.started.append(("hifz", restart))
        return True

    def start_hadith_hifz(self, restart=False):
        self.started.append(("hadith", restart))
        return True


def test_admin_mutation_fails_closed_without_configured_token(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SHAHID_ADMIN_TOKEN", raising=False)
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post("/hifz/wipe")

    assert response.status_code == 503
    assert engine.wiped is False


def test_admin_mutation_requires_valid_token(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = _Engine()
    client = TestClient(create_app(engine))

    missing = client.post("/hifz/wipe")
    wrong = client.post("/hifz/wipe", headers={"x-admin-token": "wrong"})
    ok = client.post("/hifz/wipe", headers={"x-admin-token": "secret-token"})

    assert missing.status_code == 401
    assert wrong.status_code == 403
    assert ok.status_code == 200
    assert engine.wiped is True


def test_admin_bearer_token_allows_hifz_start(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_ADMIN_TOKEN", "secret-token")
    engine = _Engine()
    client = TestClient(create_app(engine))

    response = client.post(
        "/hifz/start",
        json={"restart": False},
        headers={"authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert engine.started == [("hifz", False)]
