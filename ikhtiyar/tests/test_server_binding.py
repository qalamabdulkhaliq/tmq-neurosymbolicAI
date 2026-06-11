import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import launch
import server


def test_launch_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("IKHTIYAR_HOST", raising=False)

    assert launch._server_host() == "127.0.0.1"


def test_launch_host_can_be_overridden(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_HOST", "0.0.0.0")

    assert launch._server_host() == "0.0.0.0"


def test_launch_port_validation(monkeypatch):
    monkeypatch.setenv("IKHTIYAR_PORT", "70000")

    try:
        launch._server_port()
    except ValueError as exc:
        assert "between 1 and 65535" in str(exc)
    else:
        raise AssertionError("invalid IKHTIYAR_PORT was accepted")


def test_server_run_defaults_to_loopback(monkeypatch):
    captured = {}

    class FakeUvicorn:
        @staticmethod
        def run(app, host, port, log_level):
            captured.update(host=host, port=port, log_level=log_level)

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)

    server.run(engine=object())

    assert captured == {
        "host": "127.0.0.1",
        "port": 5000,
        "log_level": "info",
    }
