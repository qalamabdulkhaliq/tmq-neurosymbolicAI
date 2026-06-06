import importlib
import os
import sys


IKHTIYAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.dirname(IKHTIYAR_DIR)
sys.path.insert(0, IKHTIYAR_DIR)


def _load_launch(monkeypatch):
    monkeypatch.delenv("TMQ_PATH", raising=False)
    monkeypatch.delenv("TTL_PATH", raising=False)
    sys.modules.pop("launch", None)
    return importlib.import_module("launch")


def test_launch_defaults_point_to_bundled_data(monkeypatch):
    launch = _load_launch(monkeypatch)

    assert launch.TMQ_PATH == os.path.join(IKHTIYAR_DIR, "TMQ_v12.json")
    assert launch.TTL_PATH == os.path.join(PROJECT_DIR, "quran_root_ontology_v3.ttl")
    assert os.path.exists(launch.TMQ_PATH)
    assert os.path.exists(launch.TTL_PATH)


def test_preflight_passes_with_bundled_data_and_ollama_reachable(monkeypatch):
    launch = _load_launch(monkeypatch)
    monkeypatch.setattr(
        launch,
        "_check_socket",
        lambda host, port, timeout=2.0: port == 11434,
    )

    assert launch.preflight() is True
