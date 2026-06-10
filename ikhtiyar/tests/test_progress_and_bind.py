import importlib
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_hifz_progress_loader_ignores_ktbos_cursor_schema(tmp_path, monkeypatch):
    from core import hifz

    progress_file = tmp_path / "hifz_progress.json"
    _write_json(progress_file, {"surah": 4, "ayah": 12})
    monkeypatch.setattr(hifz, "_PROGRESS_FILE", str(progress_file))

    assert hifz._load_progress() == {
        "last_completed": 0,
        "total_tags": 0,
        "tags_by_type": {},
    }


def test_hadith_hifz_progress_loader_ignores_ktbos_cursor_schema(tmp_path, monkeypatch):
    from core import hadith_hifz

    progress_file = tmp_path / "hadith_hifz_progress.json"
    _write_json(progress_file, {"source": "Bukhari", "index": 64})
    monkeypatch.setattr(hadith_hifz, "_PROGRESS_FILE", str(progress_file))

    assert hadith_hifz._load_progress() == {
        "last_completed_batch": 0,
        "total_tags": 0,
        "tags_by_type": {},
    }


def test_ktbos_progress_uses_dedicated_files_and_legacy_read_only_fallback(tmp_path, monkeypatch):
    run_ktb_os = importlib.import_module("agents.run_ktb_os")
    agent = run_ktb_os.KtbOSAgent.__new__(run_ktb_os.KtbOSAgent)

    quran_progress = tmp_path / "ktbos_quran_progress.json"
    legacy_quran_progress = tmp_path / "hifz_progress.json"
    hadith_progress = tmp_path / "ktbos_hadith_progress.json"
    legacy_hadith_progress = tmp_path / "hadith_hifz_progress.json"

    _write_json(legacy_quran_progress, {"surah": 7, "ayah": 21})
    _write_json(legacy_hadith_progress, {"source": "Muslim", "index": 88})

    monkeypatch.setattr(run_ktb_os, "_QURAN_PROGRESS_PATH", str(quran_progress))
    monkeypatch.setattr(run_ktb_os, "_LEGACY_QURAN_PROGRESS_PATH", str(legacy_quran_progress))
    monkeypatch.setattr(run_ktb_os, "_HADITH_PROGRESS_PATH", str(hadith_progress))
    monkeypatch.setattr(run_ktb_os, "_LEGACY_HADITH_PROGRESS_PATH", str(legacy_hadith_progress))

    assert agent._load_quran_progress() == {"surah": 7, "ayah": 21}
    assert agent._load_hadith_progress() == {"source": "Muslim", "index": 88}

    agent._save_quran_progress(8, 1)
    agent._save_hadith_progress("Muslim", 96)

    assert json.loads(quran_progress.read_text(encoding="utf-8")) == {"surah": 8, "ayah": 1}
    assert json.loads(hadith_progress.read_text(encoding="utf-8")) == {"source": "Muslim", "index": 96}
    assert json.loads(legacy_quran_progress.read_text(encoding="utf-8")) == {"surah": 7, "ayah": 21}
    assert json.loads(legacy_hadith_progress.read_text(encoding="utf-8")) == {"source": "Muslim", "index": 88}


def test_server_run_defaults_to_loopback():
    from server import run

    assert inspect.signature(run).parameters["host"].default == "127.0.0.1"
