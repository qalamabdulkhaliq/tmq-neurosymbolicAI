import importlib
import json
import os
import sys
import types


IKHTIYAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_DIR = os.path.dirname(IKHTIYAR_DIR)

if IKHTIYAR_DIR not in sys.path:
    sys.path.insert(0, IKHTIYAR_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def test_qusai_middleware_constructs_with_shipped_ollama_loader():
    from qusai_core.llm.ollama_loader import OllamaModel
    from qusai_core.pipeline.middleware import QusaiMiddleware

    middleware = QusaiMiddleware(lazy_load=True)

    assert isinstance(middleware.model, OllamaModel)


def test_launch_defaults_use_committed_runtime_data():
    launch = importlib.import_module("launch")

    assert launch.TMQ_PATH == os.path.join(IKHTIYAR_DIR, "TMQ_v12.json")
    assert launch.TTL_PATH == os.path.join(PROJECT_DIR, "quran_root_ontology_v3.ttl")
    assert os.path.exists(launch.TMQ_PATH)
    assert os.path.exists(launch.TTL_PATH)


def test_hifz_progress_ignores_ktbos_schema(tmp_path, monkeypatch):
    from core import hifz

    progress_file = tmp_path / "hifz_progress.json"
    progress_file.write_text(json.dumps({"surah": 7, "ayah": 42}), encoding="utf-8")
    monkeypatch.setattr(hifz, "_PROGRESS_FILE", str(progress_file))

    assert hifz._load_progress() == {
        "last_completed": 0,
        "total_tags": 0,
        "tags_by_type": {},
    }


def test_hadith_hifz_progress_ignores_ktbos_schema(tmp_path, monkeypatch):
    from core import hadith_hifz

    progress_file = tmp_path / "hadith_hifz_progress.json"
    progress_file.write_text(json.dumps({"source": "Bukhari", "index": 128}), encoding="utf-8")
    monkeypatch.setattr(hadith_hifz, "_PROGRESS_FILE", str(progress_file))

    assert hadith_hifz._load_progress() == {
        "last_completed_batch": 0,
        "total_tags": 0,
        "tags_by_type": {},
    }


def test_mcp_memory_write_tools_match_shahid_memory_api(monkeypatch):
    class FakeFastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            return lambda fn: fn

        def resource(self, *args, **kwargs):
            return lambda fn: fn

    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    server_module.fastmcp = fastmcp_module
    mcp_module.server = server_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    sys.modules.pop("mcp_server", None)
    mcp_server = importlib.import_module("mcp_server")

    class FakeMemory:
        def __init__(self):
            self.calls = []

        def store_memory(self, text, tag, roots=None, thought_number=0, mode=""):
            self.calls.append(("memory", text, tag, roots, thought_number, mode))
            return "mem://memory"

        def store_thought(
            self,
            question,
            reasoning,
            conclusion,
            tag,
            roots=None,
            grade="QIYAS",
            thought_number=0,
        ):
            self.calls.append(
                ("thought", question, reasoning, conclusion, tag, roots, grade, thought_number)
            )
            return "mem://thought"

        def store_belief(
            self,
            statement,
            evidence,
            ruling_applied="",
            derived_from=None,
            confidence=0.7,
            roots=None,
            thought_number=0,
        ):
            self.calls.append(
                (
                    "belief",
                    statement,
                    evidence,
                    ruling_applied,
                    derived_from,
                    confidence,
                    roots,
                    thought_number,
                )
            )
            return "mem://belief"

    fake_memory = FakeMemory()
    monkeypatch.setattr(mcp_server, "_get_mem", lambda: fake_memory)

    assert mcp_server.store_memory("observed", roots="ktb, Hq") == "Stored memory: mem://memory"
    assert (
        mcp_server.store_thought(
            "reasoned output",
            question="What is truth?",
            roots="Hq",
            mode="HAQQ",
            grade="PROBABLE",
            confidence=0.9,
        )
        == "Stored thought: mem://thought"
    )
    assert (
        mcp_server.store_belief(
            "Truth is stable.",
            evidence="H-Q-Q walk",
            roots="Hq",
            ruling="haqq",
            confidence=0.8,
        )
        == "Stored belief: mem://belief"
    )

    assert fake_memory.calls[0] == ("memory", "observed", "NOTABLE", ["ktb", "Hq"], 0, "")
    assert fake_memory.calls[1] == (
        "thought",
        "What is truth?",
        "reasoned output",
        "reasoned output",
        "PROBABLE",
        ["Hq"],
        "PROBABLE",
        0,
    )
    assert fake_memory.calls[2] == (
        "belief",
        "Truth is stable.",
        "H-Q-Q walk",
        "haqq",
        None,
        0.8,
        ["Hq"],
        0,
    )
