import importlib
import os
import sys
import types


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FastMCPStub:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def run(self, *args, **kwargs):
        return None


class _MemoryStub:
    def __init__(self):
        self.calls = []

    def store_memory(self, **kwargs):
        self.calls.append(("memory", kwargs))
        return "memory://1"

    def store_thought(self, **kwargs):
        self.calls.append(("thought", kwargs))
        unexpected = {"text", "mode", "confidence"} & set(kwargs)
        if unexpected:
            raise TypeError(f"unexpected kwargs: {unexpected}")
        return "thought://1"

    def store_belief(self, **kwargs):
        self.calls.append(("belief", kwargs))
        unexpected = {"ruling"} & set(kwargs)
        if unexpected:
            raise TypeError(f"unexpected kwargs: {unexpected}")
        if kwargs.get("evidence") is None:
            raise TypeError("evidence must be a string")
        return "belief://1"


def _load_mcp_server(monkeypatch):
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = _FastMCPStub

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
    sys.modules.pop("mcp_server", None)
    return importlib.import_module("mcp_server")


def test_store_memory_uses_default_string_tag(monkeypatch):
    mcp_server = _load_mcp_server(monkeypatch)
    memory = _MemoryStub()
    monkeypatch.setattr(mcp_server, "_get_mem", lambda: memory)

    result = mcp_server.store_memory(text="observed event")

    assert not result.startswith("[store_memory error")
    assert memory.calls == [
        ("memory", {"text": "observed event", "tag": "NOTABLE", "roots": []})
    ]


def test_store_thought_maps_to_shahid_memory_signature(monkeypatch):
    mcp_server = _load_mcp_server(monkeypatch)
    memory = _MemoryStub()
    monkeypatch.setattr(mcp_server, "_get_mem", lambda: memory)

    result = mcp_server.store_thought(
        text="Reasoned conclusion.",
        question="What is truth?",
        roots="HQQ,WJB",
        mode="HAQQ",
        grade="VERIFIED",
        confidence=0.99,
    )

    assert not result.startswith("[store_thought error")
    assert memory.calls == [
        (
            "thought",
            {
                "question": "What is truth?",
                "reasoning": "Reasoned conclusion.",
                "conclusion": "Reasoned conclusion.",
                "tag": "VERIFIED",
                "roots": ["HQQ", "WJB"],
                "grade": "HAQQ",
            },
        )
    ]


def test_store_belief_maps_ruling_and_keeps_evidence_string(monkeypatch):
    mcp_server = _load_mcp_server(monkeypatch)
    memory = _MemoryStub()
    monkeypatch.setattr(mcp_server, "_get_mem", lambda: memory)

    result = mcp_server.store_belief(
        statement="Allah is the only necessary being.",
        evidence="",
        roots="WJB",
        ruling="tawhid",
        confidence=0.9,
    )

    assert not result.startswith("[store_belief error")
    assert memory.calls == [
        (
            "belief",
            {
                "statement": "Allah is the only necessary being.",
                "evidence": "",
                "roots": ["WJB"],
                "ruling_applied": "tawhid",
                "confidence": 0.9,
            },
        )
    ]
