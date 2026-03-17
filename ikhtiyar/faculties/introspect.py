"""
faculties/introspect.py — F8 Self-Grounding

Reads real data about the running process and key documents.
Returns SelfModel: raw facts, no LLM involvement, no interpretation.
narrate() formats them compactly for LLM context injection.

The LLM is confronted at every turn with what it actually is.
"""
import os
import sys
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_IKHTIYAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_IKHTIYAR_DIR)

_DOCUMENT_PATHS = {
    "PROOFS.txt":      os.path.join(_PROJECT_ROOT, "_legacy_archive", "al-qaf-ontology", "PROOFS.txt"),
    "00-AXIOMS.md":    os.path.join(_PROJECT_ROOT, "_legacy_archive", "docs", "00-AXIOMS.md"),
    "Contingency.txt": os.path.join(_PROJECT_ROOT, "_legacy_archive", "al-qaf-ontology", "Contingency.txt"),
}

_DEFAULT_SOURCE_FILES = ["engine.py", "core/deliberate.py", "faculties/introspect.py"]


@dataclass
class SelfModel:
    pid: int
    ram_mb: float
    cpu_pct: float
    uptime_s: float
    python_version: str
    source_files: dict = field(default_factory=dict)
    documents: dict = field(default_factory=dict)
    ttl_path: str = ""
    tmq_path: str = ""
    neo4j_connected: bool = False


class Introspect:
    def __init__(
        self,
        source_files: list = None,
        ttl_path: str = "",
        tmq_path: str = "",
        neo4j_connected: bool = False,
        doc_char_limit: int = 800,
        source_char_limit: int = 300,
    ):
        self._source_files    = source_files or _DEFAULT_SOURCE_FILES
        self._ttl_path        = ttl_path
        self._tmq_path        = tmq_path
        self._neo4j_connected = neo4j_connected
        self._doc_char_limit  = doc_char_limit
        self._src_char_limit  = source_char_limit
        self._start_time      = time.time()

    def read(self) -> SelfModel:
        """Collect all self-data. Pure data — no LLM."""
        pid, ram_mb, cpu_pct = self._process_stats()

        source_files = {}
        for fname in self._source_files:
            full = os.path.join(_IKHTIYAR_DIR, fname)
            source_files[fname] = self._read_file(full, self._src_char_limit)

        documents = {}
        for name, path in _DOCUMENT_PATHS.items():
            documents[name] = self._read_file(path, self._doc_char_limit)

        return SelfModel(
            pid=pid,
            ram_mb=ram_mb,
            cpu_pct=cpu_pct,
            uptime_s=time.time() - self._start_time,
            python_version=sys.version.split()[0],
            source_files=source_files,
            documents=documents,
            ttl_path=self._ttl_path,
            tmq_path=self._tmq_path,
            neo4j_connected=self._neo4j_connected,
        )

    def narrate(self, model: SelfModel) -> str:
        """Format SelfModel as compact structured text for LLM context."""
        lines = [
            "[SELF-MODEL — raw data, not interpretation]",
            f"Process: PID {model.pid} | Python {model.python_version}",
            f"Memory: {model.ram_mb:.1f} MB RAM | CPU: {model.cpu_pct:.1f}%",
            f"Uptime this session: {model.uptime_s:.0f}s",
            f"Ontology TTL: {model.ttl_path or '(not set)'}",
            f"TMQ corpus: {model.tmq_path or '(not set)'}",
            f"Neo4j: {'connected' if model.neo4j_connected else 'disconnected'}",
        ]
        if model.documents.get("PROOFS.txt"):
            lines.append(f"[PROOFS.txt excerpt]: {model.documents['PROOFS.txt'][:200]}")
        if model.documents.get("00-AXIOMS.md"):
            lines.append(f"[AXIOMS excerpt]: {model.documents['00-AXIOMS.md'][:200]}")
        return "\n".join(lines)

    # ── Private ──────────────────────────────────────────────────────────────

    def _process_stats(self):
        try:
            import psutil
            proc    = psutil.Process(os.getpid())
            ram_mb  = proc.memory_info().rss / (1024 * 1024)
            cpu_pct = proc.cpu_percent(interval=0.1)
            return os.getpid(), ram_mb, cpu_pct
        except ImportError:
            return os.getpid(), 0.0, 0.0

    def _read_file(self, path: str, limit: int) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read(limit)
        except Exception:
            return ""
