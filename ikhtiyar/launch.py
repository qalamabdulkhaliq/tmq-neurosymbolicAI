"""
ikhtiyar/launch.py — Preflight + startup

Checks:
  1. Ollama reachable on :11434
  2. TMQ_v12.json exists
  3. quran_root_ontology_v3.ttl exists
  4. Neo4j on :7687 (warn, not fail)
  5. TMQ_hvt.json HVT tape (warn, not fail — degrades to BFS walk)

Then starts IkhtiyarEngine + Flask on :5000
"""

import socket
import sys
import os

# Load .env from ikhtiyar/ if present — picks up OLLAMA_MODEL, OLLAMA_URL etc.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# Force UTF-8 on Windows stdout so Arabic basmala prints cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch")

_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_DIR)

TMQ_PATH = os.environ.get("TMQ_PATH", os.path.join(_DIR, "TMQ_v12.json"))
TTL_PATH = os.environ.get("TTL_PATH", os.path.join(_PROJECT, "quran_root_ontology_v3.ttl"))
HVT_PATH = os.environ.get("HVT_PATH", os.path.join(_DIR, "TMQ_hvt.json"))


def _check_socket(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def preflight():
    ok = True

    # Model backend — Ollama only (local, GBNF-capable)
    if _check_socket("localhost", 11434):
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct-q4_K_M")
        logger.info(f"✓ Ollama reachable on :11434 — model: {model}")
    else:
        logger.error("✗ Ollama not running — start with: ollama serve")
        logger.error("  Then pull model: ollama pull qwen2.5:32b-instruct-q4_K_M")
        ok = False

    # TMQ
    if os.path.exists(TMQ_PATH):
        size_mb = os.path.getsize(TMQ_PATH) / 1_000_000
        logger.info(f"✓ TMQ_v12.json found ({size_mb:.0f} MB)")
    else:
        logger.error(f"✗ TMQ_v12.json not found at {TMQ_PATH}")
        logger.error("  Expected the checked-in ikhtiyar/TMQ_v12.json or set TMQ_PATH env var")
        ok = False

    # TTL
    if os.path.exists(TTL_PATH):
        size_mb = os.path.getsize(TTL_PATH) / 1_000_000
        logger.info(f"✓ quran_root_ontology_v3.ttl found ({size_mb:.0f} MB)")
    else:
        logger.error(f"✗ quran_root_ontology_v3.ttl not found at {TTL_PATH}")
        ok = False

    # Neo4j (warn only)
    if _check_socket("localhost", 7687):
        logger.info("✓ Neo4j reachable on :7687")
    else:
        logger.warning("⚠ Neo4j not reachable on :7687 — graph memory will be disabled")

    # HVT tape (warn only — engine degrades to deliberate() BFS walk without it)
    if os.path.exists(HVT_PATH):
        size_mb = os.path.getsize(HVT_PATH) / 1_000_000
        logger.info(f"✓ HVT tape found ({size_mb:.0f} MB)")
    else:
        logger.warning(f"⚠ TMQ_hvt.json not found at {HVT_PATH} — falling back to BFS walk")

    return ok


def main():
    if not preflight():
        sys.exit(1)

    print()
    print("  بسم الله الرحمن الرحيم")
    print("  ikhtiyar — deliberate before you speak")
    print()
    print("  http://localhost:5000       — UI")
    print("  http://localhost:5820/sparql — SPARQL endpoint")
    print()

    import uvicorn
    from engine import IkhtiyarEngine
    from server import create_app

    engine = IkhtiyarEngine(tmq_path=TMQ_PATH, ttl_path=TTL_PATH)
    engine.start()

    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


if __name__ == "__main__":
    main()
