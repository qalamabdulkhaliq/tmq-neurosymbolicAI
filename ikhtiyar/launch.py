"""
ikhtiyar/launch.py — Preflight + startup

Checks:
  1. Ollama reachable on :11434
  2. TMQ_v12.json exists
  3. quran_root_ontology_v3.ttl exists
  4. Neo4j on :7687 (warn, not fail)

Then starts IkhtiyarEngine + Flask on :5000
"""

import socket
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch")

_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_DIR)
_BISMILLAH = os.path.join(_PROJECT, "bismillah")
_QUSAI_HF  = os.path.join(_BISMILLAH, "QUS-AI HF")

TMQ_PATH = os.environ.get("TMQ_PATH", os.path.join(_BISMILLAH, "TMQ_v12.json"))
TTL_PATH = os.environ.get("TTL_PATH", os.path.join(_QUSAI_HF,  "quran_root_ontology_v3.ttl"))


def _check_socket(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def preflight():
    ok = True

    # Ollama
    if _check_socket("localhost", 11434):
        logger.info("✓ Ollama reachable on :11434")
    else:
        logger.error("✗ Ollama not reachable on :11434 — start Ollama first")
        ok = False

    # TMQ
    if os.path.exists(TMQ_PATH):
        size_mb = os.path.getsize(TMQ_PATH) / 1_000_000
        logger.info(f"✓ TMQ_v12.json found ({size_mb:.0f} MB)")
    else:
        logger.error(f"✗ TMQ_v12.json not found at {TMQ_PATH}")
        logger.error("  Move TMQ_v12.json from bismillah/ to ikhtiyar/ or set TMQ_PATH env var")
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

    from engine import IkhtiyarEngine
    from server import create_app

    engine = IkhtiyarEngine(tmq_path=TMQ_PATH, ttl_path=TTL_PATH)
    engine.start()

    app = create_app(engine)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
