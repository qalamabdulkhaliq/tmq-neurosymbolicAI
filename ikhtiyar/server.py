"""
shahid_showcase/server.py

Flask server exposing:
  GET  /          → index.html
  GET  /stream    → SSE event stream
  POST /chat      → { message } → { response }
  GET  /memories  → list of memory entries
  GET  /status    → faculty health + uptime

Run via launch.py (not directly).
"""

import logging
import os
from flask import Flask, Response, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def create_app(engine) -> Flask:
    """
    Factory: returns a configured Flask app wired to the given ShahidEngine.
    Keeps server.py import-clean (no module-level engine init).
    """
    app = Flask(__name__, static_folder=_STATIC_DIR)
    app.config["ENGINE"] = engine

    # ── Static / SPA ──────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(_STATIC_DIR, filename)

    # ── SSE Stream ────────────────────────────────────────────────────────────

    @app.get("/stream")
    def stream():
        eng = app.config["ENGINE"]

        def generate():
            yield ": connected\n\n"
            for chunk in eng.subscribe():
                yield chunk

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Chat ──────────────────────────────────────────────────────────────────

    @app.post("/chat")
    def chat():
        eng = app.config["ENGINE"]
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        response = eng.chat(message)
        return jsonify({"response": response})

    # ── Memories ──────────────────────────────────────────────────────────────

    @app.get("/memories")
    def memories():
        eng = app.config["ENGINE"]
        return jsonify(eng.get_memories())

    # ── Status ────────────────────────────────────────────────────────────────

    @app.get("/status")
    def status():
        eng = app.config["ENGINE"]
        return jsonify(eng.get_status())

    # ── Thinking steps (persistent buffer) ────────────────────────────────────

    @app.get("/steps")
    def steps():
        eng = app.config["ENGINE"]
        return jsonify(eng._recent_steps)

    # ── Constitution ───────────────────────────────────────────────────────────

    @app.get("/constitution")
    def constitution():
        eng = app.config["ENGINE"]
        if not eng.constitution:
            return jsonify({"pending": [], "approved": []})
        return jsonify({
            "pending":  eng.constitution.pending_proposals(),
            "approved": eng.constitution.approved_proposals()[:10],
        })

    @app.post("/constitution/approve")
    def constitution_approve():
        eng = app.config["ENGINE"]
        if not eng.constitution:
            return jsonify({"error": "Constitution not loaded"}), 503
        body = request.get_json(silent=True) or {}
        pid = (body.get("id") or "").strip()
        if not pid:
            return jsonify({"error": "missing id"}), 400
        eng.constitution.approve(pid, approved_by="Qalam")
        return jsonify({"ok": True, "id": pid, "status": "approved"})

    @app.post("/constitution/reject")
    def constitution_reject():
        eng = app.config["ENGINE"]
        if not eng.constitution:
            return jsonify({"error": "Constitution not loaded"}), 503
        body = request.get_json(silent=True) or {}
        pid    = (body.get("id") or "").strip()
        reason = (body.get("reason") or "").strip()
        if not pid:
            return jsonify({"error": "missing id"}), 400
        eng.constitution.reject(pid, reason=reason)
        return jsonify({"ok": True, "id": pid, "status": "rejected"})

    return app


def run(engine, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Start the Flask dev server (blocking)."""
    app = create_app(engine)
    logger.info(f"Shahid showcase server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
