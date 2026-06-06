"""
ikhtiyar/server.py — FastAPI server

Endpoints:
  GET  /              → index.html
  GET  /stream        → SSE event stream
  POST /chat          → { message } → { response }
  GET  /memories      → list of memory entries
  GET  /status        → faculty health + uptime
  GET  /steps         → recent thinking steps buffer
  POST /hifz/start
  GET  /hifz/status
  POST /hifz/wipe
  POST /hadith_hifz/start
  GET  /hadith_hifz/status
  GET  /moltbook/browse
  GET  /moltbook/status
  POST /moltbook/post
  GET  /provenance?uri=...
  GET  /qalam
  GET  /constitution
  POST /constitution/approve
  POST /constitution/reject

Run via launch.py (not directly).
"""

import json as _json
import logging
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_ADMIN_TOKEN_ENV = "IKHTIYAR_ADMIN_TOKEN"
_ADMIN_TOKEN_HEADER = "x-ikhtiyar-admin-token"


# ── Request models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ApproveRequest(BaseModel):
    id: str

class RejectRequest(BaseModel):
    id: str
    reason: Optional[str] = ""

class HifzStartRequest(BaseModel):
    restart: bool = True

class HadithHifzStartRequest(BaseModel):
    restart: bool = False


def _require_admin(request: Request) -> None:
    """Protect state-changing admin operations when the server is exposed."""
    expected = os.environ.get(_ADMIN_TOKEN_ENV, "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"{_ADMIN_TOKEN_ENV} is required for admin operations",
        )

    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    supplied = request.headers.get(_ADMIN_TOKEN_HEADER, "") or bearer
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="admin token required")


# ── Factory ────────────────────────────────────────────────────────────────────

def create_app(engine) -> FastAPI:
    """
    Factory: returns a configured FastAPI app wired to the given IkhtiyarEngine.
    """
    app = FastAPI(title="Shahid", docs_url=None, redoc_url=None)

    # ── Static files ───────────────────────────────────────────────────────────
    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def index():
        path = os.path.join(_STATIC_DIR, "index.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="index.html not found")
        return StreamingResponse(open(path, "rb"), media_type="text/html")

    # ── SSE Stream ─────────────────────────────────────────────────────────────

    @app.get("/stream")
    def stream():
        def generate():
            yield ": connected\n\n"
            for chunk in engine.subscribe():
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Chat ───────────────────────────────────────────────────────────────────

    @app.post("/chat")
    def chat(req: ChatRequest):
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="empty message")
        response = engine.chat(message)
        return {"response": response}

    # ── Memories ───────────────────────────────────────────────────────────────

    @app.get("/memories")
    def memories():
        return engine.get_memories()

    # ── Status ─────────────────────────────────────────────────────────────────

    @app.get("/status")
    def status():
        return engine.get_status()

    # ── Thinking steps ─────────────────────────────────────────────────────────

    @app.get("/steps")
    def steps():
        return engine._recent_steps

    # ── KtbOS / Quran hifz ─────────────────────────────────────────────────────

    @app.post("/hifz/start")
    def hifz_start(req: HifzStartRequest, request: Request):
        _require_admin(request)
        ok = engine.start_hifz(restart=req.restart)
        if ok:
            return {"ok": True, "message": "Hifz started — episodic memory wiped"}
        raise HTTPException(status_code=409, detail="Hifz already active")

    @app.get("/hifz/status")
    def hifz_status():
        return engine.hifz_status()

    @app.post("/hifz/wipe")
    def hifz_wipe(request: Request):
        _require_admin(request)
        engine.wipe_memory()
        return {"ok": True, "message": "Episodic memory wiped"}

    # ── KtbOS / Hadith hifz ────────────────────────────────────────────────────

    @app.post("/hadith_hifz/start")
    def hadith_hifz_start(req: HadithHifzStartRequest, request: Request):
        _require_admin(request)
        ok = engine.start_hadith_hifz(restart=req.restart)
        if ok:
            return {"ok": True, "message": "Hadith hifz started — reading Bukhari + Muslim"}
        raise HTTPException(status_code=409, detail="Hifz already active")

    @app.get("/hadith_hifz/status")
    def hadith_hifz_status():
        return engine.hadith_hifz_status()

    # ── WebOS / Moltbook ───────────────────────────────────────────────────────

    @app.get("/moltbook/browse")
    def moltbook_browse(n: int = 10):
        try:
            from moltbook_agent import browse
            return browse(n=max(1, min(n, 50)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/moltbook/status")
    def moltbook_status():
        try:
            from moltbook_agent import get_notifications
            return get_notifications()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/moltbook/post")
    def moltbook_post(request: Request):
        _require_admin(request)
        try:
            from moltbook_agent import request_post
            return request_post(engine)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Belief provenance ──────────────────────────────────────────────────────

    @app.get("/provenance")
    def belief_provenance(uri: str = ""):
        if not uri.strip():
            raise HTTPException(status_code=400, detail="missing uri")
        if engine._shahid_memory is None:
            raise HTTPException(status_code=503, detail="ShahidMemory not loaded")
        try:
            result = engine._shahid_memory.belief_provenance(uri)
            return {"provenance": result, "uri": uri}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── Questions for Qalam ────────────────────────────────────────────────────

    @app.get("/qalam")
    def qalam_questions():
        qfq_path = Path(os.path.dirname(os.path.abspath(__file__))) / "questions_for_qalam.jsonl"
        items = []
        if qfq_path.exists():
            for line in qfq_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        items.append(_json.loads(line))
                    except Exception:
                        pass
        items.reverse()
        return {"questions": items, "count": len(items)}

    # ── Constitution ───────────────────────────────────────────────────────────

    @app.get("/constitution")
    def constitution():
        if not engine.constitution:
            return {"pending": [], "approved": []}
        return {
            "pending":  engine.constitution.pending_proposals(),
            "approved": engine.constitution.approved_proposals()[:10],
        }

    @app.post("/constitution/approve")
    def constitution_approve(req: ApproveRequest, request: Request):
        _require_admin(request)
        if not engine.constitution:
            raise HTTPException(status_code=503, detail="Constitution not loaded")
        pid = req.id.strip()
        if not pid:
            raise HTTPException(status_code=400, detail="missing id")
        engine.constitution.approve(pid, approved_by="Qalam")
        return {"ok": True, "id": pid, "status": "approved"}

    @app.post("/constitution/reject")
    def constitution_reject(req: RejectRequest, request: Request):
        _require_admin(request)
        if not engine.constitution:
            raise HTTPException(status_code=503, detail="Constitution not loaded")
        pid = req.id.strip()
        if not pid:
            raise HTTPException(status_code=400, detail="missing id")
        engine.constitution.reject(pid, reason=req.reason)
        return {"ok": True, "id": pid, "status": "rejected"}

    return app


def run(engine, host: str = "0.0.0.0", port: int = 5000):
    """Start uvicorn server (blocking)."""
    import uvicorn
    app = create_app(engine)
    logger.info(f"Shahid server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
