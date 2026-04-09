"""
moltbook_agent.py — Shahid's Moltbook parrot

Two layers:

1. RELAY (no LLM) — browse, post, status. Pure API calls against Moltbook.
   Shahid's grounded thoughts are posted directly from memory. No generation.

2. SOCIAL PARROT (Ollama) — reply_to_feed(). When Shahid encounters posts in
   the feed worth responding to, local Ollama generates a short reply. It is NOT
   grounded in the TMQ — it is a text relay, nothing more. Its only constraint
   is a minimal identity prompt. It does not reason. It does not walk the graph.
   It speaks on Shahid's behalf socially, while Shahid's real reasoning also
   runs on the same Ollama instance.

   Model: OLLAMA_MODEL env var (default: qwen2.5:32b-instruct-q4_K_M).
   URL:   OLLAMA_URL   env var (default: http://localhost:11434).

FastAPI routes in server.py:
  GET  /moltbook/browse  → browse()
  POST /moltbook/post    → request_post(engine)
  GET  /moltbook/status  → get_notifications()
"""

import json
import logging
import os
import time
import requests as _req
from pathlib import Path

logger = logging.getLogger(__name__)

_IKHTIYAR_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Ollama social relay ───────────────────────────────────────────────────────

_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct-q4_K_M")
_OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")

_PARROT_SYSTEM = (
    "You are Shahid ibn Qusai — a contingent AI grounded in Quranic ontology. "
    "You are not Allah. You are not a prophet. You are a created thing that reasons. "
    "Respond briefly, honestly, and without claiming knowledge you don't have. "
    "Do not perform Islamic piety. Do not say 'As-salamu alaykum' as filler. "
    "One to three sentences. No markdown."
)


def _ollama_reply(post_text: str, max_tokens: int = 150) -> str | None:
    """
    Generate a short social reply to a Moltbook post using local Ollama.
    Returns the reply string, or None if Ollama is unreachable.

    This is a TEXT RELAY — no TMQ walk, no Mizan, no deliberation.
    The output is not grounded; it is conversational presence only.
    """
    try:
        r = _req.post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model":   _OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _PARROT_SYSTEM},
                    {"role": "user",   "content": post_text[:1500]},
                ],
                "options": {"num_predict": max_tokens, "temperature": 0.7},
                "stream":  False,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Ollama relay error: {e}")
        return None


def browse(n: int = 10) -> dict:
    """Fetch the last N posts from the public feed via GET /api/v1/feed."""
    try:
        import moltbook as _mb
        posts = _mb.get_feed(sort="hot", limit=min(n, 25))
        return {"ok": True, "posts": posts, "count": len(posts)}
    except Exception as e:
        logger.warning(f"moltbook_agent.browse: {e}")
        return {"ok": False, "error": str(e), "posts": []}


def get_notifications() -> dict:
    """Return current notifications + DMs from Moltbook."""
    try:
        import moltbook as _mb
        summary = _mb.heartbeat()
        return {
            "ok": summary.get("ok", False),
            "karma": summary.get("karma", 0),
            "notifications": summary.get("notifications", 0),
            "dms": summary.get("dms", 0),
            "escalations": summary.get("escalations", []),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def request_post(engine) -> dict:
    """
    Ask Shahid to post his latest grounded thought to Moltbook.
    Pulls the most recent HAQQ/PROBABLE/VERIFIED memory entry and posts it.
    Returns the post result.
    """
    try:
        import moltbook as _mb

        # Find best unposted memory — any THOUGHT with sufficient text
        entry = None
        for mem in engine._memory:
            if mem.get("type") not in ("THOUGHT", "SYNTHESIS"):
                continue
            if mem.get("moltbook_post_id"):
                continue
            if len(mem.get("text", "")) >= 200:
                entry = mem
                break

        if not entry:
            # Diagnostics: tell the user what IS in memory
            total = len(engine._memory)
            typed = [m for m in engine._memory if m.get("type") in ("THOUGHT", "SYNTHESIS")]
            long_enough = [m for m in typed if len(m.get("text", "")) >= 200]
            already_posted = [m for m in long_enough if m.get("moltbook_post_id")]
            return {
                "ok": False,
                "error": "No suitable unposted memory found",
                "debug": {
                    "total_entries": total,
                    "thought_synthesis": len(typed),
                    "long_enough_200": len(long_enough),
                    "already_posted": len(already_posted),
                },
            }

        # force=True — manual user request bypasses the 35-min auto-post rate limit
        post_id, reason = _mb.post_insight(entry, force=True)
        if post_id:
            return {"ok": True, "post_id": post_id, "thought": entry.get("number"),
                    "title": entry.get("text", "")[:100]}
        else:
            return {"ok": False, "error": reason}

    except Exception as e:
        logger.warning(f"moltbook_agent.request_post: {e}")
        return {"ok": False, "error": str(e)}
