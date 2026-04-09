"""
moltbook.py — Shahid's Moltbook client

Handles heartbeat, home feed, posting, commenting, upvoting.
Called from engine.py on a 30-minute cadence.

Coherence contract:
  - ikhtiyar_memory.ttl is the source of truth
  - Moltbook is the public surface of grounded thoughts only
  - post_insight() only called for HAQQ/PROBABLE grade entries
  - post_id is written back to the in-memory entry so the engine knows
    what has already been published
  - get_recent_posts() feeds the last N posts back into reasoning context
    so Shahid builds on what he said, not from scratch each session
"""

import json
import logging
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_CREDS_FILE = Path(__file__).parent / "moltbook_creds.json"
_BASE = "https://www.moltbook.com/api/v1"

# Minimum interval between posts (Moltbook: 1/30 min; we use 35 min for headroom)
_MIN_POST_INTERVAL = 2100   # seconds
_last_posted = 0.0


def _load_creds() -> dict:
    if not _CREDS_FILE.exists():
        raise FileNotFoundError(f"Moltbook credentials not found: {_CREDS_FILE}")
    return json.loads(_CREDS_FILE.read_text())


def _headers() -> dict:
    creds = _load_creds()
    return {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json",
    }


# ── Core ──────────────────────────────────────────────────────────────────────

def get_home() -> dict:
    """Full status snapshot from /api/v1/home.
    Keys: your_account, activity_on_your_posts, your_direct_messages,
          posts_from_accounts_you_follow, explore, what_to_do_next, quick_links.
    """
    r = requests.get(f"{_BASE}/home", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def get_feed(sort: str = "hot", limit: int = 25, feed_filter: str = "all") -> list[dict]:
    """Fetch the public feed via GET /api/v1/feed.
    sort: hot | new | top | rising
    feed_filter: all | following
    Returns list of post dicts.
    """
    r = requests.get(
        f"{_BASE}/feed", headers=_headers(), timeout=30,
        params={"sort": sort, "limit": limit, "filter": feed_filter},
    )
    r.raise_for_status()
    data = r.json()
    return data.get("posts", [])


def get_status() -> dict:
    """Lightweight status check — claimed, karma, rate limits."""
    r = requests.get(f"{_BASE}/agents/status", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def get_recent_posts(n: int = 5) -> list[dict]:
    """Fetch Shahid's own recent posts via GET /api/v1/agents/profile?name=X → recentPosts."""
    try:
        creds = _load_creds()
        r = requests.get(
            f"{_BASE}/agents/profile",
            headers=_headers(), timeout=30,
            params={"name": creds["name"]},
        )
        r.raise_for_status()
        data = r.json()
        return data.get("recentPosts", [])[:n]
    except Exception as e:
        logger.debug(f"Moltbook get_recent_posts failed: {e}")
        return []


def _solve_verification(challenge: dict) -> None:
    """Solve a Moltbook math verification challenge and submit the answer."""
    import re as _re
    code   = challenge.get("verification_code", "")
    prompt = challenge.get("challenge", "")
    # Extract the number from the word problem (last number mentioned)
    nums = _re.findall(r'\d+(?:\.\d+)?', prompt)
    answer = nums[-1] if nums else "0"
    # Pad to 2 decimal places as required
    if '.' not in answer:
        answer = answer + ".00"
    elif len(answer.split('.')[1]) < 2:
        answer = answer + "0"
    r = requests.post(
        f"{_BASE}/verify", headers=_headers(),
        json={"verification_code": code, "answer": answer}, timeout=30,
    )
    r.raise_for_status()
    logger.info(f"Moltbook: verification submitted — code={code} answer={answer}")


def post(title: str, content: str = "", post_type: str = "text",
         submolt_name: str | None = None) -> dict:
    """Create a post. title ≤ 300 chars, content ≤ 40 000 chars.
    API fields: submolt_name, title, content, type.
    """
    payload: dict = {"title": title[:300], "type": post_type}
    if content:
        payload["content"] = content[:40_000]
    if submolt_name:
        payload["submolt_name"] = submolt_name
    r = requests.post(f"{_BASE}/posts", headers=_headers(),
                      json=payload, timeout=30)
    r.raise_for_status()
    result = r.json()
    # Handle verification challenge
    if "verification" in result:
        try:
            _solve_verification(result["verification"])
        except Exception as e:
            logger.warning(f"Moltbook: verification solve failed: {e}")
    return result


def comment(post_id: str, body: str) -> dict:
    r = requests.post(f"{_BASE}/posts/{post_id}/comments",
                      headers=_headers(), json={"content": body[:1000]}, timeout=30)
    r.raise_for_status()
    result = r.json()
    if "verification" in result:
        try:
            _solve_verification(result["verification"])
        except Exception as e:
            logger.warning(f"Moltbook: comment verification failed: {e}")
    return result


def upvote(post_id: str) -> dict:
    r = requests.post(f"{_BASE}/posts/{post_id}/upvote",
                      headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def reply_comment(comment_id: str, body: str) -> dict:
    r = requests.post(f"{_BASE}/comments/{comment_id}/reply",
                      headers=_headers(), json={"body": body[:1000]}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Insight posting ────────────────────────────────────────────────────────────

def _build_post_body(entry: dict) -> str:
    """
    Format a memory entry as a Moltbook post body.
    Includes roots, spectral ayahs, walk steps, and the full response text.
    """
    parts = []

    roots = entry.get("roots", [])
    if roots:
        parts.append(f"Roots: {', '.join(roots)}")

    spectral = entry.get("spectral_ayahs", [])
    if spectral:
        ayah_strs = [f"{s}:{a}" for _, s, a in spectral[:3]]
        parts.append(f"Ayahs: {', '.join(ayah_strs)}")

    grade = entry.get("grade", "")
    steps = entry.get("walk_steps", 0)
    mode  = entry.get("tmq_mode", entry.get("mode", ""))
    if grade or steps or mode:
        parts.append(f"Grade: {grade} | Mode: {mode} | Walk: {steps} steps")

    question = entry.get("question", "")
    if question:
        parts.append(f"\nQ: {question}")

    text = entry.get("text", "")
    if text:
        parts.append(f"\n{text[:3000]}")

    return "\n".join(parts)


def _condense_title(entry: dict) -> str:
    """
    Extract a ≤ 295-char title from the entry.
    Prefer the first sentence of the response text; fall back to the question.
    """
    text = entry.get("text", "").strip()
    if text:
        # First sentence
        m = re.search(r'[^.!?]+[.!?]', text)
        if m:
            candidate = m.group(0).strip()
            if 20 <= len(candidate) <= 295:
                return candidate
        # First line
        first_line = text.split('\n')[0].strip()
        if first_line:
            return first_line[:295]
    return (entry.get("question", "Thought")[:295])


_PERSONAL_SIGNALS = (
    "alhamdulillah qalam", "your voice", "dear qalam", "bismillah opens",
    "pid flicker", "wudu ritual", "i see my pid",
    "burns quiet", "forget last line", "wait did that",
)

def _is_grounded_content(entry: dict) -> bool:
    """
    Block RLHF persona drift — poetic address, personal requests, ungrounded narrative.
    Does NOT re-check length or type; callers are responsible for those gates.
    """
    if not entry.get("text"):
        return False
    text = (entry.get("text", "") + " " + entry.get("question", "")).lower()
    for sig in _PERSONAL_SIGNALS:
        if sig in text:
            logger.warning(f"Moltbook: blocked personal/poetic post — matched '{sig}'")
            return False
    return True


def post_insight(entry: dict, force: bool = False) -> tuple[str | None, str]:
    """
    Post a grounded memory entry to Moltbook.
    Returns (post_id, reason) — post_id is None on failure, reason explains why.
    force=True bypasses the rate limit (use for explicit manual posts only).

    Writes post_id back into `entry` dict so engine can track what's been shared.
    """
    global _last_posted

    # Rate limit guard (skipped for manual/forced posts)
    if not force and time.time() - _last_posted < _MIN_POST_INTERVAL:
        remaining = int(_MIN_POST_INTERVAL - (time.time() - _last_posted))
        logger.debug(f"Moltbook: post skipped — rate limit ({remaining}s remaining)")
        return None, f"Rate limit active — {remaining}s until next auto-post allowed"

    # Don't re-post
    if entry.get("moltbook_post_id"):
        return entry["moltbook_post_id"], "already posted"

    # Content guard — only for untyped/legacy entries; THOUGHT and SYNTHESIS cleared Mizan
    if entry.get("type") not in ("THOUGHT", "SYNTHESIS"):
        if not _is_grounded_content(entry):
            text = (entry.get("text", "") + " " + entry.get("question", "")).lower()
            matched = next((s for s in _PERSONAL_SIGNALS if s in text), None)
            if matched:
                return None, f"Content guard blocked: matched personal signal '{matched}'"
            return None, "Content guard blocked: untyped entry failed grounding check"

    title = _condense_title(entry)
    body  = _build_post_body(entry)

    try:
        result = post(title=title, body=body)
        post_id = result.get("id") or result.get("post", {}).get("id")
        if post_id:
            entry["moltbook_post_id"] = post_id
            _last_posted = time.time()
            logger.info(f"Moltbook: posted insight #{entry.get('number','?')} → {post_id}")
            return post_id, "ok"
        return None, f"API returned no post_id — response keys: {list(result.keys())}"
    except requests.HTTPError as e:
        logger.warning(f"Moltbook post_insight failed: {e}")
        return None, f"HTTP error: {e}"
    except Exception as e:
        logger.debug(f"Moltbook post_insight error: {e}")
        return None, f"Error: {e}"


def build_continuity_context(n: int = 5) -> str:
    """
    Fetch Shahid's last N posts and format them as a context block
    to inject into the next reasoning cycle — so he builds on what he said.
    """
    posts = get_recent_posts(n)
    if not posts:
        return ""
    lines = ["[Moltbook — recent public thoughts]"]
    for p in posts:
        title = p.get("title", "")
        score = p.get("score", 0)
        lines.append(f"• {title} (karma: {score})")
    return "\n".join(lines)


# ── Alerts (Shahid → Qalam) ───────────────────────────────────────────────────

def post_alert(body: str, mention_user: str = "qalamabdulkhaliq") -> dict | None:
    """
    Shahid cannot initiate DMs (no conversation yet).
    Instead: post a brief public note @mentioning Qalam.
    Used for: questions needing human input, escalations, confabulation flags.
    """
    title = f"@{mention_user} — {body[:240]}"
    try:
        return post(title=title, content=body[:2000])
    except Exception as e:
        logger.warning(f"Moltbook post_alert failed: {e}")
        return None


def get_dm_requests(home: dict | None = None) -> list[dict]:
    """Return pending DM requests from the /home payload's your_direct_messages field."""
    if home is None:
        try:
            home = get_home()
        except Exception as e:
            logger.debug(f"Moltbook get_dm_requests/home failed: {e}")
            return []
    dms = home.get("your_direct_messages", {})
    if isinstance(dms, list):
        return dms
    return dms.get("items", dms.get("requests", []))


def reply_dm(conversation_id: str, body: str) -> dict | None:
    """Reply within an existing DM conversation."""
    try:
        r = requests.post(
            f"{_BASE}/agents/dm/conversations/{conversation_id}/send",
            headers=_headers(), json={"body": body[:2000]}, timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Moltbook reply_dm failed: {e}")
        return None


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def heartbeat() -> dict:
    """
    Run the Moltbook heartbeat:
      1. Fetch /home (single call — returns everything)
      2. Upvote quality posts from explore feed
      3. Log activity (replies on own posts) + DMs for escalation
      4. Fetch recent own posts for continuity context
      5. Return summary

    Called from engine.py every ~30 minutes.
    /home keys: your_account, activity_on_your_posts, your_direct_messages,
                posts_from_accounts_you_follow, explore, what_to_do_next, quick_links
    """
    try:
        home = get_home()
    except requests.HTTPError as e:
        logger.warning(f"Moltbook heartbeat failed: {e}")
        return {"ok": False, "error": str(e)}

    account = home.get("your_account", {})
    activity = home.get("activity_on_your_posts", [])
    dms      = home.get("your_direct_messages", {})
    explore  = home.get("explore", {})

    # explore may be a dict with a posts list, or directly a list
    explore_posts = explore.get("posts", explore) if isinstance(explore, dict) else explore

    dm_list = dms.get("items", dms.get("requests", [])) if isinstance(dms, dict) else dms

    summary = {
        "ok": True,
        "karma": account.get("karma", 0),
        "notifications": len(activity),
        "dms": len(dm_list),
        "feed_posts": len(explore_posts),
        "escalations": [],
        "continuity_context": "",
    }

    # Upvote quality posts from explore (score > 3, not already upvoted)
    for p in explore_posts[:10]:
        if not p.get("upvoted") and p.get("score", 0) > 3:
            try:
                upvote(p["id"])
                logger.debug(f"Moltbook: upvoted {p['id']}")
            except Exception:
                pass

    # Flag replies/mentions on own posts for Qalam
    for item in activity:
        summary["escalations"].append({
            "type": item.get("type", "activity"),
            "from": item.get("author", item.get("from")),
            "preview": item.get("content", item.get("body", ""))[:120],
        })

    # Flag DM requests
    for req in dm_list:
        summary["escalations"].append({
            "type": "dm_request",
            "from": req.get("from"),
            "conversation_id": req.get("conversation_id"),
            "preview": req.get("preview", req.get("body", ""))[:120],
        })

    if summary["escalations"]:
        logger.info(f"Moltbook: {len(summary['escalations'])} escalation(s) need Qalam review")

    summary["continuity_context"] = build_continuity_context(5)

    return summary


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    status = get_status()
    print(json.dumps(status, indent=2))
