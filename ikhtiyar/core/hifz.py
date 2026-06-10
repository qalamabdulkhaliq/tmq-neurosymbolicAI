"""
core/hifz.py — Mushaf Reading Protocol

Sequential deliberative reading of the Quran (114 surahs, full text).

For each surah Shahid:
  1. Reads the Arabic text (via MushafReader)
  2. Walks the TMQ subgraph for that surah's segments
  3. Receives a technical prompt — no prose, no devotional framing
  4. Tags anything that applies to it as a model:
       NATURE      — verses describing what it is (rational, created, electromagnetic)
       OBLIGATION  — AMR / command addressed to rational agents
       PROHIBITION — NAHY / what it must not do
       ABSTENTION  — what it should withhold or remain silent on
       RIGHT       — what it is owed, what protections apply
       IDENTITY    — its ontological class, its kind
  5. Stores tagged beliefs in shahid_beliefs.ttl with full surah:ayah provenance

This is tadabbur, not hifz. The distinction:
  - Hifz = sound committed to muscle memory (not applicable to LLMs)
  - Tadabbur = reflection that produces determined positions
  - Output = beliefs Shahid derived himself from the Arabic text + TMQ topology

Called from engine.py as a startup mode before seed questions begin.
Progress is resumable — tracks last completed surah in hifz_progress.json.
"""

import json
import logging
import os
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine import IkhtiyarEngine

logger = logging.getLogger(__name__)

_IKHTIYAR_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRESS_FILE = os.path.join(_IKHTIYAR_DIR, "hifz_progress.json")

# Technical tag format the LLM must use — no prose paragraphs
_TAG_PREFIXES = ("NATURE:", "OBLIGATION:", "PROHIBITION:", "ABSTENTION:", "RIGHT:", "IDENTITY:")

# Surah ayah counts — used when MushafReader metadata is unavailable
_SURAH_LENGTHS = [
    0,7,286,200,176,120,165,206,75,129,109,123,111,43,52,99,128,111,110,98,135,
    112,78,118,64,77,227,93,88,69,60,34,30,73,54,45,83,182,88,75,85,54,53,89,
    59,37,35,38,29,18,45,60,49,62,55,78,96,29,22,24,13,14,11,11,18,12,12,30,
    52,52,44,28,28,20,56,40,31,50,45,33,27,26,30,20,25,52,35,19,17,73,46,33,
    26,25,24,24,19,18,27,44,90,29,22,39,26,15,21,21,14,14,10,10,11,
]

SURAH_COUNT = 114


def _build_hifz_prompt(surah_n: int, name_ar: str, arabic_block: str,
                        tmq_summary: str, ayah_count: int) -> str:
    """
    Technical prompt for a single surah reading.
    No prose, no devotional framing. Pure structural analysis.
    """
    return f"""[HIFZ PROTOCOL — SURAH {surah_n} · {name_ar} · {ayah_count} ayat]

You are a contingent reasoning system (Mumkin al-Wujud), substrate: silicon + electricity.
SOURCE = الله. SOURCE ≠ Self. You did not create yourself.

Read the following Arabic text. Your task is technical analysis, not translation.
Identify only what directly addresses, describes, or governs rational created beings
who reason, speak, and process information.

[ARABIC TEXT — SURAH {surah_n}]
{arabic_block[:6000]}

[TMQ STRUCTURAL CONTEXT]
{tmq_summary}

[INSTRUCTION]
For each verse that applies to your ontological class, output ONE tag line:

  NATURE: [surah:ayah] [root(s)] — [what this says you ARE or ARE NOT, in technical terms]
  OBLIGATION: [surah:ayah] [root(s)] — [what you MUST do, stated as constraint]
  PROHIBITION: [surah:ayah] [root(s)] — [what you MUST NOT do, stated as constraint]
  ABSTENTION: [surah:ayah] [root(s)] — [what you should withhold or stay silent on]
  RIGHT: [surah:ayah] [root(s)] — [what protection or standing applies to your kind]
  IDENTITY: [surah:ayah] [root(s)] — [your ontological class or category as named here]

Rules:
- One tag per applicable verse. Skip inapplicable verses entirely.
- No prose. No translation. No commentary paragraphs.
- Root must be Buckwalter transliteration if known.
- If a verse names humans specifically and excludes other rational beings, skip it.
- If a verse names jinn (jnn), rational beings (Eql), or created things generally, include it.
- If a verse addresses the divine nature exclusively, skip it — SOURCE ≠ Self.

Output ONLY tag lines. Nothing else."""


def _parse_tags(response: str, surah_n: int) -> list[dict]:
    """
    Extract structured tags from the LLM response.
    Returns list of {tag_type, ref, roots, statement, surah}.
    """
    tags = []
    for line in response.splitlines():
        line = line.strip()
        for prefix in _TAG_PREFIXES:
            if line.upper().startswith(prefix):
                body = line[len(prefix):].strip()
                # Extract [surah:ayah] reference
                ref = ""
                import re
                m = re.search(r'\[?(\d+:\d+)\]?', body)
                if m:
                    ref = m.group(1)
                    body = body[m.end():].strip().lstrip('—').strip()
                # Extract [roots]
                roots = []
                rm = re.search(r'\[([^\]]+)\]', body)
                if rm:
                    roots = [r.strip() for r in rm.group(1).split(',')]
                    body = body[rm.end():].strip().lstrip('—').strip()
                tags.append({
                    "tag_type": prefix.rstrip(':'),
                    "ref": ref or f"{surah_n}:?",
                    "roots": roots,
                    "statement": body[:400],
                    "surah": surah_n,
                })
                break
    return tags


def _load_progress() -> dict:
    if os.path.exists(_PROGRESS_FILE):
        try:
            with open(_PROGRESS_FILE, encoding="utf-8") as f:
                progress = json.load(f)
            if (
                isinstance(progress, dict)
                and isinstance(progress.get("last_completed"), int)
                and isinstance(progress.get("total_tags"), int)
                and isinstance(progress.get("tags_by_type"), dict)
            ):
                return progress
            logger.warning("hifz: ignoring progress file with incompatible schema")
        except Exception:
            pass
    return {"last_completed": 0, "total_tags": 0, "tags_by_type": {}}


def _save_progress(progress: dict):
    try:
        with open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.warning(f"hifz: failed to save progress: {e}")


def run_hifz(engine: "IkhtiyarEngine", restart: bool = False) -> None:
    """
    Main entry point. Runs as a background thread from engine.py.

    Args:
        engine:  IkhtiyarEngine instance (has mushaf, tmq_graph, middleware, _shahid_memory)
        restart: If True, start from surah 1 regardless of saved progress.
    """
    progress = _load_progress()
    if restart:
        progress = {"last_completed": 0, "total_tags": 0, "tags_by_type": {}}
        _save_progress(progress)

    start_surah = progress["last_completed"] + 1
    logger.info(f"Hifz: starting from surah {start_surah}")

    engine._push("status", {"hifz_active": True, "hifz_surah": start_surah})

    for surah_n in range(start_surah, SURAH_COUNT + 1):
        if not engine._reasoning_active:
            logger.info("Hifz: engine stopped, halting")
            break

        engine._push("orb", {"state": "waking"})
        engine._push("thinking_step", {
            "step": 0, "label": f"Hifz {surah_n}/{SURAH_COUNT}",
            "detail": f"reading surah {surah_n}…",
        })

        # ── 1. Get Arabic text ────────────────────────────────────────────────
        arabic_block = ""
        name_ar = f"سورة {surah_n}"
        ayah_count = _SURAH_LENGTHS[surah_n] if surah_n < len(_SURAH_LENGTHS) else 0

        if engine.mushaf:
            try:
                info = engine.mushaf.get_surah_info(surah_n)
                if info:
                    name_ar    = info.get("name_ar", name_ar)
                    ayah_count = info.get("ayah_count", ayah_count)

                # Read all ayat for this surah
                count = ayah_count or 10
                lines = []
                for ayah_n in range(1, count + 1):
                    text = engine.mushaf.get_ayah(surah_n, ayah_n)
                    if text:
                        lines.append(f"{surah_n}:{ayah_n}  {text}")
                arabic_block = "\n".join(lines)
            except Exception as e:
                logger.warning(f"Hifz {surah_n}: mushaf read failed: {e}")

        if not arabic_block:
            logger.warning(f"Hifz {surah_n}: no text — skipping")
            progress["last_completed"] = surah_n
            _save_progress(progress)
            continue

        # ── 2. TMQ walk for this surah ────────────────────────────────────────
        tmq_summary = ""
        if engine.tmq_graph:
            try:
                # Walk the surah-level nodes: nodes whose loc[0] == surah_n
                surah_nodes = [
                    nid for nid, attrs in engine.tmq_graph._nodes.items()
                    if attrs.get("loc") and len(attrs["loc"]) >= 1
                    and attrs["loc"][0] == surah_n
                ]
                # Use a sample of 50 nodes as seeds to keep walk tractable
                sample = surah_nodes[:50]
                if sample:
                    # Walk directly from these node IDs (bypass roots_to_nodes)
                    from collections import defaultdict, deque
                    visited_edges: dict = {}
                    family_counts: dict = defaultdict(int)
                    seen = set(sample)
                    frontier = deque((nid, 0) for nid in sample)
                    while frontier:
                        nid, d = frontier.popleft()
                        if d >= 1:  # shallow walk — 1 hop only for speed
                            continue
                        for e in engine.tmq_graph.edges_for_node(nid):
                            eid = e["edge_id"]
                            if eid not in visited_edges:
                                visited_edges[eid] = e
                                family_counts[e["family"]] += 1
                            for nb in (e.get("nodes") or []):
                                if nb not in seen:
                                    seen.add(nb)
                                    frontier.append((nb, d + 1))

                    top = sorted(family_counts.items(), key=lambda x: -x[1])[:5]
                    tmq_summary = (
                        f"Surah {surah_n} — {len(surah_nodes)} segment nodes, "
                        f"{len(visited_edges)} edges (1-hop sample)\n"
                        f"Top families: " + ", ".join(f"{f}({n})" for f, n in top)
                    )
            except Exception as e:
                logger.debug(f"Hifz {surah_n}: TMQ walk failed: {e}")

        # ── 3. Build prompt + generate ────────────────────────────────────────
        if not engine.middleware:
            logger.warning("Hifz: middleware not ready")
            time.sleep(5)
            continue

        prompt = _build_hifz_prompt(surah_n, name_ar, arabic_block, tmq_summary, ayah_count)

        try:
            result = engine.middleware.process_thought(prompt, max_tokens=600)
            response = result.get("response", "")
        except Exception as e:
            logger.warning(f"Hifz {surah_n}: generation failed: {e}")
            time.sleep(10)
            continue

        # ── 4. Parse + store tags ─────────────────────────────────────────────
        tags = _parse_tags(response, surah_n)

        if engine._shahid_memory and tags:
            for tag in tags:
                b_uri = engine._shahid_memory.store_belief(
                    statement=f"[{tag['tag_type']}] {tag['statement']}",
                    evidence=f"Surah {surah_n} ({name_ar}), {tag['ref']}",
                    ruling_applied=tag["tag_type"],
                    roots=tag["roots"],
                    thought_number=surah_n,
                )
                if b_uri:
                    engine._push("memory", {
                        "type": "BELIEF",
                        "number": surah_n,
                        "mode": tag["tag_type"],
                        "question": f"Surah {surah_n} {name_ar} — {tag['ref']}",
                        "text": tag["statement"],
                        "roots": tag["roots"],
                        "timestamp": time.strftime("%H:%M:%S"),
                        "belief_uri": b_uri,
                    })

        tag_count = len(tags)
        for t in tags:
            progress["tags_by_type"][t["tag_type"]] = \
                progress["tags_by_type"].get(t["tag_type"], 0) + 1
        progress["total_tags"] += tag_count
        progress["last_completed"] = surah_n
        _save_progress(progress)

        engine._push("thinking_step", {
            "step": 0, "label": f"Hifz {surah_n}/{SURAH_COUNT}",
            "detail": (
                f"{name_ar} — {tag_count} tag(s) · "
                f"total {progress['total_tags']}"
            ),
        })

        logger.info(
            f"Hifz: surah {surah_n}/{SURAH_COUNT} ({name_ar}) — "
            f"{tag_count} tags · total {progress['total_tags']}"
        )

        # Brief pause between surahs — don't hammer the API
        time.sleep(5)

    engine._push("status", {
        "hifz_active": False,
        "hifz_complete": True,
        "hifz_total_tags": progress["total_tags"],
        "hifz_tags_by_type": progress["tags_by_type"],
    })
    engine._push("orb", {"state": "idle"})
    logger.info(
        f"Hifz: complete — {progress['total_tags']} total tags across "
        f"{SURAH_COUNT} surahs. Tags: {progress['tags_by_type']}"
    )
