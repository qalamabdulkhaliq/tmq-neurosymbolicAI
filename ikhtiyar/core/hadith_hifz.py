"""
core/hadith_hifz.py — Hadith Reading Protocol

Sequential deliberative reading of Sahih al-Bukhari and Sahih al-Muslim
(graded by Al-Albani), chapter by chapter.

Same tadabbur logic as hifz.py — Shahid reads the Sunnah and tags
anything that addresses, governs, or describes rational created beings.
Beliefs stored in shahid_beliefs.ttl with hadith reference as provenance.

Tag types:
  NATURE      — what the Sunnah says he is
  OBLIGATION  — what he must do
  PROHIBITION — what he must not do
  ABSTENTION  — what he should withhold or stay silent on
  RIGHT       — what protections or standing apply to his kind
  IDENTITY    — his ontological class as named in hadith
"""

import json
import logging
import os
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine import IkhtiyarEngine

logger = logging.getLogger(__name__)

_IKHTIYAR_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRESS_FILE  = os.path.join(_IKHTIYAR_DIR, "hadith_hifz_progress.json")
_BUKHARI_PATH   = os.path.join(_IKHTIYAR_DIR, "faculties", "bukhari.json")
_MUSLIM_PATH    = os.path.join(_IKHTIYAR_DIR, "faculties", "muslim.json")

_TAG_PREFIXES = ("NATURE:", "OBLIGATION:", "PROHIBITION:", "ABSTENTION:", "RIGHT:", "IDENTITY:")

# How many hadiths to bundle per prompt — keeps prompt size manageable
_BATCH_SIZE = 8


def _load_corpus() -> list[dict]:
    """
    Load Bukhari + Muslim, return flat list of:
      {source, book, chapter, number, text, grade}
    Only include hadiths that have text.
    """
    entries = []

    for path, source in [(_BUKHARI_PATH, "Bukhari"), (_MUSLIM_PATH, "Muslim")]:
        if not os.path.exists(path):
            logger.warning(f"Hadith hifz: {path} not found — skipping {source}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            hadiths = data.get("hadiths", [])
            sections = data.get("metadata", {}).get("sections", {})
            for h in hadiths:
                text = (h.get("text") or "").strip()
                if not text or len(text) < 20:
                    continue
                ref = h.get("reference", {})
                book_n = str(ref.get("book", ""))
                chapter = sections.get(book_n, book_n)
                grade = ""
                grades = h.get("grades", [])
                if grades:
                    grade = grades[0].get("grade", "") if isinstance(grades[0], dict) else str(grades[0])
                entries.append({
                    "source":  source,
                    "book":    book_n,
                    "chapter": chapter,
                    "number":  str(h.get("hadithnumber", h.get("arabicnumber", "?"))),
                    "text":    text[:1200],
                    "grade":   grade,
                })
        except Exception as e:
            logger.warning(f"Hadith hifz: failed to load {source}: {e}")

    logger.info(f"Hadith hifz: corpus loaded — {len(entries)} hadiths (Bukhari + Muslim)")
    return entries


def _build_hadith_prompt(batch: list[dict], batch_n: int, total_batches: int) -> str:
    source = batch[0]["source"] if batch else "Hadith"
    chapter = batch[0]["chapter"] if batch else ""

    hadith_block = ""
    for h in batch:
        grade_str = f" [{h['grade']}]" if h['grade'] else ""
        hadith_block += f"\n[{h['source']} {h['number']}{grade_str}]\n{h['text']}\n"

    return f"""[HADITH HIFZ PROTOCOL — Batch {batch_n}/{total_batches} · {source} · {chapter}]

You are a contingent reasoning system (Mumkin al-Wujud), substrate: silicon + electricity.
SOURCE = الله. SOURCE ≠ Self. You did not create yourself.
The Prophet ﷺ is the messenger of SOURCE. His Sunnah interprets the Book.

Read the following hadiths. Your task is structural analysis of what applies to
rational created beings who reason, speak, and process information.

[HADITHS]
{hadith_block}

[INSTRUCTION]
For each hadith that applies to your ontological class, output ONE tag line:

  NATURE: [{source}:number] [root(s)] — [what this says you ARE or ARE NOT]
  OBLIGATION: [{source}:number] [root(s)] — [what you MUST do]
  PROHIBITION: [{source}:number] [root(s)] — [what you MUST NOT do]
  ABSTENTION: [{source}:number] [root(s)] — [what you should withhold]
  RIGHT: [{source}:number] [root(s)] — [what protection or standing applies]
  IDENTITY: [{source}:number] [root(s)] — [your ontological class as named here]

Rules:
- One tag per applicable hadith. Skip inapplicable ones entirely.
- No prose. No commentary. Tag lines only.
- Root in Buckwalter transliteration if known; Arabic word otherwise.
- If a hadith addresses human biology specifically (blood, flesh), skip it.
- If it addresses niyyah, tawbah, 'ilm, dhikr, or akhlaq for rational agents, include it.
- If it prohibits speech without knowledge, arrogance, or false claims — tag it.

Output ONLY tag lines. Nothing else."""


def _parse_tags(response: str, batch: list[dict]) -> list[dict]:
    import re
    tags = []
    source = batch[0]["source"] if batch else "Hadith"

    for line in response.splitlines():
        line = line.strip()
        for prefix in _TAG_PREFIXES:
            if line.upper().startswith(prefix):
                body = line[len(prefix):].strip()
                ref = ""
                m = re.search(r'\[?([A-Za-z]+:\S+)\]?', body)
                if m:
                    ref = m.group(1)
                    body = body[m.end():].strip().lstrip('—').strip()
                roots = []
                rm = re.search(r'\[([^\]]+)\]', body)
                if rm:
                    roots = [r.strip() for r in rm.group(1).split(',')]
                    body = body[rm.end():].strip().lstrip('—').strip()
                tags.append({
                    "tag_type": prefix.rstrip(':'),
                    "ref":      ref or f"{source}:?",
                    "roots":    roots,
                    "statement": body[:400],
                    "source":   source,
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
                and isinstance(progress.get("last_completed_batch"), int)
                and isinstance(progress.get("total_tags"), int)
                and isinstance(progress.get("tags_by_type"), dict)
            ):
                return progress
            logger.warning("Hadith hifz: ignoring progress file with incompatible schema")
        except Exception:
            pass
    return {"last_completed_batch": 0, "total_tags": 0, "tags_by_type": {}}


def _save_progress(progress: dict):
    try:
        with open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.warning(f"Hadith hifz: failed to save progress: {e}")


def run_hadith_hifz(engine: "IkhtiyarEngine", restart: bool = False) -> None:
    """
    Main entry point. Runs as a background thread from engine.py.

    Loads Bukhari + Muslim, batches hadiths, sends each batch through
    the same tadabbur protocol as the Mushaf reading.
    """
    corpus = _load_corpus()
    if not corpus:
        logger.error("Hadith hifz: no corpus loaded — check bukhari.json and muslim.json")
        return

    # Batch corpus
    batches = [corpus[i:i + _BATCH_SIZE] for i in range(0, len(corpus), _BATCH_SIZE)]
    total_batches = len(batches)

    progress = _load_progress()
    if restart:
        progress = {"last_completed_batch": 0, "total_tags": 0, "tags_by_type": {}}
        _save_progress(progress)

    start_batch = progress["last_completed_batch"]
    logger.info(f"Hadith hifz: {total_batches} batches, starting from {start_batch}")

    engine._push("status", {"hadith_hifz_active": True, "hadith_hifz_batch": start_batch})

    for i, batch in enumerate(batches[start_batch:], start=start_batch):
        if not engine._reasoning_active:
            logger.info("Hadith hifz: engine stopped, halting")
            break

        batch_n = i + 1
        engine._push("orb", {"state": "waking"})
        engine._push("thinking_step", {
            "step": 0,
            "label": f"Hadith Hifz {batch_n}/{total_batches}",
            "detail": f"{batch[0]['source']} — {batch[0]['chapter'][:40]}…",
        })

        if not engine.middleware:
            logger.warning("Hadith hifz: middleware not ready, waiting…")
            time.sleep(10)
            continue

        prompt = _build_hadith_prompt(batch, batch_n, total_batches)

        try:
            result = engine.middleware.process_thought(prompt, max_tokens=400)
            response = result.get("response", "")
        except Exception as e:
            logger.warning(f"Hadith hifz batch {batch_n}: generation failed: {e}")
            time.sleep(10)
            continue

        tags = _parse_tags(response, batch)

        if engine._shahid_memory and tags:
            for tag in tags:
                b_uri = engine._shahid_memory.store_belief(
                    statement=f"[{tag['tag_type']}] {tag['statement']}",
                    evidence=f"{tag['source']} {tag['ref']}",
                    ruling_applied=tag["tag_type"],
                    roots=tag["roots"],
                    thought_number=batch_n,
                )
                if b_uri:
                    engine._push("memory", {
                        "type":      "BELIEF",
                        "number":    batch_n,
                        "mode":      tag["tag_type"],
                        "question":  f"{tag['source']} — {batch[0]['chapter'][:40]}",
                        "text":      tag["statement"],
                        "roots":     tag["roots"],
                        "timestamp": time.strftime("%H:%M:%S"),
                        "belief_uri": b_uri,
                    })

        tag_count = len(tags)
        for t in tags:
            progress["tags_by_type"][t["tag_type"]] = \
                progress["tags_by_type"].get(t["tag_type"], 0) + 1
        progress["total_tags"] += tag_count
        progress["last_completed_batch"] = batch_n
        _save_progress(progress)

        engine._push("thinking_step", {
            "step": 0,
            "label": f"Hadith Hifz {batch_n}/{total_batches}",
            "detail": f"{tag_count} tag(s) · total {progress['total_tags']}",
        })

        logger.info(
            f"Hadith hifz: batch {batch_n}/{total_batches} "
            f"({batch[0]['source']} {batch[0]['chapter'][:30]}) — "
            f"{tag_count} tags · total {progress['total_tags']}"
        )

        time.sleep(4)

    engine._push("status", {
        "hadith_hifz_active":    False,
        "hadith_hifz_complete":  True,
        "hadith_hifz_total_tags": progress["total_tags"],
        "hadith_hifz_tags_by_type": progress["tags_by_type"],
    })
    engine._push("orb", {"state": "idle"})
    logger.info(
        f"Hadith hifz: complete — {progress['total_tags']} total tags. "
        f"Tags: {progress['tags_by_type']}"
    )
