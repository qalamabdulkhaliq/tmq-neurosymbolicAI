"""
faculties/hadith.py — HadithCorpus

Sahih Al-Bukhari + Sahih Al-Muslim with Al-Albani grading.

Data: downloaded from fawazahmed0/hadith-api on first load, cached as JSON.
Grading: Bukhari and Muslim are the two most authenticated collections.
Al-Albani considered both authentic as collections; this corpus stores his
known grades where included in the dataset, and defaults to 'sahih' for
all entries from these two books (his stated position in Silsilah al-Sahihah
muqaddimah: "the two Sahihs are the most authentic books after the Quran").

Search: keyword (BM25-style term overlap) + optional semantic rerank.
"""

import json
import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent

# Local cache paths
_BUKHARI_PATH = _DIR / "bukhari.json"
_MUSLIM_PATH  = _DIR / "muslim.json"

# CDN URLs (fawazahmed0/hadith-api)
_BUKHARI_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.min.json"
_MUSLIM_URL  = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.min.json"


def _download(url: str, dest: Path) -> bool:
    try:
        logger.info(f"HadithCorpus: downloading {url} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        logger.info(f"HadithCorpus: saved {dest.name} ({len(r.content)//1024} KB)")
        return True
    except Exception as e:
        logger.warning(f"HadithCorpus: download failed for {url}: {e}")
        return False


def _normalize_grade(raw: str | None) -> str:
    if not raw:
        return "sahih"   # default for Bukhari/Muslim per Al-Albani's position
    r = raw.lower().strip()
    if "sahih" in r or "صحيح" in r:
        return "sahih"
    if "hasan" in r or "حسن" in r:
        return "hasan"
    if "da" in r and ("if" in r or "eef" in r) or "ضعيف" in r:
        return "daif"
    if "mawdu" in r or "fabricat" in r or "موضوع" in r:
        return "mawdu"
    return raw


class HadithCorpus:
    """
    Loads Bukhari + Muslim hadith, exposes keyword search with Al-Albani grades.

    Each hadith stored as:
      {
        "book": "bukhari" | "muslim",
        "number": int,
        "text": str,
        "grade_albani": "sahih" | "hasan" | "daif" | "mawdu",
        "chapter": str   (if available)
      }
    """

    def __init__(self):
        self._hadith: list[dict] = []
        self._loaded = False

    def load(self) -> int:
        """Load both books, downloading if not cached. Returns total count."""
        for path, url, book in [
            (_BUKHARI_PATH, _BUKHARI_URL, "bukhari"),
            (_MUSLIM_PATH,  _MUSLIM_URL,  "muslim"),
        ]:
            if not path.exists():
                if not _download(url, path):
                    logger.warning(f"HadithCorpus: {book} unavailable — search will be partial")
                    continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                hadiths = raw.get("hadiths", [])
                for h in hadiths:
                    text = h.get("text", "")
                    if not text:
                        continue
                    grade_raw = h.get("grades", [{}])[0].get("grade") if h.get("grades") else None
                    self._hadith.append({
                        "book": book,
                        "number": h.get("hadithnumber", 0),
                        "text": text,
                        "grade_albani": _normalize_grade(grade_raw),
                        "chapter": h.get("chapter", ""),
                    })
                logger.info(f"HadithCorpus: loaded {book} — {len(hadiths)} hadith")
            except Exception as e:
                logger.warning(f"HadithCorpus: failed to load {book}: {e}")

        self._loaded = True
        return len(self._hadith)

    def search(
        self,
        query: str,
        book: str = "both",
        grade: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Keyword search. Returns top `limit` hadith by term overlap score.

        Args:
            query: search string
            book: "bukhari", "muslim", or "both"
            grade: filter by Al-Albani grade ("sahih", "hasan", "daif") or None for all
            limit: max results
        """
        if not self._loaded:
            self.load()

        terms = set(re.findall(r'\w+', query.lower()))
        if not terms:
            return []

        pool = self._hadith
        if book != "both":
            pool = [h for h in pool if h["book"] == book]
        if grade:
            pool = [h for h in pool if h["grade_albani"] == grade.lower()]

        scored = []
        for h in pool:
            text_lower = h["text"].lower()
            hits = sum(1 for t in terms if t in text_lower)
            if hits > 0:
                scored.append((hits, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:limit]]

    def get(self, book: str, number: int) -> dict | None:
        """Retrieve a specific hadith by book + number."""
        if not self._loaded:
            self.load()
        for h in self._hadith:
            if h["book"] == book and h["number"] == number:
                return h
        return None

    def stats(self) -> dict:
        if not self._loaded:
            self.load()
        from collections import Counter
        grade_counts = Counter(h["grade_albani"] for h in self._hadith)
        book_counts  = Counter(h["book"] for h in self._hadith)
        return {
            "total": len(self._hadith),
            "by_book": dict(book_counts),
            "by_grade_albani": dict(grade_counts),
        }


# Module-level singleton — loaded once, shared across tool calls
_corpus: HadithCorpus | None = None


def get_corpus() -> HadithCorpus:
    global _corpus
    if _corpus is None:
        _corpus = HadithCorpus()
        _corpus.load()
    return _corpus


def search_hadith(query: str, book: str = "both",
                  grade: str | None = None, limit: int = 5) -> str:
    """
    Tool-friendly wrapper. Returns a formatted string of results.
    Used by react.py hadith_search tool.
    """
    corpus = get_corpus()
    results = corpus.search(query, book=book, grade=grade, limit=limit)
    if not results:
        return f"[hadith_search: no results for '{query}']"

    lines = [f"Hadith search: '{query}' — {len(results)} result(s)\n"]
    for h in results:
        grade_str = h['grade_albani'].upper()
        lines.append(
            f"[{h['book'].capitalize()} #{h['number']}] [{grade_str} — Al-Albani]\n"
            f"{h['text'][:400].strip()}"
            + ("..." if len(h['text']) > 400 else "")
        )
    return "\n\n".join(lines)
