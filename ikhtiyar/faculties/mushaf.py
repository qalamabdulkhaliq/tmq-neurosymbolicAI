"""
faculties/mushaf.py — MushafReader

Gives Shahid direct access to the Arabic text of the Quran.

Data source: quran-simple.txt  (surah|ayah|arabic_text, one line per ayah)
Supplemental: mushaf.xml        (surah names + nozol metadata)

All lookups are O(1) dict access after a one-time parse at startup.
No embeddings, no model loading — just the Words.
"""

import os
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

# Path relative to ikhtiyar/ directory
_DEFAULT_TXT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "quran-simple.txt")
_DEFAULT_XML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "bismillah", "mushaf", "mushaf.xml"
)


class MushafReader:
    """
    Lightweight Arabic text provider for Shahid's reasoning tools.

    Primary API:
        get_ayah(surah, ayah)         → Arabic text string
        get_ayat_range(s, start, end) → list of {surah, ayah, text}
        get_surah_info(surah)         → {name_ar, ayah_count, nozol}
        get_ayat_for_nodes(node_ids, tmq_graph) → ayat for TMQ node locations
        compare_ayat(refs)            → formatted side-by-side Arabic block
        narrative_range(s, start, end)→ contiguous ayat as narrative string
    """

    def __init__(self, txt_path: str = _DEFAULT_TXT, xml_path: str = _DEFAULT_XML):
        # {(surah, ayah): arabic_text}
        self._ayat: dict[tuple, str] = {}
        # {surah: {name_ar, ayah_count, nozol}}
        self._surah_meta: dict[int, dict] = {}

        self._load_txt(txt_path)
        self._load_xml(xml_path)

        logger.info(f"MushafReader: {len(self._ayat):,} ayat loaded")

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_txt(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"MushafReader: quran-simple.txt not found at {path}")
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    try:
                        s, a, text = int(parts[0]), int(parts[1]), parts[2]
                        self._ayat[(s, a)] = text
                    except ValueError:
                        pass

    def _load_xml(self, path: str):
        if not os.path.exists(path):
            logger.debug(f"MushafReader: mushaf.xml not found at {path} — skipping metadata")
            return
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            for sura in root.findall("Sura"):
                try:
                    sid    = int(sura.get("ID", "0"))
                    name   = sura.get("Name", "")
                    count  = int(sura.get("Nb_aya", "0"))
                    nozol  = sura.get("Nozol", "")
                    self._surah_meta[sid] = {
                        "name_ar":    name,
                        "ayah_count": count,
                        "nozol":      nozol,
                    }
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            logger.warning(f"MushafReader: XML parse failed: {e}")

    # ── Primary API ───────────────────────────────────────────────────────────

    def get_ayah(self, surah: int, ayah: int) -> str:
        """Return Arabic text for a single ayah, or empty string if not found."""
        return self._ayat.get((surah, ayah), "")

    def get_ayat_range(self, surah: int, start: int, end: int) -> list:
        """
        Return [{surah, ayah, text}, ...] for ayat start..end (inclusive)
        within a surah.
        """
        result = []
        for a in range(start, end + 1):
            text = self._ayat.get((surah, a), "")
            if text:
                result.append({"surah": surah, "ayah": a, "text": text})
        return result

    def get_surah_info(self, surah: int) -> dict:
        """Return {name_ar, ayah_count, nozol} for a surah, or empty dict."""
        return self._surah_meta.get(surah, {})

    def get_ayat_for_nodes(self, node_ids: list, tmq_graph) -> list:
        """
        Given a list of TMQ node IDs, extract their surah/verse locations
        and return the corresponding ayat, deduplicated and sorted by
        Quranic order.

        Returns [{surah, ayah, text, node_ids: [...]}, ...]
        """
        verse_to_nodes: dict[tuple, list] = {}
        for nid in node_ids:
            attrs = tmq_graph.node(nid)
            if not attrs:
                continue
            loc = attrs.get("loc", [])
            if len(loc) >= 2:
                key = (int(loc[0]), int(loc[1]))
                verse_to_nodes.setdefault(key, []).append(nid)

        result = []
        for (s, a), nids in sorted(verse_to_nodes.items()):
            text = self._ayat.get((s, a), "")
            if text:
                result.append({"surah": s, "ayah": a, "text": text, "node_ids": nids})
        return result

    def compare_ayat(self, refs: list) -> str:
        """
        Side-by-side formatted block for multiple ayat.
        refs = [(surah, ayah), ...]
        Returns multi-line Arabic text block suitable for LLM context.
        """
        if not refs:
            return "[No ayat references provided]"
        lines = []
        for s, a in refs:
            text = self._ayat.get((s, a), "")
            meta = self._surah_meta.get(s, {})
            surah_name = meta.get("name_ar", f"Surah {s}")
            if text:
                lines.append(f"[{s}:{a} — {surah_name}]")
                lines.append(text)
                lines.append("")
            else:
                lines.append(f"[{s}:{a} — not found]")
        return "\n".join(lines).strip()

    def narrative_range(self, surah: int, start: int, end: int) -> str:
        """
        Return a contiguous block of ayat as a narrative text string.
        For following stories of the Messengers across multiple verses.
        """
        ayat = self.get_ayat_range(surah, start, end)
        if not ayat:
            return f"[No ayat found for {surah}:{start}-{end}]"
        meta       = self._surah_meta.get(surah, {})
        surah_name = meta.get("name_ar", f"Surah {surah}")
        header     = f"[{surah_name} — {surah}:{start}-{end}]"
        body       = "\n".join(
            f"({a['ayah']}) {a['text']}" for a in ayat
        )
        return f"{header}\n{body}"

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def ayah_count(self) -> int:
        return len(self._ayat)

    def is_ready(self) -> bool:
        return len(self._ayat) > 0
