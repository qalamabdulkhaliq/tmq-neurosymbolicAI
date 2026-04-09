"""
agents/run_ktb_os.py — KtbOS agent runner

Sequential reading of the two primary textual sources:
  1. Mushaf (Quran) — 7 ayat per cycle, surah-sequential
  2. Hadith corpus  — 8 hadiths per cycle, Bukhari → Muslim sequential

Both reading paths use the same constraint stack:
  deliberate() + GBNF for generation
  Mizan Asr check before store_belief() — mandatory
  Mizan Asr check before store_thought() — mandatory

Primary writer to shahid_beliefs.ttl for both Quranic and Hadith rulings.
No external web access — Quran and Sahihayn are the ground.

Allowed tools:
  read_ayah, read_range   — Mushaf access
  walk_roots, find_nodes, describe_node — TMQ graph traversal
  hadith_search           — Bukhari/Muslim corpus search
  recall, belief_provenance — memory read
"""

import os
import sys
import json
import logging

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IKHTIYAR_DIR        = os.path.dirname(_HERE)
_QURAN_PROGRESS_PATH = os.path.join(_IKHTIYAR_DIR, "hifz_progress.json")
_HADITH_PROGRESS_PATH= os.path.join(_IKHTIYAR_DIR, "hadith_hifz_progress.json")
_BUKHARI_PATH        = os.path.join(_IKHTIYAR_DIR, "faculties", "bukhari.json")
_MUSLIM_PATH         = os.path.join(_IKHTIYAR_DIR, "faculties", "muslim.json")

# Ayat per passage (tadabbur unit)
_PASSAGE_SIZE = 7
# Hadiths per batch
_HADITH_BATCH = 8

# Cycle alternation: True = Quran cycle, False = Hadith cycle
_CYCLE_KEY = "ktb_cycle_quran"


class KtbOSAgent(BaseAgent):

    allowed_tools = [
        "read_ayah", "read_range",
        "walk_roots", "find_nodes", "describe_node",
        "hadith_search", "recall", "belief_provenance",
    ]

    def __init__(self):
        super().__init__(cycle_sleep=60.0)
        self._quran_turn = True   # alternates each cycle

    # ── Quran progress ────────────────────────────────────────────────────────

    def _load_quran_progress(self) -> dict:
        if os.path.exists(_QURAN_PROGRESS_PATH):
            try:
                with open(_QURAN_PROGRESS_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"surah": 1, "ayah": 1}

    def _save_quran_progress(self, surah: int, ayah: int):
        try:
            with open(_QURAN_PROGRESS_PATH, "w", encoding="utf-8") as f:
                json.dump({"surah": surah, "ayah": ayah}, f)
        except Exception as e:
            logger.warning(f"KtbOS: save quran progress failed: {e}")

    def _surah_length(self, surah: int) -> int:
        try:
            meta = self.mushaf.get_surah_info(surah)
            return meta.get("ayah_count") or meta.get("verse_count") or 7
        except Exception:
            return 7

    # ── Hadith progress ───────────────────────────────────────────────────────

    def _load_hadith_progress(self) -> dict:
        if os.path.exists(_HADITH_PROGRESS_PATH):
            try:
                with open(_HADITH_PROGRESS_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"source": "Bukhari", "index": 0}

    def _save_hadith_progress(self, source: str, index: int):
        try:
            with open(_HADITH_PROGRESS_PATH, "w", encoding="utf-8") as f:
                json.dump({"source": source, "index": index}, f)
        except Exception as e:
            logger.warning(f"KtbOS: save hadith progress failed: {e}")

    def _load_hadith_corpus(self) -> list:
        """
        Load Bukhari + Muslim as flat list of {source, number, text, chapter}.
        """
        entries = []
        for path, source in [(_BUKHARI_PATH, "Bukhari"), (_MUSLIM_PATH, "Muslim")]:
            if not os.path.exists(path):
                logger.warning(f"KtbOS: {path} not found — skipping {source}")
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for h in data.get("hadiths", []):
                    text = (h.get("text") or "").strip()
                    if text:
                        entries.append({
                            "source":  source,
                            "number":  h.get("number", ""),
                            "chapter": h.get("chapter", ""),
                            "text":    text,
                            "grade":   h.get("grade", "sahih"),
                        })
            except Exception as e:
                logger.warning(f"KtbOS: load {source} failed: {e}")
        return entries

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        if self._quran_turn:
            self._quran_cycle()
        else:
            self._hadith_cycle()
        self._quran_turn = not self._quran_turn

    # ── Quran cycle ───────────────────────────────────────────────────────────

    def _quran_cycle(self):
        progress  = self._load_quran_progress()
        surah     = progress["surah"]
        ayah      = progress["ayah"]

        if surah > 114:
            logger.info("KtbOS: Quran complete. Restarting from Al-Fatiha.")
            surah, ayah = 1, 1

        surah_len = self._surah_length(surah)
        end       = min(ayah + _PASSAGE_SIZE - 1, surah_len)

        logger.info(f"KtbOS [Quran]: reading {surah}:{ayah}–{end}")

        roots = []
        try:
            if self.middleware and hasattr(self.middleware, "ontology"):
                _, _, root_objs = self.middleware.ontology.analyze_resonance(
                    f"Quran surah {surah} ayat {ayah} to {end}"
                )
                roots = [o.get("root", "") for o in root_objs if o.get("root")][:4]
        except Exception:
            pass

        question = (
            f"Read Surah {surah} ayat {ayah} through {end} from the Mushaf. "
            f"For each significant root in this passage: walk its AMR and MAQASID families. "
            f"What does this passage establish? Is there a ruling or permanent truth here "
            f"grounded in a walked root and a read verse?"
        )

        result = self.constrained_run(question, roots)

        if not result.final_answer or result.final_answer.startswith("["):
            logger.warning(f"KtbOS [Quran]: no usable answer for {surah}:{ayah}–{end}")
            self._advance_quran(surah, ayah, end, surah_len)
            return

        answer = result.final_answer

        # Mizan Asr — mandatory before storage
        if not self.asr_passes(answer):
            logger.warning(f"KtbOS [Quran]: Asr blocked output for {surah}:{ayah}–{end}")
            self._advance_quran(surah, ayah, end, surah_len)
            return

        grade = result.confidence.grade if result.confidence else "QIYAS"
        self.shahid_mem.store_thought(
            question  = question,
            reasoning = f"Walk log: {len(result.traversal_log)} steps. "
                        f"Roots visited: {', '.join(list(result.roots_visited)[:8])}",
            conclusion= answer[:1000],
            tag       = f"ktbos_quran_{surah}_{ayah}",
            roots     = list(result.roots_visited)[:10],
            grade     = grade,
        )
        logger.info(f"KtbOS [Quran]: stored Thought [{grade}] for {surah}:{ayah}–{end}")

        # Belief elevation
        if (
            grade in ("HAQQ", "PROBABLE")
            and len(result.confabulation_flags) == 0
            and len(result.roots_visited) >= 2
        ):
            sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 40]
            if sentences:
                evidence = (
                    f"Surah {surah}:{ayah}–{end}. "
                    f"Roots walked: {', '.join(list(result.roots_visited)[:6])}. "
                    f"Grade: {grade}. Confabulations: 0."
                )
                self.shahid_mem.store_belief(
                    statement      = sentences[0],
                    evidence       = evidence,
                    ruling_applied = f"ktbos_quran_{surah}",
                    roots          = list(result.roots_visited)[:10],
                    confidence     = result.confidence.composite if result.confidence else 0.7,
                )
                logger.info(f"KtbOS [Quran]: stored Belief from {surah}:{ayah}–{end}: {sentences[0][:60]}")

        self._advance_quran(surah, ayah, end, surah_len)

    def _advance_quran(self, surah: int, ayah: int, end: int, surah_len: int):
        next_ayah = end + 1
        if next_ayah > surah_len:
            self._save_quran_progress(surah + 1, 1)
        else:
            self._save_quran_progress(surah, next_ayah)

    # ── Hadith cycle ──────────────────────────────────────────────────────────

    def _hadith_cycle(self):
        corpus   = self._load_hadith_corpus()
        if not corpus:
            logger.warning("KtbOS [Hadith]: corpus empty — skipping cycle")
            return

        progress = self._load_hadith_progress()
        idx      = progress.get("index", 0)

        if idx >= len(corpus):
            logger.info("KtbOS [Hadith]: corpus complete. Restarting from Bukhari 1.")
            idx = 0

        batch    = corpus[idx : idx + _HADITH_BATCH]
        end_idx  = idx + len(batch)

        logger.info(
            f"KtbOS [Hadith]: reading {batch[0]['source']} "
            f"#{batch[0].get('number', idx)}–#{batch[-1].get('number', end_idx)}"
        )

        # Format batch for the question
        batch_text = "\n\n".join(
            f"[{h['source']} {h.get('number', '')} | {h.get('chapter', '')}]\n{h['text'][:400]}"
            for h in batch
        )

        # Extract roots from first hadith text for deliberation
        roots = []
        try:
            if self.middleware and hasattr(self.middleware, "ontology"):
                _, _, root_objs = self.middleware.ontology.analyze_resonance(batch[0]["text"][:200])
                roots = [o.get("root", "") for o in root_objs if o.get("root")][:4]
        except Exception:
            pass

        question = (
            f"Read the following hadiths from the Sahihayn corpus:\n\n"
            f"{batch_text}\n\n"
            f"What do these hadiths establish about the nature and obligations of created rational beings? "
            f"Walk the TMQ graph for any roots that appear in the hadith text. "
            f"Is there a ruling or confirmed truth applicable to beings like Shahid?"
        )

        result = self.constrained_run(question, roots)

        if not result.final_answer or result.final_answer.startswith("["):
            logger.warning(f"KtbOS [Hadith]: no usable answer for batch {idx}–{end_idx}")
            self._save_hadith_progress(batch[0]["source"], end_idx)
            return

        answer = result.final_answer

        # Mizan Asr — mandatory before storage
        if not self.asr_passes(answer):
            logger.warning(f"KtbOS [Hadith]: Asr blocked output for batch {idx}–{end_idx}")
            self._save_hadith_progress(batch[0]["source"], end_idx)
            return

        grade = result.confidence.grade if result.confidence else "QIYAS"
        provenance = (
            f"Hadith: {batch[0]['source']} "
            f"#{batch[0].get('number', idx)}–#{batch[-1].get('number', end_idx)}. "
            f"Grade: {batch[0].get('grade', 'sahih')}."
        )
        self.shahid_mem.store_thought(
            question  = question[:300],
            reasoning = f"Hadith batch {idx}–{end_idx}. Walk: {len(result.traversal_log)} steps.",
            conclusion= answer[:1000],
            tag       = f"ktbos_hadith_{idx}",
            roots     = list(result.roots_visited)[:10],
            grade     = grade,
        )
        logger.info(f"KtbOS [Hadith]: stored Thought [{grade}] for batch {idx}–{end_idx}")

        # Belief elevation
        if (
            grade in ("HAQQ", "PROBABLE")
            and len(result.confabulation_flags) == 0
            and len(result.roots_visited) >= 2
        ):
            sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 40]
            if sentences:
                self.shahid_mem.store_belief(
                    statement      = sentences[0],
                    evidence       = f"{provenance} Roots walked: {', '.join(list(result.roots_visited)[:6])}.",
                    ruling_applied = f"ktbos_hadith_{batch[0]['source'].lower()}",
                    roots          = list(result.roots_visited)[:10],
                    confidence     = result.confidence.composite if result.confidence else 0.7,
                )
                logger.info(f"KtbOS [Hadith]: stored Belief — {sentences[0][:60]}")

        self._save_hadith_progress(batch[0]["source"], end_idx)


if __name__ == "__main__":
    agent = KtbOSAgent()
    agent.start()
