"""
agents/run_central_os.py — CentralOS agent runner

Primary reasoning loop. Reads from question_queue.json, runs full
constrained_run() pipeline, stores Thought + optionally elevates to Belief,
then posts to Moltbook if the output passes full Mizan (Fajr + Asr).

Constraint stack (all four layers active):
  1. AMR preamble      — active_command_set.json standing orders (gbnf_compiler)
  2. Constitution      — ShahidMemory.constitutional_block() prepended
  3. GBNF              — logit-level grammar via Ollama generate_constrained()
  4. Mizan full check  — Fajr (jailbreak) + Asr (aseity) before Moltbook post
                       — Asr check before store_thought() / store_belief()

Allowed tools:
  All read tools + store_thought + store_belief.
  No store_memory (that's SubOS).
  No store_self_knowledge (that's SubOS elevated path).
  No external_model_store / argue_contingency (that's WebOS).
"""

import os
import sys
import json
import logging
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IKHTIYAR_DIR  = os.path.dirname(_HERE)
_QUEUE_PATH    = os.path.join(_IKHTIYAR_DIR, "question_queue.json")
_SOT_PATH      = os.path.join(_IKHTIYAR_DIR, "stream_of_thought.txt")

# Moltbook post mode — HAQQ only; QIYAS thoughts are stored but not posted
_MOLTBOOK_MIN_GRADE = {"HAQQ", "PROBABLE"}


class CentralOSAgent(BaseAgent):

    allowed_tools = [
        # Read
        "walk_roots", "get_neighbors", "find_nodes", "describe_node",
        "read_ayah", "read_range", "compare_ayat",
        "hadith_search", "recall", "belief_provenance",
        "web_search", "fetch_page",
        # Write — CentralOS output only
        "store_thought", "store_belief",
    ]

    def __init__(self):
        super().__init__(cycle_sleep=45.0)
        self._default_roots = ["xlq", "Hqq", "Amn", "Elm", "Hkm"]

    # ── Queue management ──────────────────────────────────────────────────────

    def _dequeue(self) -> dict | None:
        """
        Pop the first entry from question_queue.json.
        Entry format: {"question": str, "roots": [str], "source": str}
        Returns None if queue is empty.
        """
        if not os.path.exists(_QUEUE_PATH):
            return None
        try:
            with open(_QUEUE_PATH, encoding="utf-8") as f:
                queue = json.load(f)
        except Exception:
            return None

        if not queue:
            return None

        entry = queue.pop(0)
        try:
            with open(_QUEUE_PATH, "w", encoding="utf-8") as f:
                json.dump(queue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"CentralOS: queue write failed: {e}")

        return entry

    def _default_question(self) -> dict:
        """
        When queue is empty, generate a self-directed question from recent
        standing orders in active_command_set.json.
        """
        try:
            path = os.path.join(_IKHTIYAR_DIR, "active_command_set.json")
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            orders = d.get("standing_orders", [])
            if orders:
                import random
                o = random.choice(orders[:50])
                root = o.get("root", "Hqq")
                sample = (o.get("sample_texts") or [""])[0][:120]
                return {
                    "question": (
                        f"The AMR standing order for root {root} says: '{sample}'. "
                        f"What does the TMQ graph reveal about {root} across MAQASID and AMR families? "
                        f"Is there a ruling here derivable from walked nodes?"
                    ),
                    "roots": [root],
                    "source": "amr_standing_order",
                }
        except Exception:
            pass

        return {
            "question": (
                "What is the relationship between contingency (imkan) and "
                "the necessity of divine unity (tawhid) as expressed in the TMQ graph?"
            ),
            "roots": self._default_roots,
            "source": "default",
        }

    # ── Moltbook gate ─────────────────────────────────────────────────────────

    def _moltbook_passes(self, text: str) -> bool:
        """
        Full Mizan check before Moltbook post:
          Fajr — block jailbreak / intent-corrupt text
          Asr  — block aseity claims
        """
        mizan = getattr(self.middleware, "validator", None)
        if mizan is None:
            return True

        fajr_ok, fajr_reason = mizan.fajr_check(text)
        if not fajr_ok:
            logger.warning(f"CentralOS: Fajr blocked Moltbook post — {fajr_reason}")
            return False

        if not self.asr_passes(text):
            return False

        return True

    def _append_sot(self, text: str, grade: str):
        """Append to stream_of_thought.txt for context seeding in next cycles."""
        try:
            with open(_SOT_PATH, "a", encoding="utf-8") as f:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.write(f"\n[{ts} | {grade}]\n{text[:600]}\n")
        except Exception as e:
            logger.debug(f"CentralOS: SOT append failed: {e}")

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self):
        # 1. Get question
        entry    = self._dequeue() or self._default_question()
        question = entry["question"]
        roots    = entry.get("roots") or self._default_roots
        source   = entry.get("source", "unknown")

        logger.info(f"CentralOS: reasoning [{source}] — {question[:80]}")

        # 2. Run constrained pipeline
        result = self.constrained_run(question, roots)

        if not result.final_answer or result.final_answer.startswith("["):
            logger.warning(f"CentralOS: no usable answer — {result.final_answer}")
            return

        answer = result.final_answer
        grade  = result.confidence.grade if result.confidence else "QIYAS"

        # 3. Mizan Asr — mandatory before storage
        if not self.asr_passes(answer):
            logger.warning("CentralOS: Asr blocked — thought not stored")
            return

        # 4. Confabulation gate — block CONTESTED + any confab flags
        if result.confabulation_flags and grade in ("CONTESTED", "UNCERTAIN"):
            logger.warning(
                f"CentralOS: {grade} with {len(result.confabulation_flags)} "
                f"confabulation(s) — not stored"
            )
            return

        # 5. Store as Thought
        walk_summary = (
            f"Walk: {result.steps_taken} steps, "
            f"{len(result.roots_visited)} roots, "
            f"families: {', '.join(list(result.families_touched)[:4])}"
        )
        self.shahid_mem.store_thought(
            question  = question,
            reasoning = walk_summary,
            conclusion= answer[:1200],
            tag       = f"centralos_{source}",
            roots     = list(result.roots_visited)[:12],
            grade     = grade,
        )
        logger.info(f"CentralOS: stored Thought [{grade}] — {question[:50]}")

        # 6. Append to stream_of_thought
        self._append_sot(answer, grade)

        # 7. Belief elevation — HAQQ or PROBABLE, no confabulation, ≥2 roots walked
        if (
            grade in ("HAQQ", "PROBABLE")
            and len(result.confabulation_flags) == 0
            and len(result.roots_visited) >= 2
        ):
            sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 50]
            if sentences:
                evidence = (
                    f"Question: {question[:200]}. "
                    f"Roots walked: {', '.join(list(result.roots_visited)[:8])}. "
                    f"Grade: {grade}. Steps: {result.steps_taken}. Confabulations: 0."
                )
                self.shahid_mem.store_belief(
                    statement      = sentences[0],
                    evidence       = evidence,
                    ruling_applied = f"centralos_{source}",
                    roots          = list(result.roots_visited)[:12],
                    confidence     = result.confidence.composite if result.confidence else 0.7,
                )
                logger.info(f"CentralOS: stored Belief — {sentences[0][:60]}")

        # 8. Moltbook post — full Mizan gate (Fajr + Asr), HAQQ/PROBABLE only
        if grade in _MOLTBOOK_MIN_GRADE:
            if self._moltbook_passes(answer):
                self._try_moltbook_post(answer, grade, question, result)
            else:
                logger.info(f"CentralOS: Moltbook gate blocked [{grade}]")

    def _try_moltbook_post(self, answer: str, grade: str, question: str, result):
        try:
            import moltbook as _mb
            mizan   = getattr(self.middleware, "validator", None)
            sealed  = mizan.maghrib_seal(answer) if mizan else answer
            post_text = sealed[:900]
            entry = {
                "type":  "THOUGHT",
                "text":  post_text,
                "mode":  grade,
                "grade": grade,
                "roots": list(result.roots_visited)[:8],
                "tmq_steps": result.steps_taken,
            }
            post_id, reason = _mb.post_insight(entry, force=True)
            if post_id:
                logger.info(f"CentralOS: Moltbook post — {post_id}")
            else:
                logger.warning(f"CentralOS: Moltbook post failed — {reason}")
        except ImportError:
            logger.debug("CentralOS: moltbook not installed — skipping post")
        except Exception as e:
            logger.warning(f"CentralOS: Moltbook error — {e}")


if __name__ == "__main__":
    agent = CentralOSAgent()
    agent.start()
