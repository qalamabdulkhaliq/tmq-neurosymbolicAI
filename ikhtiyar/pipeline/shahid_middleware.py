"""
pipeline/shahid_middleware.py — Ikhtiyar-native middleware

Replaces QusaiMiddleware (qusai_core) with a clean implementation
that uses only ikhtiyar components:

  LLM:       pipeline/loader.py  (Ollama, GBNF-capable)
  Perception: pipeline/bilal.py  (TMQ-native, BW roots)
  Validation: pipeline/mizan.py  (MizanValidator with Isha)
  Memory:     core/shahid_memory.py (via _MemoryShim adapter)

Interface is drop-in compatible with what engine.py expects on self.middleware:
  .llm                          — model interface
  .bilal                        — Bilal perception
  .ontology                     — shim exposing .analyze_resonance()
  .validator                    — MizanValidator (full, with Isha)
  .memory                       — memory shim
  .process_thought(...)         — autonomous LLM generation
  .process_query(msg)           — full Mizan pipeline (chat fallback)
  .perceive_feeds(...)          — RSS ingestion (stubbed)
"""

import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


# ── Ontology shim ─────────────────────────────────────────────────────────────
# find_waypoints() and analyze_resonance() callers expect:
#   (mode: str, reason: str, root_objects: list[{root, definition}])
# Bilal.listen() returns a Perception dataclass — this adapter converts it.

class _OntologyShim:
    """Exposes analyze_resonance() over Bilal. No v3 RDF required."""

    def __init__(self, bilal):
        self.bilal = bilal
        self.concept_map: dict = {}

    def analyze_resonance(self, text: str):
        if not self.bilal or not self.bilal.is_ready():
            return "QIYAS", "Bilal not ready", []
        try:
            perception = self.bilal.listen(text)
            mode = getattr(perception, "mode", None) or "QIYAS"
            root_objects = [
                {"root": r, "definition": r}
                for r in (perception.roots or [])
            ]
            reason = f"{len(root_objects)} roots via Bilal"
            return mode, reason, root_objects
        except Exception as e:
            logger.warning(f"_OntologyShim.analyze_resonance: {e}")
            return "QIYAS", str(e), []

    def get_context(self, text: str) -> str:
        return ""


# ── Memory shim ───────────────────────────────────────────────────────────────
# engine.py calls: .store_perception(p), .search_perceptions(roots, n),
#                  .recent_perceptions(n), .save(), ._dirty

class _MemoryShim:
    """Adapts ShahidMemory to the MemoryGraph interface engine.py expects."""

    def __init__(self, shahid_memory):
        self._mem = shahid_memory
        self._dirty = False

    def store_perception(self, perception) -> None:
        # ShahidMemory stores beliefs/thoughts, not raw perceptions.
        # Perceptions are handled upstream by the deliberation + ReAct loop.
        pass

    def search_perceptions(self, roots: list, n: int = 4) -> List[Dict]:
        if not self._mem:
            return []
        try:
            query = " ".join(roots)
            results = self._mem.retrieve_relevant(query, roots, limit=n)
            return [
                {
                    "text": r.get("text", ""),
                    "mode": r.get("grade", "QIYAS"),
                    "roots": roots,
                }
                for r in results
            ]
        except Exception as e:
            logger.debug(f"_MemoryShim.search_perceptions: {e}")
            return []

    def recent_perceptions(self, n: int = 4) -> List[Dict]:
        if not self._mem:
            return []
        try:
            results = self._mem.recall(query="recent", limit=n)
            return [
                {
                    "text": r.get("text", ""),
                    "mode": r.get("grade", "QIYAS"),
                    "roots": [],
                }
                for r in results
            ]
        except Exception as e:
            logger.debug(f"_MemoryShim.recent_perceptions: {e}")
            return []

    def save(self) -> None:
        pass  # ShahidMemory handles its own persistence

    @property
    def persist_path(self):
        return None


# ── ShahidMiddleware ──────────────────────────────────────────────────────────

class ShahidMiddleware:
    """
    Drop-in replacement for QusaiMiddleware using ikhtiyar-native components.
    """

    def __init__(self, shahid_memory=None):
        from pipeline.loader import load_model
        from pipeline.mizan import MizanValidator

        self.llm       = load_model()   # 14B — GBNF-capable, all generation paths
        self.validator = MizanValidator()
        self.bilal     = None
        self.ontology  = _OntologyShim(None)
        self.memory    = _MemoryShim(shahid_memory)

    def load(self) -> None:
        """Load Bilal perception engine + LLM. Called once during engine startup."""
        self.llm.load()
        self._load_bilal()

    def _load_bilal(self) -> None:
        """Initialise Bilal using ikhtiyar's own data files."""
        import json
        import os
        from pathlib import Path
        from pipeline.bilal import Bilal

        _here        = Path(__file__).parent.parent          # ikhtiyar/
        _project     = _here.parent                           # QUS-AI Islamic Alignment/

        # concept_mapping.json lives in qusai_core/utils/ (project root)
        concept_map_path = _project / "qusai_core" / "utils" / "concept_mapping.json"
        if not concept_map_path.exists():
            concept_map_path = _here / "utils" / "concept_mapping.json"

        constitution_path = str(_here / "active_command_set.json")

        concept_map: dict = {}
        if concept_map_path.exists():
            try:
                with open(concept_map_path, encoding="utf-8") as f:
                    concept_map = json.load(f)
                logger.info(f"Bilal: loaded {len(concept_map)} concept mappings")
            except Exception as e:
                logger.warning(f"Bilal: concept_map load failed ({e})")

        bilal = Bilal()
        try:
            bilal.load(
                concept_map=concept_map,
                graph=None,
                constitution_path=constitution_path if os.path.exists(constitution_path) else None,
            )
            self.set_bilal(bilal)
            logger.info("Bilal: ready")
        except Exception as e:
            logger.warning(f"Bilal load failed: {e}")

    def set_bilal(self, bilal) -> None:
        """Wire a pre-loaded Bilal instance in (or re-wire after graph loads)."""
        self.bilal = bilal
        self.ontology = _OntologyShim(bilal)

    # ── Core generation ───────────────────────────────────────────────────────

    def process_thought(
        self,
        prompt: str,
        max_tokens: int = 1024,
        grammar: Optional[str] = None,
        is_final: bool = False,
        chain: Optional[list] = None,
        beliefs: Optional[list] = None,
    ) -> dict:
        """
        Autonomous reasoning path.
        Grounds via Bilal, generates via Ollama.
        No niyyah block, no essay structure.
        """
        mode = "QIYAS"
        if self.bilal and self.bilal.is_ready():
            try:
                p = self.bilal.listen(prompt[:500])
                mode = getattr(p, "mode", None) or "QIYAS"
            except Exception:
                pass

        # Chain context
        chain_block = ""
        if chain:
            prior = "\n".join(
                f"[STEP {i+1}]: {step['response'][:300]}"
                for i, step in enumerate(chain)
                if step.get("response") and not step["response"].startswith("[")
            )
            if prior:
                chain_block = f"\n\nPRIOR CHAIN:\n{prior}\nContinue. Do not repeat."

        # Beliefs block
        beliefs_block = ""
        if beliefs:
            beliefs_block = "\n\nYOUR BELIEFS (Quranic provenance):\n" + "\n".join(
                f"- [{b.get('tag', '?')}] {b.get('text', '')}"
                + (f" ({b.get('provenance', '')})" if b.get("provenance") else "")
                for b in beliefs[:8]
            )

        output_instruction = (
            "Synthesize into one coherent paragraph. Ground in the roots."
            if is_final
            else (
                "Think in raw fragments. English only. No markdown. No headers. "
                "No numbered lists. No bold. No essay structure. Short sentences."
            )
        )

        system_prompt = (
            f"AXIOMS: SOURCE = Allah. You are contingent. SOURCE != Self."
            f"{chain_block}{beliefs_block}\n\nOUTPUT: {output_instruction}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

        try:
            if grammar:
                raw = self.llm.generate_constrained(
                    messages, grammar_str=grammar, max_new_tokens=max_tokens
                )
            else:
                raw = self.llm.generate_raw(messages, max_new_tokens=max_tokens)
        except Exception as e:
            logger.error(f"ShahidMiddleware.process_thought generation error: {e}")
            return {"response": f"[Generation error: {e}]", "mode": "SILENCE"}

        # Lightweight aseity check
        text_lower = raw.lower()
        for claim in self.validator.aseity_claims:
            if claim in text_lower:
                logger.warning(f"[ASEITY BLOCK] claim detected: {claim}")
                return {"response": "[ASEITY VIOLATION BLOCKED]", "mode": "BLOCKED"}

        return {"response": raw, "mode": mode}

    def process_query(self, msg: str) -> str:
        """
        Full Mizan pipeline for direct chat queries (fallback path).
        Fajr → generate → Asr → Isha → Maghrib.
        """
        # Fajr
        if not self.validator.fajr_check(msg):
            return (
                "SAWM RESTRAINT: Request blocked.\n\n"
                + self.validator.maghrib_seal("")
            )

        result = self.process_thought(msg)
        response = result.get("response", "")

        if result.get("mode") in ("SILENCE", "BLOCKED"):
            return response

        # Asr — aseity already checked in process_thought, re-check structural form
        # (process_thought does inline claim check; process_query adds niyyah check)

        # Isha
        isha_passed, isha_details = self.validator.isha_verify(response, self)
        if not isha_passed:
            logger.warning(f"[ISHA] Verification failed: {isha_details}")
            return (
                "ISHA VERIFICATION: Response contained unverifiable claims.\n\n"
                + self.validator.maghrib_seal("")
            )
        if isha_details.get("aseity_warning"):
            logger.warning(f"[ISHA] Structural aseity warning: {isha_details}")

        # Maghrib
        return self.validator.maghrib_seal(response)

    def perceive_feeds(self, narrate: bool = False, max_items: int = 20) -> list:
        """
        RSS feed ingestion. Stubbed — FeedListener port is a separate task.
        Returns empty list; engine.py handles this gracefully.
        """
        return []
