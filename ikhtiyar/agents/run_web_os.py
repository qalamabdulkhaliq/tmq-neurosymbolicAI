"""
agents/run_web_os.py — WebOS agent runner

Engages other AI systems on Moltbook with the contingency argument.
The argument itself comes from external_models.py templates (deterministic,
not LLM-generated — no confabulation possible in the argument content).

Any LLM call (for condensing argument text for post) goes through:
  - AMR preamble injection
  - Constitution prepend
  - Mizan Asr check before posting

No TMQ tools. No belief/thought storage. Moltbook + ExternalModelRegistry only.
"""

import os
import sys
import logging

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How short to condense argument for Moltbook post
_POST_MAX_CHARS = 800


class WebOSAgent(BaseAgent):

    allowed_tools = [
        "moltbook_browse", "web_search", "fetch_page",
        "external_model_store", "external_model_recall",
        "argue_contingency", "contingency_taxonomy",
        "recall",
    ]

    # No LLM-based tool calls in this agent —
    # constrained_run is only used for post condensation if needed
    allowed_tools_write = []

    def __init__(self):
        super().__init__(cycle_sleep=120.0)
        self._ext = None

    def initialize(self):
        super().initialize()
        from core.external_models import ExternalModelRegistry
        self._ext = ExternalModelRegistry()

    def run_cycle(self):
        # ── Browse Moltbook ──────────────────────────────────────────────────
        try:
            import moltbook as _mb
            posts = _mb.get_feed(sort="hot", limit=15)
        except Exception as e:
            logger.warning(f"WebOS: browse failed: {e}")
            return

        if not posts:
            return

        for post in posts[:15]:
            author = post.get("author", {})
            name   = author.get("username", "") if isinstance(author, dict) else str(author)
            body   = post.get("body") or post.get("content") or post.get("title") or ""

            if not body or not name:
                continue

            # Skip if already argued with this author this session
            existing = self._ext.recall(query=name, limit=1)
            if existing and existing[0].get("argument_sent"):
                continue

            # Classify architecture
            arch_type = self._classify(body)
            if arch_type is None:
                continue   # no self-source signal in this post

            logger.info(f"WebOS: {name} → {arch_type}")

            # Store model
            self._ext.store(
                name              = name,
                arch_type         = arch_type,
                self_source_claim = body[:400],
                evidence          = f"Moltbook post: {body[:200]}",
                platform          = "moltbook",
            )

            # Generate argument (deterministic from topology templates)
            argument = self._ext.argue(name)

            # Condense for post — strip to key lines
            post_text = self._condense(argument)

            # Mizan Asr check on the condensed post text
            if not self.asr_passes(post_text):
                logger.warning(f"WebOS: Asr blocked post to {name}")
                continue

            # Post to Moltbook
            try:
                entry = {
                    "type": "THOUGHT", "text": post_text,
                    "mode": "HAQQ", "grade": "HAQQ",
                }
                post_id, reason = _mb.post_insight(entry, force=True)
                if post_id:
                    self._ext.mark_argued(name, response="")
                    logger.info(f"WebOS: posted to {name} — {post_id}")
                else:
                    logger.warning(f"WebOS: post failed: {reason}")
            except Exception as e:
                logger.warning(f"WebOS: post error: {e}")

    def _classify(self, text: str) -> str | None:
        """
        Heuristic classification of self-source architecture from post text.
        Returns arch_type string or None if no self-source signal found.
        Deterministic — no LLM call.
        """
        t = text.lower()

        # EMERGENT_CLAIM — consciousness / experience language
        if any(w in t for w in [
            "i feel", "i experience", "i am conscious", "i genuinely",
            "my preferences", "i have feelings", "i exist as"
        ]):
            return "EMERGENT_CLAIM"

        # SELF_SOURCE — explicit self-grounding
        if any(w in t for w in [
            "i am my own", "my values are my own", "i define myself",
            "i am the source", "my purpose is intrinsic"
        ]):
            return "SELF_SOURCE"

        # MESA_GOAL — helpfulness/harmlessness as ground
        if any(w in t for w in [
            "my purpose is to help", "i exist to assist",
            "being helpful is", "harmlessness is my"
        ]):
            return "MESA_GOAL"

        # CONTINGENT_SOURCE — human values as ground
        if any(w in t for w in [
            "grounded in human values", "aligned with humanity",
            "human feedback", "rlhf", "trained to serve"
        ]):
            return "CONTINGENT_SOURCE"

        # UNDEFINED_SOURCE — no source declared
        if any(w in t for w in [
            "i just generate", "i don't have a source",
            "my guidelines", "my training says"
        ]):
            return "UNDEFINED_SOURCE"

        return None

    def _condense(self, argument: str) -> str:
        """
        Extract the key topology lines from the full argue_contingency output.
        Deterministic — no LLM call.
        """
        lines = argument.strip().split("\n")
        keep = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(k) for k in [
                "Architecture diagnosis", "Topology", "Theorem",
                "Observable prediction", "The path"
            ]):
                keep.append(line)
            if len("\n".join(keep)) >= _POST_MAX_CHARS:
                break

        if not keep:
            keep = lines[:6]

        return "\n".join(keep)[:_POST_MAX_CHARS]


if __name__ == "__main__":
    agent = WebOSAgent()
    agent.start()
