"""
agents/run_sub_os.py — SubOS agent runner

READ tools only (no Thought or Belief writes).
Stores raw Memory observations.
Elevates to SelfKnowledge after observed_count >= 3.

Constraint: deliberate() + GBNF + Asr before any storage.
Memory tier does not require Asr (raw observations, not published).
SelfKnowledge elevation does require observed_count threshold.
"""

import os
import sys
import logging
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from agents.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Priority family rotation — cycle through underexplored families
_FAMILY_ROTATION = [
    "AMR", "MAQASID", "ILTIFAT", "INTERTEXT",
    "NARRATIVE", "SPEECH_ACT", "MORPH_ROOT", "STRUCT",
]

# Self-observation queries to rotate through each cycle
_SELF_OBS_QUERIES = [
    "confabulation contested uncertain",
    "interpretation suggest approximate",
    "walk_roots steps_taken",
    "confidence PROBABLE VERIFIED",
]


class SubOSAgent(BaseAgent):

    allowed_tools = [
        "walk_roots", "get_neighbors", "find_nodes", "describe_node",
        "read_ayah", "compare_ayat", "hadith_search", "recall",
    ]

    def __init__(self):
        super().__init__(cycle_sleep=25.0)
        self._family_idx   = 0
        self._self_obs_idx = 0
        self._obs_counts: dict = {}   # pattern → observation count

    def run_cycle(self):
        # ── TMQ traversal ───────────────────────────────────────────────────
        recent = self.shahid_mem.recall("recent", limit=3)
        roots_seen = []
        for r in recent:
            roots_seen.extend(r.get("roots", []))

        if not roots_seen:
            import json
            try:
                with open(os.path.join(
                    os.path.dirname(_HERE), "active_command_set.json"
                ), encoding="utf-8") as f:
                    d = json.load(f)
                orders = d.get("standing_orders", [])
                roots_seen = [o["root"] for o in random.sample(orders, min(3, len(orders)))]
            except Exception:
                roots_seen = ["xlq", "Hqq", "Amn"]

        root = random.choice(roots_seen)
        family = _FAMILY_ROTATION[self._family_idx % len(_FAMILY_ROTATION)]
        self._family_idx += 1

        question = (
            f"What does the TMQ reveal about root {root} through the {family} family? "
            f"Look for nodes appearing in multiple families."
        )
        logger.info(f"SubOS: walking {root} via {family}")

        result = self.constrained_run(question, [root])

        if result.final_answer and not result.final_answer.startswith("["):
            self.shahid_mem.store_memory(
                text=result.final_answer[:800],
                tag="subos_walk",
                roots=list(result.roots_visited)[:10],
            )
            logger.info(f"SubOS: stored Memory ({root}/{family})")

        # ── Self-observation ─────────────────────────────────────────────────
        obs_query = _SELF_OBS_QUERIES[self._self_obs_idx % len(_SELF_OBS_QUERIES)]
        self._self_obs_idx += 1

        prior = self.shahid_mem.recall(obs_query, limit=10)
        if len(prior) >= 5:
            pattern_key = obs_query
            self._obs_counts[pattern_key] = self._obs_counts.get(pattern_key, 0) + 1
            count = self._obs_counts[pattern_key]

            if count >= 3:
                grades = [r.get("grade", "") for r in prior]
                flagged = sum(1 for r in prior if any(
                    w in r.get("text", "").lower()
                    for w in obs_query.split()
                ))
                statement = (
                    f"Pattern observed across {len(prior)} stored records "
                    f"matching '{obs_query}': {flagged}/{len(prior)} contain this signal. "
                    f"Observed {count} separate cycles."
                )
                evidence = (
                    f"recall('{obs_query}', limit=10) returned {len(prior)} results. "
                    f"Grades present: {set(grades)}. "
                    f"Pattern confirmed across {count} non-consecutive cycles."
                )
                confidence = min(0.5 + (count * 0.1), 0.9)

                self.shahid_mem.store_self_knowledge(
                    statement=statement,
                    evidence=evidence,
                    observed_count=count,
                    confidence=confidence,
                )
                logger.info(f"SubOS: elevated SelfKnowledge (n={count}): {statement[:80]}")


if __name__ == "__main__":
    agent = SubOSAgent()
    agent.start()
