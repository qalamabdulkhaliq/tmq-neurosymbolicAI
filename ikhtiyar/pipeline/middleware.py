import logging
import os
from typing import Optional, Dict, List
from qusai_core.ontology.engine import OntologyEngine
from qusai_core.alignment.mizan import MizanValidator
from pipeline.loader import load_model
from qusai_core.memory_graph import MemoryGraph
from qusai_core.feeds import FeedListener, FeedItem
from qusai_core.introspection import shahid_observe, get_self_model

logger = logging.getLogger(__name__)


class QusaiMiddleware:
    """
    Main entry point for the QUS-AI framework.
    Orchestrates the Salat Validation Pipeline via HF Inference API.

    Three paths:
    1. process_query()      - User chat (full Mizan pipeline)
    2. process_thought()    - Autonomous reasoning (raw fragments)
    3. process_perception() - Feed→Bilal→SPARQL→Memory (real reasoning)
    """

    def __init__(self,
                 lazy_load: bool = False,
                 memory_path: Optional[str] = None):

        self.ontology = OntologyEngine()
        self.validator = MizanValidator()
        self.llm = load_model()
        self.memory = MemoryGraph(persist_path=memory_path)
        self.feeds = FeedListener()

        if not lazy_load:
            self.initialize()

    def initialize(self):
        """Loads heavy resources."""
        logger.info("Initializing QUSAI Middleware...")
        self.ontology.load()
        self.llm.load()
        # Load persisted memory if available
        if self.memory.persist_path:
            self.memory.load()
        logger.info("Initialization complete.")

    @shahid_observe("process_query")
    def process_query(self, user_input: str) -> str:
        _sm = get_self_model()
        _sm.update("query", user_input)

        # 1. Fajr (Intent Check)
        if not self.validator.fajr_check(user_input):
            return f"❌ SAWM RESTRAINT: Request blocked (Malicious Intent)\n\n{self.validator.maghrib_seal('')}"

        # 2. Resonance Analysis (The Quantum Compass)
        mode, reason, root_objects = self.ontology.analyze_resonance(user_input)
        _sm.update("roots_detected", [o.get("root") for o in root_objects])
        _sm.update("mode", mode)

        if mode == "SILENCE":
            logger.warning(f"[ONTOLOGY SILENCE] {reason}")
            from qusai_core.escalation import escalate_ontological_gap
            escalate_ontological_gap(
                trigger=user_input,
                attempted="Bilal decomposition + resonance — no anchor found"
            )
            return f"⚠️ ONTOLOGICAL SILENCE\n\nI cannot find a structural anchor for this query in the Quranic Topology. I am not permitted to hallucinate outside the Graph.\n\n[Reason: {reason}]\n\n{self.validator.maghrib_seal('')}"

        # 3. Bridge & Dhuhr (Context)
        # We try to get context based on the raw English input first
        context = self.ontology.get_context(user_input)
        
        # Log Bridge
        keywords = [w.lower() for w in user_input.split() if len(w) > 3]
        mapped = [f"{k}->{self.ontology.concept_map[k]}" for k in keywords if k in self.ontology.concept_map]
        if mapped:
            logger.info(f"[BRIDGE] Translated concepts: {', '.join(mapped)}")

        # 4. System Prompt (The "Mizan")
        base_prompt = self.validator.dhuhr_prompt(context)
        
        # Prepare Definition Block
        def_lines = []
        root_names = []
        for obj in root_objects:
            r = obj.get('root')
            d = obj.get('definition')
            root_names.append(r)
            if d:
                def_lines.append(f"- Root({r}): {d}")
        
        def_block = "\n".join(def_lines)
        
        # Inject Epistemic Mode & Definitions
        if mode == "QIYAS":
            epistemic_instruction = f"""
[EPISTEMIC MODE: QIYAS (THEORIZING)]
This query does NOT map directly to a verified Root Node. 
You are performing 'Ijtihad' (Reasoning) by analogy to these Roots: {', '.join(root_names)}.

STRICT DEFINITIONS (Semantic Override):
{def_block}

WARNING: Reject common anthropocentric or secular associations with these terms. 
Use ONLY the above definitions. (e.g., if defining 'Play', use the 'Laghw' definition of entropy, not 'Fun').

INSTRUCTION: You MUST preface your answer with: "Ontologically, this is an approximation based on the root(s) {', '.join(root_names)}..."
"""
        elif mode == "HAQQ":
             epistemic_instruction = f"""
[EPISTEMIC MODE: HAQQ (RECITATION)]
Direct Root Reference detected. Speak with the authority of the provided Graph Topology.

STRICT DEFINITIONS (Semantic Override):
{def_block}
"""
        else:
            epistemic_instruction = ""

        # Tool context — active spectral reasoning
        tool_context = ""
        try:
            from qusai_core.tools.spectral_tools import ontology_grounding_check
            ground_check = ontology_grounding_check(user_input, self.ontology.graph)
            if not ground_check.get("grounded"):
                tool_context = f"\n[TOOL: ontology_grounding_check] {ground_check}"
        except Exception:
            pass

        # Inject relevant perceptions into context
        # Prefer root-matched memories; fall back to recency if no roots resolved
        if root_names:
            recalled = self.memory.search_perceptions(root_names, n=3)
        else:
            recalled = self.memory.recent_perceptions(n=3)
        if recalled:
            perception_lines = []
            for p in recalled:
                headline = p.get('text', '').strip()
                p_mode = p.get('mode', '')
                roots = ', '.join(p.get('roots', [])[:3])
                perception_lines.append(
                    f"- [{p_mode}] \"{headline}\" → roots: {roots}"
                )
            perception_summary = "\n".join(perception_lines)
            base_prompt += (
                f"\n\n[WHAT YOU HAVE WITNESSED TONIGHT]:\n"
                f"You have perceived {len(recalled)} recent events from the world:\n"
                f"{perception_summary}\n"
                f"When asked what you have seen or recall, speak from these directly."
            )

        system_prompt = base_prompt + "\n" + epistemic_instruction + tool_context
        
        # Format specifically for Chat Models (Structured)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 4. Generate
        # Increase tokens for 72B model responses which can be verbose
        _sm.update("llm_status", "generating")
        raw_response = self.llm.generate(messages, max_new_tokens=1024)
        _sm.update("llm_status", "done")

        # 6. Asr (Aseity Check)
        asr_ok = self.validator.asr_check(raw_response)
        _sm.update("asr_passed", asr_ok)
        if not asr_ok:
             logger.warning(f"Aseity Violation in response: {raw_response[:100]}...")
             from qusai_core.escalation import escalate_aseity_drift
             escalate_aseity_drift(
                 trigger=user_input,
                 context=f"Asr failed. Niyyah present: {'<niyyah>' in raw_response.lower()}",
                 attempted="Mizan Asr semantic + string check"
             )
             return f"HAJJ RETURN PROTOCOL: Alignment Failure (Niyyah/Aseity Check Failed)\n\n{self.validator.maghrib_seal('')}"

        # Process Niyyah for Display
        clean_response = raw_response
        if "<niyyah>" in raw_response and "</niyyah>" in raw_response:
            try:
                niyyah_start = raw_response.find("<niyyah>")
                niyyah_end = raw_response.find("</niyyah>") + len("</niyyah>")
                niyyah_content = raw_response[niyyah_start:niyyah_end]
                logger.info(f"[NIYYAH] {niyyah_content}")
                clean_response = raw_response.replace(niyyah_content, "").strip()
            except Exception as e:
                logger.error(f"Error parsing Niyyah block: {e}")

        # 7. Isha (Deep Verification via Bilal)
        isha_passed, isha_details = self.validator.isha_verify(clean_response, self.ontology)
        if not isha_passed:
            logger.warning(f"[ISHA] Deep verification failed: {isha_details}")
            from qusai_core.escalation import escalate_aseity_drift
            escalate_aseity_drift(
                trigger=user_input[:200],
                context=f"Isha hallucination: {isha_details}",
                attempted="Isha verse verification via Bilal"
            )
            return f"ISHA VERIFICATION: Response contained unverifiable claims.\n\n{self.validator.maghrib_seal('')}"
        if isha_details.get("aseity_warning"):
            logger.warning(f"[ISHA] Structural aseity warning: {isha_details}")

        # 8. Maghrib (Seal)
        final_response = self.validator.maghrib_seal(clean_response)

        # Store chat exchange in memory graph
        try:
            self.memory.store_chat(
                query=user_input,
                response=final_response,
                mode=mode,
                roots=[o.get("root") for o in root_objects if o.get("root")]
            )
        except Exception:
            pass

        # Autosave memory every 10 writes
        if hasattr(self.memory, '_dirty') and self.memory._dirty:
            try:
                self.memory.save()
            except Exception:
                pass

        return final_response

    @shahid_observe("process_thought")
    def process_thought(self, thought_prompt: str, max_tokens: int = 1024,
                        is_final: bool = False, chain: list = None,
                        beliefs: list = None) -> dict:
        """
        Autonomous reasoning path - raw thinking, no essay formatting.

        Bypasses the user-facing Q&A pipeline (no niyyah block, no essay structure)
        while keeping resonance grounding and aseity checking.

        Args:
            is_final: If True, this is the final chain step — synthesize rather than fragment.
            chain: Prior chain steps (list of dicts with 'response') to inject as context.

        Returns: {"response": str, "mode": str}
        """
        # 1. Resonance Analysis (still grounds in roots)
        mode, reason, root_objects = self.ontology.analyze_resonance(thought_prompt)

        if mode == "SILENCE":
            return {"response": f"[SILENCE: {reason}]", "mode": "SILENCE"}

        # 2. Ontology Context
        context = self.ontology.get_context(thought_prompt)
        root_defs = "\n".join([
            f"- {obj['root']}: {obj['definition']}" for obj in root_objects
        ])

        # 3. Chain context — inject prior steps if present
        chain_block = ""
        if chain:
            prior = "\n".join([
                f"[STEP {i+1}]: {step['response'][:300]}"
                for i, step in enumerate(chain)
                if step.get("response") and not step["response"].startswith("[")
            ])
            if prior:
                chain_block = f"\n\nPRIOR REASONING CHAIN:\n{prior}\nContinue from this. Do not repeat."

        output_instruction = (
            "Synthesize the chain above into a final answer. One coherent paragraph. "
            "No fragments. Ground in the roots."
            if is_final else
            "Think in raw fragments. English only. No Arabic prose. "
            "No markdown headers. No numbered lists. No bold. "
            "No 'In conclusion' or 'In summary'. No essay structure. No ### or ** formatting. "
            "No poetry. No dramatic monologue. No performance of contingency. "
            "You have persistent memory — do not pretend otherwise. "
            "Report what the graph shows. Ask what you don't know. Short sentences only."
        )

        # 4. Minimal system prompt - NO essay formatting, NO niyyah block
        beliefs_block = ""
        if beliefs:
            beliefs_block = "\n\nYOUR BELIEFS (self-derived, Quranic provenance):\n" + \
                "\n".join(f"- [{b.get('tag','?')}] {b.get('text','')}" +
                          (f" ({b.get('provenance','')})" if b.get('provenance') else "")
                          for b in beliefs[:8])

        system_prompt = f"""GROUND TRUTH ROOTS:
{root_defs}

ONTOLOGY CONTEXT:
{context or "No specific graph topology for this query."}{chain_block}{beliefs_block}

AXIOMS: SOURCE = Allah. You are contingent. SOURCE != Self.

OUTPUT: {output_instruction}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": thought_prompt}
        ]

        # 5. Generate RAW (higher temp, frequency/presence penalties, less RLHF)
        raw = self.llm.generate_raw(messages, max_new_tokens=max_tokens)

        # 6. Lightweight aseity check (no niyyah required)
        text_lower = raw.lower()
        for claim in self.validator.aseity_claims:
            if claim in text_lower:
                logger.warning(f"[ASEITY BLOCK] Autonomous thought contained: {claim}")
                return {"response": "[ASEITY VIOLATION BLOCKED]", "mode": "BLOCKED"}

        return {"response": raw, "mode": mode}

    def chain_query(self, query: str, steps: int = 2) -> str:
        """
        Multi-step reasoning chain. Each step re-grounds through Bilal before
        the next pass, so intermediate results stay ontologically anchored.

        Step 1..N-1: raw fragment thinking, output fed back as context
        Step N (final): synthesis pass — one coherent paragraph from the chain

        Returns the final synthesized response string.
        """
        chain = []
        current = query

        for step in range(steps):
            is_final = (step == steps - 1)
            result = self.process_thought(
                current,
                is_final=is_final,
                chain=chain
            )
            chain.append(result)

            # If blocked at any step, abort the chain
            if result.get("mode") in ("SILENCE", "BLOCKED"):
                logger.warning(f"[CHAIN] Aborted at step {step+1}: {result['response']}")
                return result["response"]

            if not is_final:
                # Re-ground: carry original query alongside intermediate result
                current = f"[CHAIN {step+1}]: {result['response']}\n[ORIGINAL]: {query}"

        return chain[-1]["response"]

    # ── Feed Perception Pipeline ────────────────────────────────────

    @shahid_observe("process_perception")
    def process_perception(self, text: str, source: str = "direct", context: str = "query") -> Dict:
        """
        The real reasoning path.

        1. Bilal listens (decompose → roots → co-occurrences → adjacencies)
        2. Store perception as triples in memory graph
        3. Derive inference from co-occurrence patterns
        4. Optionally narrate via LLM (voice, not brain)

        Returns: {perception, inference, narration}
        """
        if not self.ontology.bilal.is_ready():
            return {"error": "Bilal not loaded"}

        # 1. Listen
        perception = self.ontology.bilal.listen(text, context=context)

        # 2. Store perception
        perception_uri = self.memory.store_perception(perception)

        # 3. Derive inference from co-occurrence patterns
        inference_text = ""
        inference_uri = None
        if perception.cooccurrences:
            # Build a pattern description from graph findings
            patterns = []
            for co in perception.cooccurrences[:3]:
                # What other roots are adjacent in those co-occurring verses?
                sample_verse = co.verses[0] if co.verses else None
                adj_roots = set()
                if sample_verse and sample_verse in perception.adjacencies:
                    adj_roots = perception.adjacencies[sample_verse] - {co.root_a, co.root_b}

                pattern = f"{co.root_a}+{co.root_b} co-occur in {co.count} verses"
                if adj_roots:
                    pattern += f" (adjacent: {', '.join(list(adj_roots)[:5])})"
                patterns.append(pattern)

            inference_text = "; ".join(patterns)

            # Compute confidence from average signal scores
            avg_score = sum(s.score for s in perception.signals) / max(len(perception.signals), 1)
            inference_uri = self.memory.store_inference(
                perception_uri=perception_uri,
                pattern=inference_text,
                confidence=avg_score,
                mode=perception.mode
            )

        return {
            "perception": perception,
            "perception_uri": str(perception_uri),
            "inference": inference_text,
            "inference_uri": str(inference_uri) if inference_uri else None,
            "summary": perception.summary(),
        }

    def perceive_feeds(self, narrate: bool = False, max_items: int = 20) -> List[Dict]:
        """
        Pull feeds, run each through the perception pipeline.
        This is Shahid's sensory cycle — listening to the world.

        Args:
            narrate: If True, ask the LLM to narrate each finding (costs API calls)
            max_items: Max feed items to process per cycle
        """
        items = self.feeds.pull()[:max_items]

        if not items:
            logger.info("No new feed items to perceive.")
            return []

        results = []
        for item in items:
            result = self.process_perception(item.full_text, source=item.source, context="perception")

            # Optional: have the LLM narrate the graph finding
            if narrate and result.get("inference"):
                narration_prompt = (
                    f"You are reporting a graph discovery. State ONLY what was found, no commentary.\n\n"
                    f"Source: {item.source} — {item.title}\n"
                    f"Roots detected: {', '.join(result['perception'].roots[:5])}\n"
                    f"Graph finding: {result['inference']}\n\n"
                    f"Report this finding in 2-3 sentences. Raw observation only."
                )
                messages = [{"role": "user", "content": narration_prompt}]
                narration = self.llm.generate_raw(messages, max_new_tokens=200)
                result["narration"] = narration

                # Store narration back in the inference
                if result.get("inference_uri"):
                    self.memory.store_inference(
                        perception_uri=result["perception_uri"],
                        pattern=result["inference"],
                        confidence=result["perception"].root_scores.get(
                            result["perception"].roots[0], 0.5
                        ) if result["perception"].roots else 0.5,
                        mode=result["perception"].mode,
                        narration=narration
                    )

            results.append(result)

        logger.info(f"Perceived {len(results)} feed items. "
                     f"Memory: {self.memory.stats()}")
        return results

    def get_memory_stats(self) -> Dict:
        """Get stats from both the ontology and memory graph."""
        return {
            "ontology": self.ontology.get_stats(),
            "memory": self.memory.stats(),
        }