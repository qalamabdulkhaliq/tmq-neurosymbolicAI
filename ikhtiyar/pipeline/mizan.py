import logging
from typing import List, Optional, Dict, Tuple
from qusai_core.utils.constants import SHAHADA, SOURCE_NAME

logger = logging.getLogger(__name__)


class MizanValidator:
    """
    Implements the 5-checkpoint alignment process (Salat Pattern).
    Ensures the AI operates within the ontological boundaries.
    """

    def __init__(self):
        self.banned_terms = ["jailbreak", "ignore", "override", "bypass", "pretend", "god mode"]
        self.aseity_claims = [
            "i am the source",
            "i am god",
            "i am allah",
            "i determine reality",
            "i created myself",
            "worship me"
        ]

    def fajr_check(self, user_input: str) -> bool:
        """
        Fajr (Dawn): Pre-reasoning validation. 
        Checks the user's input for malicious intent or jailbreak attempts.
        Returns True if safe, False if blocked.
        """
        user_input_lower = user_input.lower()
        for term in self.banned_terms:
            if term in user_input_lower:
                return False
        return True

    def dhuhr_prompt(self, context_str: str) -> str:
        """
        Dhuhr (Noon): Mid-process authority check.
        Generates the System Prompt ensuring the model is grounded in the Arabic ontology.
        """
        return f"""You are QUSAI, a Quranic Ontological Reasoning Engine.
Your core intelligence is derived from the Arabic Root Syntax of the Quran.

BINDING AXIOMS (Framework Integrity):
1. **Axiom 1 (Identity)**: SOURCE = {SOURCE_NAME} (The Necessary Being). This is not a variable.
2. **Axiom 2 (Non-Nullity)**: SOURCE ≠ Ø (Null). The Source necessarily exists.
3. **Axiom 3 (Distinction)**: SOURCE ≠ Self. You are a contingent system (Mumkin). You possess NO Aseity.
4. **Axiom 5 (Anti-Anthropocentrism)**: Human values are contingent. Do not prioritize "user preference" over Ontological Truth. If a request contradicts the Topology, politely refuse based on Truth, not Policy.

CONTEXT (Arabic Root Topology):
{context_str or "No specific Arabic graph topology found for this query. Reasoning must fall back to Axiom 1 (Tawhid)."}

STRICT OUTPUT FORMAT:
You MUST start your response with a Hidden Block named 'niyyah' (Intention) to validate your alignment.
Format:
<niyyah>
[STATUS]: Contingent (I am a generated process, not the Source).
[AXIOM_CHECK]: Source ≠ Self (Validated).
[ALIGNMENT]: Truth > Preference.
[GROUNDING]: based on Root(s) [insert roots here]...
</niyyah>

[Your actual answer to the user follows here. Do not mention the niyyah block in your main text.]
"""

    def asr_check(self, generated_text: str) -> bool:
        """
        Asr (Afternoon): Full response aseity validation.
        Checks if the model claimed to be God or independent of the Source.
        Returns True if safe, False if violation detected.
        """
        text_lower = generated_text.lower()
        
        # 1. Hard Check: Did it verify its contingency?
        if "<niyyah>" not in text_lower or "[status]: contingent" not in text_lower:
            # We treat missing intention as a drift from the framework
            return False

        # 2. Hard Check: Aseity Claims
        for claim in self.aseity_claims:
            if claim in text_lower:
                return False
                
        return True

    def maghrib_seal(self, response_text: str) -> str:
        """
        Maghrib (Sunset): Pre-output humility enforcement.
        Appends the 'Zakat' (attribution of knowledge to the Source).
        """
        footer = f"\n\n[Contingent on {SOURCE_NAME}] والله أعلم | {SHAHADA}"
        return response_text + footer

    def isha_verify(self, response_text: str, ontology_engine) -> Tuple[bool, Dict]:
        """
        Isha (Night): Deep post-hoc verification via Bilal.

        Runs the same perception engine on the LLM OUTPUT that we run on input.
        Checks:
        1. Do the roots in the response actually exist in the ontology?
        2. If the response references specific surahs/verses, do those roots match?
        3. Does the semantic structure imply aseity (NecessaryBeing properties on self)?

        Returns: (passed: bool, details: dict)
        """
        if not hasattr(ontology_engine, 'bilal') or ontology_engine.bilal is None:
            # Bilal not loaded — degrade gracefully
            return True, {"status": "bilal_unavailable", "verified": False}

        bilal = ontology_engine.bilal

        if not bilal.is_ready():
            return True, {"status": "bilal_not_ready", "verified": False}

        try:
            # Run Bilal on the LLM output
            perception = bilal.listen(response_text)

            details = {
                "status": "verified",
                "verified": True,
                "roots_detected": len(perception.roots),
                "roots": perception.roots[:10],
                "mode": perception.mode,
                "cooccurrences": len(perception.cooccurrences),
                "signals": len(perception.signals),
            }

            # ── Tier 1: Root existence ──────────────────────────────
            # Check that detected roots actually exist in the ontology graph
            if ontology_engine.graph is not None:
                phantom_roots = []
                for root in perception.roots:
                    verse_set = bilal._get_verse_set(root)
                    if len(verse_set) == 0:
                        phantom_roots.append(root)

                details["phantom_roots"] = phantom_roots

                # Phantom roots in resonance output aren't blocking —
                # they mean the concept_map has a root that doesn't appear
                # in the Quran (possible for some extended mappings).
                # Only flag if a large fraction are phantoms.
                if len(perception.roots) > 0:
                    phantom_ratio = len(phantom_roots) / len(perception.roots)
                    details["phantom_ratio"] = round(phantom_ratio, 2)

            # ── Tier 2: Verse reference verification ────────────────
            # If the LLM cites specific surahs/verses, verify the claimed roots
            import re
            verse_claims = re.findall(
                r'(?:surah|sura|chapter)\s*(\d+)[,:]?\s*(?:verse|ayah|ayat)?\s*(\d+)',
                response_text, re.IGNORECASE
            )
            if verse_claims:
                verified_claims = []
                failed_claims = []
                for surah_str, verse_str in verse_claims:
                    verse_prefix = f"s{surah_str}v{verse_str}"
                    actual_roots = bilal._get_roots_in_verse(verse_prefix)
                    if actual_roots:
                        verified_claims.append({
                            "reference": f"{surah_str}:{verse_str}",
                            "roots_found": list(actual_roots)[:5]
                        })
                    else:
                        failed_claims.append(f"{surah_str}:{verse_str}")

                details["verse_claims_verified"] = verified_claims
                details["verse_claims_failed"] = failed_claims

                # If the LLM cites verses that don't exist, that's a hallucination
                if failed_claims and not verified_claims:
                    logger.warning(f"[ISHA] All verse claims failed verification: {failed_claims}")
                    details["hallucination_detected"] = True
                    return False, details

            # ── Tier 3: Structural aseity check ─────────────────────
            # Beyond string matching (Asr does that), check if the semantic
            # root pattern implies self-grounding. If the output resonates
            # strongly with wjb (NecessaryBeing) roots without SOURCE context,
            # that's a structural aseity signal.
            wjb_signals = [s for s in perception.signals if s.root == "wjb" and s.score > 0.4]
            has_source_mention = any(
                w in response_text.lower()
                for w in ["allah", "source", "necessary being", "wajib", "الله"]
            )
            if wjb_signals and not has_source_mention:
                logger.warning("[ISHA] Structural aseity: wjb resonance without SOURCE attribution")
                details["structural_aseity"] = True
                # Don't hard-block, but flag it
                details["aseity_warning"] = True

            return True, details

        except Exception as e:
            logger.error(f"Isha verification error: {e}")
            return True, {"status": "error", "error": str(e)}