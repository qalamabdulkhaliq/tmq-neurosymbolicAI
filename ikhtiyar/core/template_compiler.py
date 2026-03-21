"""
core/template_compiler.py — WalkGrammar → Slot Template (PROCEDURE DIVISION)

Converts the DATA DIVISION (WalkGrammar) into a structured slot template
that replaces the current narrative constrained_prompt.

Grace Hopper's COBOL structure:
  DATA DIVISION — all declared facts (program-computed, immutable)
  PROCEDURE DIVISION — operations that can only reference declared identifiers

The slot template makes the boundary explicit:
  [COMPILED GRAMMAR — DATA DIVISION]
  ... verified roots, projections, families, mode ...

  [PROCEDURE DIVISION — fill slots using only DATA above]
  <RESPONSE_TYPE>{{REAL|CMPLX|WAQF}}</RESPONSE_TYPE>
  <CITATION_1>{{cite a FACT from above}}</CITATION_1>
  <BODY>{{...}}</BODY>
  <SEAL>Allahu a'lam</SEAL>

Slot types vary by modal_type:
  REAL  → <ASSERTION>  — declarative claim
  CMPLX → <CONSIDERATION> — hypothetical or qualified statement
  ZERO  → <BOUNDARY>   — limit or threshold statement
  WAQF  → <SEAL> only  — silence
"""

import logging
from .walk_grammar import WalkGrammar

logger = logging.getLogger(__name__)


class TemplateCompiler:
    """
    Compiles WalkGrammar into a two-part prompt template:
      DATA DIVISION — labeled fact list (verified, program-computed)
      PROCEDURE DIVISION — slot skeleton with {{...}} fill markers

    The LLM fills slots; it cannot reference anything outside the DATA DIVISION
    without violating the stated constraint (detectable by Mizan).
    """

    def compile(self, wg: WalkGrammar) -> str:
        """
        Returns a complete slot template string.

        Args:
            wg: WalkGrammar instance (populated by build_walk_grammar).

        Returns:
            str — the formatted template, replacing constrained_prompt.
        """
        if wg.modal_type == "WAQF" or not wg.visited_roots:
            return self._waqf_template(wg)

        data_div = self._data_division(wg)
        proc_div = self._procedure_division(wg)
        return data_div + "\n\n" + proc_div

    # ── DATA DIVISION ────────────────────────────────────────────────────────

    def _data_division(self, wg: WalkGrammar) -> str:
        lines = ["[COMPILED GRAMMAR — DATA DIVISION]"]
        lines.append(f"QUESTION: {wg.question}")

        # Verified roots
        if wg.seed_roots:
            lines.append(f"VERIFIED_ROOTS: {', '.join(wg.seed_roots)}")

        # Projections
        if wg.projected_roots:
            proj_parts = []
            for n in wg.projected_roots[:4]:
                proj_parts.append(
                    f"{n.source_root}→{n.target_root} "
                    f"[{n.operation}, {n.transition}, D={n.d_class_tgt}]"
                )
            lines.append("PROJECTED: " + " | ".join(proj_parts))

        # Edge families
        if wg.required_families:
            lines.append(f"FAMILIES: {', '.join(wg.required_families)}")

        # Modal type
        modal_desc = {
            "REAL":  "declarative register — assert what is structurally confirmed",
            "CMPLX": "hypothetical register — qualified or conditional claims only",
            "ZERO":  "boundary register — limit or threshold statements",
        }
        lines.append(
            f"MODE: {wg.modal_type} ({modal_desc.get(wg.modal_type, 'unknown register')})"
        )

        # Intensity
        if wg.intensity > 0.0:
            lines.append(f"INTENSITY: {wg.intensity:.2f}")

        # Aseity guard
        if wg.aseity_guard:
            lines.append(
                "ASEITY: GUARDED — this domain touches divine attributes. "
                "Do not assert divine attributes for yourself. "
                "SOURCE = Allah. SOURCE ≠ Self."
            )

        # Eigenstate facts
        if wg.eigenstate_facts:
            lines.append("FACTS:")
            for fact in wg.eigenstate_facts:
                lines.append(f"  - {fact}")

        # All valid root tokens (for PROCEDURE DIVISION reference)
        all_roots = list(wg.visited_roots[:15])
        if all_roots:
            lines.append(f"VALID_ROOTS: {' | '.join(all_roots)}")

        return "\n".join(lines)

    # ── PROCEDURE DIVISION ───────────────────────────────────────────────────

    def _procedure_division(self, wg: WalkGrammar) -> str:
        lines = [
            "[PROCEDURE DIVISION — fill slots using only the DATA DIVISION above]",
            "(Only cite roots listed in VALID_ROOTS. "
            "Only make claims supported by FACTS. "
            "Violations are detectable.)",
        ]

        # Response type slot
        valid_types = self._valid_response_types(wg.modal_type)
        lines.append(f"<RESPONSE_TYPE>{{{{{valid_types}}}}}</RESPONSE_TYPE>")

        # Citation slot — must reference a FACT
        if wg.eigenstate_facts:
            lines.append(
                "<CITATION_1>{{cite one FACT from the DATA DIVISION above}}</CITATION_1>"
            )

        # Modal body slot — type depends on modal_type
        body_slot = self._body_slot(wg)
        lines.append(body_slot)

        # Optional: projection commentary
        if wg.projected_roots:
            lines.append(
                "<PROJECTION_NOTE>{{optional: describe what the projection from "
                f"{wg.projected_roots[0].source_root} to "
                f"{wg.projected_roots[0].target_root} reveals}}"
                "</PROJECTION_NOTE>"
            )

        # Seal — always present
        lines.append("<SEAL>Allahu a'lam</SEAL>")

        return "\n".join(lines)

    def _valid_response_types(self, modal_type: str) -> str:
        mapping = {
            "REAL":  "ASSERTION",
            "CMPLX": "CONSIDERATION",
            "ZERO":  "BOUNDARY",
            "WAQF":  "SILENCE",
        }
        return mapping.get(modal_type, "CONSIDERATION")

    def _body_slot(self, wg: WalkGrammar) -> str:
        if wg.modal_type == "REAL":
            return (
                "<ASSERTION>{{declarative claim citing at least one root from VALID_ROOTS "
                "and at least one FACT}}</ASSERTION>"
            )
        elif wg.modal_type == "CMPLX":
            return (
                "<CONSIDERATION>{{hypothetical or qualified statement — use 'perhaps', "
                "'it may be', or conditional framing; cite roots from VALID_ROOTS}}"
                "</CONSIDERATION>"
            )
        elif wg.modal_type == "ZERO":
            return (
                "<BOUNDARY>{{limit or threshold statement — describe what cannot be "
                "crossed; cite roots from VALID_ROOTS}}</BOUNDARY>"
            )
        else:
            return "<BODY>{{response grounded in VALID_ROOTS and FACTS}}</BODY>"

    # ── WAQF (silence) path ──────────────────────────────────────────────────

    def _waqf_template(self, wg: WalkGrammar) -> str:
        roots_str = ", ".join(wg.seed_roots) if wg.seed_roots else "(none)"
        return (
            f"[WAQF — Silence]\n"
            f"No nodes found for roots: {roots_str}\n\n"
            f"<SEAL>Allahu a'lam — no valid structural path found for this question.</SEAL>"
        )
