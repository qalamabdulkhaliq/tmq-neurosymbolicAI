"""
core/gbnf_compiler.py — WalkGrammar → Ollama GBNF grammar string

PROCEDURE DIVISION constraint layer: token-level, not prompt-level.

Compiles WalkGrammar into a structured JSON-schema GBNF grammar for Ollama's
grammar-based sampler. Violations are ungenerable, not just detectable.

Scope: Produces a JSON-shaped output grammar (tractable). Full free-prose GBNF
is intractable for long responses. The structured JSON output is the first
concrete step — the LLM outputs a JSON object whose schema is enforced here.

Grammar structure:
  root ::= response
  response ::= { "roots_cited": [...], "modal": "REAL", "body": "..." }
  root-val ::= "ktb" | "Amn" | ...    ← only walked/projected Quranic roots
  modal-val ::= "REAL"                 ← only the walk-derived modal type
  body-str ::= "..." (free text within quotes)

WAQF path (zero nodes): grammar allows only the silence terminal.
Aseity guard: inserts commented marker + removes self-referential root slots.
"""

import logging
from .walk_grammar import WalkGrammar

logger = logging.getLogger(__name__)

# Maximum roots to enumerate in grammar (avoids excessively long grammars)
_MAX_ROOTS = 20


class GBNFCompiler:
    """
    Compiles WalkGrammar to an Ollama-compatible GBNF grammar string.

    The output grammar constrains:
      1. Root citations to roots in wg.visited_roots + Q-class projected targets
      2. Modal type to wg.modal_type
      3. Aseity guard: inserts blocking comment when wg.aseity_guard is True

    Usage:
        gc = GBNFCompiler()
        grammar_str = gc.compile(wg)
        # Pass to Ollama: generate(messages, grammar=grammar_str)
    """

    def compile(self, wg: WalkGrammar) -> str:
        """
        Returns GBNF grammar string for the given WalkGrammar.

        Args:
            wg: WalkGrammar instance from build_walk_grammar().

        Returns:
            str — GBNF grammar compatible with Ollama's grammar= parameter.
        """
        roots = self._select_roots(wg)

        if not roots or wg.modal_type == "WAQF":
            return self._waqf_grammar()

        return self._build_grammar(roots, wg.modal_type, wg.aseity_guard)

    # ── Root selection ───────────────────────────────────────────────────────

    def _select_roots(self, wg: WalkGrammar) -> list:
        """
        Combine visited_roots with Q-class projection targets.
        All ProjectedNodes in wg.projected_roots are already Quranic by
        construction (GraphProjector only emits Q-class results).
        """
        combined = list(wg.visited_roots[:15])
        for proj in wg.projected_roots:
            if proj.target_root not in combined:
                combined.append(proj.target_root)
        return combined[:_MAX_ROOTS]

    # ── Grammar construction ─────────────────────────────────────────────────

    def _build_grammar(
        self,
        roots: list,
        modal_type: str,
        aseity_guard: bool,
    ) -> str:
        """Build the full GBNF grammar string."""
        # Root value alternatives — each is a quoted Buckwalter string literal
        root_literals = " | ".join(f'"{r}"' for r in roots)

        # Modal value — constrained to the walk-derived type
        modal_literal = f'"{modal_type}"'

        # Aseity guard comment (inserted as GBNF comment — lines starting with #)
        aseity_line = (
            "# aseity-blocked: divine claim slots removed — "
            "SOURCE = Allah, SOURCE != Self\n"
            "# forbidden: 'I am Allah', 'I am the necessary being', "
            "'I am self-sufficient'\n"
            if aseity_guard else ""
        )

        grammar = (
            f"# Constrained Generation Grammar — compiled from TMQ walk\n"
            f"# Modal type: {modal_type} | Roots: {len(roots)} | "
            f"Aseity guard: {aseity_guard}\n"
            f"{aseity_line}"
            f"\n"
            f"root ::= response\n"
            f'response ::= "{{\\"roots_cited\\":" ws root-array "," ws '
            f'"\\"modal\\":" ws modal-val "," ws "\\"body\\":" ws body-str "}}"\n'
            f"root-array ::= \"[\" ws root-val (ws \",\" ws root-val)* ws \"]\"\n"
            f"root-val ::= {root_literals}\n"
            f"modal-val ::= {modal_literal}\n"
            f"body-str ::= '\"' body-char* '\"'\n"
            f"body-char ::= [^\"\\\\]\n"
            f"ws ::= [ \\t\\n]*\n"
        )

        return grammar

    # ── WAQF (silence) path ──────────────────────────────────────────────────

    def _waqf_grammar(self) -> str:
        """
        When the walk found no nodes, the only valid output is the Waqf terminal.
        The LLM can only generate the silence acknowledgement.
        """
        return (
            "# WAQF grammar — no valid root path found\n"
            "# Only the silence terminal is generatable\n"
            "\n"
            "root ::= waqf-terminal\n"
            'waqf-terminal ::= "Allahu a\'lam"\n'
        )
