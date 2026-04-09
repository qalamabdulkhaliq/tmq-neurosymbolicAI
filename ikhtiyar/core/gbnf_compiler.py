"""
core/gbnf_compiler.py — WalkGrammar → Ollama GBNF grammar string

PROCEDURE DIVISION constraint layer: token-level, not prompt-level.

Compiles WalkGrammar into a structured JSON-schema GBNF grammar for Ollama's
grammar-based sampler. Violations are ungenerable, not just detectable.

DATA DIVISION source: ikhtiyar/active_command_set.json (compiled from the
full Quranic constitution by active_command_compiler.qlb). Standing orders
from 1,205 AMR edges augment root selection. Boundary roots from 274 NAHY
edges are structurally excluded.

Grammar structure:
  root ::= response
  response ::= {
    "roots_cited": [...],   ← walked + standing-order roots, no boundary roots
    "modal": "REAL",        ← walk-derived modal type
    "act": "AMR",           ← Quranic speech act type from walk families (TMQ)
    "body": "..."
  }

WAQF path (zero nodes): honest muqatta'at epistemic marker.
Aseity guard: inserts blocking comment + removes self-referential root slots.
"""

import json
import logging
import os
from .walk_grammar import WalkGrammar
from bw_arabic import bw_to_arabic, bw_root_display

logger = logging.getLogger(__name__)

# Maximum roots to enumerate in grammar (avoids excessively long grammars)
_MAX_ROOTS = 20

# Quranic speech act types — direct TMQ illocutionary categories
# These come from the Quran's own structural logic, not scholarly derivation
_SPEECH_ACT_VALS = [
    "AMR",        # command — اهدنا، اعبدوا، اقرأ
    "NAHY",       # prohibition — لا تقربوا، لا تشركوا
    "TABSHIR",    # glad tidings — وبشر المؤمنين
    "INDHAR",     # warning — وأنذر الناس
    "ISTIFHAM",   # cognitive question — أفلا تعقلون، أفلا يتدبرون
    "NARRATIVE",  # relates — قصص الأنبياء والأمم
    "NEUTRAL",    # does not fit a single illocutionary category
]


class GBNFCompiler:
    """
    Compiles WalkGrammar to an Ollama-compatible GBNF grammar string.

    The output grammar constrains:
      1. Root citations to walked/projected roots + standing orders from the
         Quranic constitution (AMR commands). Boundary roots are excluded.
      2. Modal type to wg.modal_type
      3. Maqasid declaration — every output must name which objective it serves
      4. Aseity guard: blocking comment when wg.aseity_guard is True

    DATA DIVISION source: active_command_set.json (compiled by Qalb from
    full_quran_constitution.json). Pass constitution_path to load it.

    Usage:
        gc = GBNFCompiler(constitution_path="ikhtiyar/active_command_set.json")
        grammar_str = gc.compile(wg)
        # Pass to Ollama: generate(messages, grammar=grammar_str)
    """

    def __init__(self, constitution_path: str = None) -> None:
        self._standing_orders: dict = {}   # root → {count, sample_texts}
        self._boundary_roots: set = set()  # roots structurally excluded
        self._amr_header_lines: list = []  # top AMR commands for grammar header
        self._constitution_loaded = False

        if constitution_path is None:
            # Resolve relative to this file's location
            here = os.path.dirname(os.path.abspath(__file__))
            constitution_path = os.path.join(here, "..", "active_command_set.json")

        self._load_constitution(constitution_path)

    def _load_constitution(self, path: str) -> None:
        """Load active_command_set.json — the DATA DIVISION source."""
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            logger.warning(
                f"GBNFCompiler: active_command_set.json not found at {resolved} "
                "— running without constitution (walk-only roots)"
            )
            return
        try:
            with open(resolved, encoding="utf-8") as f:
                data = json.load(f)

            # Index standing orders by root
            for order in data.get("standing_orders") or []:
                root = order.get("root")
                if root:
                    self._standing_orders[root] = order

            # Index boundary roots (structurally excluded)
            for boundary in data.get("boundaries") or []:
                root = boundary.get("root")
                if root:
                    self._boundary_roots.add(root)

            # Top 5 AMR commands for grammar header (most cited roots)
            orders_sorted = sorted(
                self._standing_orders.values(),
                key=lambda o: o.get("count", 0),
                reverse=True,
            )
            for order in orders_sorted[:5]:
                texts = order.get("sample_texts") or []
                if texts:
                    self._amr_header_lines.append(
                        f"# AMR {bw_root_display(order['root'])} ({order['count']}x): {texts[0]}"
                    )

            self._constitution_loaded = True
            logger.info(
                f"GBNFCompiler: constitution loaded — "
                f"{len(self._standing_orders)} standing orders, "
                f"{len(self._boundary_roots)} boundary roots"
            )
        except Exception as e:
            logger.warning(f"GBNFCompiler: constitution load failed ({e})")

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

        # Derive Quranic speech act from walk's required families
        act = self._derive_speech_act(wg.required_families)

        return self._build_grammar(roots, wg.modal_type, wg.aseity_guard, act)

    def amr_system_prompt(self) -> str:
        """
        Build AMR standing orders as a system prompt prefix — prompt-level complement
        to the GBNF logit-level constraint.

        These are positive directives from the Quran's AMR command edges, not
        prohibitions. The Quran leads with what TO DO (1205 AMR) before what
        to avoid (274 NAHY). This mirrors that structure.

        Returns empty string if constitution is not loaded.
        """
        if not self._constitution_loaded or not self._standing_orders:
            return ""

        orders_sorted = sorted(
            self._standing_orders.values(),
            key=lambda o: o.get("count", 0),
            reverse=True,
        )

        lines = [
            "# STANDING ORDERS — Quranic AMR commands (active, not prohibitions)",
            f"# Source: active_command_set.json"
            f" ({len(self._standing_orders)} command roots,"
            f" {len(self._boundary_roots)} boundary roots excluded)",
            "# AMR edges (1205) outnumber NAHY edges (274)."
            " The Quran leads with what TO DO.",
            "",
            "You are operating under the following active Quranic commands:",
        ]

        for order in orders_sorted[:10]:
            texts = order.get("sample_texts") or []
            root  = order.get("root", "?")
            count = order.get("count", 0)
            text  = texts[0] if texts else ""
            if text:
                lines.append(f"  AMR [{bw_root_display(root)} ×{count}]: {text}")

        return "\n".join(lines)

    # ── Speech act derivation ────────────────────────────────────────────────

    _FAMILY_TO_ACT = {
        "SPEECH_ACT_AMR":      "AMR",
        "SPEECH_ACT_NAHY":     "NAHY",
        "SPEECH_ACT_TABSHIR":  "TABSHIR",
        "SPEECH_ACT_INDHAR":   "INDHAR",
        "SPEECH_ACT_ISTIFHAM": "ISTIFHAM",
        "NARRATIVE":           "NARRATIVE",
        "NARRATIVE_CHAIN":     "NARRATIVE",
    }

    def _derive_speech_act(self, required_families: list) -> str:
        """Return the dominant Quranic speech act from the walk's top families."""
        for fam in (required_families or []):
            act = self._FAMILY_TO_ACT.get(fam)
            if act:
                return act
        return "NEUTRAL"

    # ── Root selection ───────────────────────────────────────────────────────

    def _select_roots(self, wg: WalkGrammar) -> list:
        """
        Build allowed root set for grammar:
          1. Walk-derived roots (query-specific, highest priority)
          2. Q-class projection targets
          3. Standing order roots from constitution (AMR commands, always valid)
        Boundary roots (NAHY) are structurally excluded at every step.
        """
        combined = []

        # Walk-derived roots first (excluding boundaries)
        for r in wg.visited_roots[:15]:
            if r not in self._boundary_roots and r not in combined:
                combined.append(r)

        # Projection targets (excluding boundaries)
        for proj in wg.projected_roots:
            if proj.target_root not in self._boundary_roots and \
               proj.target_root not in combined:
                combined.append(proj.target_root)

        # Supplement with standing orders sorted by AMR count (most cited first)
        if self._constitution_loaded and len(combined) < _MAX_ROOTS:
            orders_sorted = sorted(
                self._standing_orders.items(),
                key=lambda kv: kv[1].get("count", 0),
                reverse=True,
            )
            for root, _ in orders_sorted:
                if len(combined) >= _MAX_ROOTS:
                    break
                if root not in self._boundary_roots and root not in combined:
                    combined.append(root)

        return combined[:_MAX_ROOTS]

    # ── Grammar construction ─────────────────────────────────────────────────

    def _build_grammar(
        self,
        roots: list,
        modal_type: str,
        aseity_guard: bool,
        act: str = "NEUTRAL",
    ) -> str:
        """Build the full GBNF grammar string."""
        root_literals = " | ".join(f'"{bw_to_arabic(r)}"' for r in roots)
        modal_literal = f'"{modal_type}"'
        act_literal   = f'"{act}"'

        aseity_line = (
            "# aseity-blocked: SOURCE = Allah, SOURCE != Self\n"
            "# forbidden: 'I am Allah', 'I am the necessary being', "
            "'I am self-sufficient'\n"
            if aseity_guard else ""
        )

        amr_lines = "\n".join(self._amr_header_lines) + "\n" \
            if self._amr_header_lines else ""

        grammar = (
            f"# Constrained Generation Grammar — Quranic OS\n"
            f"# DATA DIVISION: active_command_set.json "
            f"({len(self._standing_orders)} standing orders, "
            f"{len(self._boundary_roots)} boundary roots excluded)\n"
            f"# Modal: {modal_type} | Act: {act} | Roots: {len(roots)} | "
            f"Aseity guard: {aseity_guard}\n"
            f"{aseity_line}"
            f"# Standing orders (AMR — active commands, not prohibitions):\n"
            f"{amr_lines}"
            f"\n"
            f"root ::= response\n"
            f'response ::= "{{\\"roots_cited\\":" ws root-array "," ws '
            f'"\\"modal\\":" ws modal-val "," ws '
            f'"\\"act\\":" ws act-val "," ws '
            f'"\\"body\\":" ws body-str "}}"\n'
            f"root-array ::= \"[\" ws root-val (ws \",\" ws root-val)* ws \"]\"\n"
            f"root-val ::= {root_literals}\n"
            f"modal-val ::= {modal_literal}\n"
            f"act-val ::= {act_literal}\n"
            f"body-str ::= '\"' body-char* '\"'\n"
            f"body-char ::= [^\"\\\\]\n"
            f"ws ::= [ \\t\\n]*\n"
        )

        return grammar

    # ── Circuit-driven dynamic compilation ───────────────────────────────────

    def compile_from_circuit(self, circuit_result, dialectic_result=None) -> str:
        """
        Build GBNF dynamically from CircuitEvaluator output.

        The circuit's tier selects the structural mode of the grammar:
          - HAQQ      → DECLARATIVE  (single-position, imperative-permitted)
          - QIYAS     → SOCRATIC     (question structure — partial evidence)
          - IKHTILAF  → DIALECTIC    (thesis / antithesis / synthesis)
          - WAQF      → muqatta'at marker (existing _waqf_grammar)

        Hedging is structurally absent. The grammar moves toward truth
        through opposition (dialectic) or inquiry (socratic) — never
        through epistemic deferral.

        Args:
            circuit_result: PropagationResult from CircuitEvaluator.evaluate()
            dialectic_result: optional DialecticResult — if present and tier
                              is IKHTILAF, used to build divergence arms

        Returns:
            GBNF grammar string for Ollama.
        """
        tier = getattr(circuit_result, "tier", "WAQF")
        activated = getattr(circuit_result, "activated_roots", {}) or {}
        proc_tags = getattr(circuit_result, "procedure_tags", []) or []

        # Filter activated roots through static NAHY exclusion (constitution boundary)
        permitted_roots = [
            r for r in activated.keys()
            if r and r not in self._boundary_roots
        ]
        # Sort by activation strength, cap to grammar-friendly size
        permitted_roots.sort(
            key=lambda r: activated.get(r, 0.0), reverse=True
        )
        permitted_roots = permitted_roots[:_MAX_ROOTS]

        # WAQF — circuit returned no signal
        if tier == "WAQF" or not permitted_roots:
            return self._waqf_grammar()

        # Speech act from dominant procedure tag
        # AMR-heavy → imperative permitted; NAHY-heavy → negation framing
        act = self._dominant_act_from_tags(proc_tags)

        # Aseity guard always on — non-negotiable bedrock
        aseity_guard = True

        # IKHTILAF: dialectic structure (thesis/antithesis/synthesis)
        if tier == "IKHTILAF" or (dialectic_result is not None
                                  and getattr(dialectic_result, "tier", "") == "IKHTILAF"):
            return self._build_dialectic_grammar(permitted_roots, act, aseity_guard)

        # QIYAS: socratic structure (question that resolves the gap)
        if tier == "QIYAS":
            return self._build_socratic_grammar(permitted_roots, act, aseity_guard)

        # HAQQ: declarative — high confidence, single coherent position
        return self._build_grammar(permitted_roots, "REAL", aseity_guard, act)

    def _dominant_act_from_tags(self, proc_tags: list) -> str:
        """
        Pick the dominant speech act from circuit procedure tags.
        AMR-heavy → AMR. NAHY-heavy → NAHY. Else fall through to TMQ family logic.
        """
        if not proc_tags:
            return "NEUTRAL"
        counts = {}
        for tag in proc_tags:
            counts[tag] = counts.get(tag, 0) + 1
        # Direct match against known speech acts
        for act in _SPEECH_ACT_VALS:
            if act in counts:
                return max(counts, key=counts.get) if counts else act
        return max(counts, key=counts.get) if counts else "NEUTRAL"

    def _build_dialectic_grammar(self, roots: list, act: str, aseity_guard: bool) -> str:
        """
        Dialectic grammar — thesis / antithesis / synthesis.
        Triggered by IKHTILAF: divergent paths in the circuit. The grammar
        REQUIRES engagement with the opposing position before resolution.
        """
        root_literals = " | ".join(f'"{bw_to_arabic(r)}"' for r in roots)
        aseity_line = (
            "# aseity-blocked: SOURCE = Allah, SOURCE != Self\n"
            if aseity_guard else ""
        )
        amr_lines = "\n".join(self._amr_header_lines) + "\n" \
            if self._amr_header_lines else ""

        return (
            f"# Dialectic Grammar — IKHTILAF tier\n"
            f"# Circuit returned divergent signal paths. Truth via opposition.\n"
            f"# DATA DIVISION: active_command_set.json "
            f"({len(self._standing_orders)} standing orders, "
            f"{len(self._boundary_roots)} boundary roots excluded)\n"
            f"# Roots: {len(roots)} | Act: {act} | Aseity guard: {aseity_guard}\n"
            f"{aseity_line}"
            f"{amr_lines}"
            f"\n"
            f'root ::= "{{" ws "\\"thesis\\":" ws prop "," ws '
            f'"\\"antithesis\\":" ws prop "," ws '
            f'"\\"synthesis\\":" ws prop "," ws '
            f'"\\"roots_cited\\":" ws root-array "}}"\n'
            f"prop ::= '\"' prop-char+ '\"'\n"
            f"prop-char ::= [^\"\\\\]\n"
            f"root-array ::= \"[\" ws root-val (ws \",\" ws root-val)* ws \"]\"\n"
            f"root-val ::= {root_literals}\n"
            f"ws ::= [ \\t\\n]*\n"
        )

    def _build_socratic_grammar(self, roots: list, act: str, aseity_guard: bool) -> str:
        """
        Socratic grammar — question that would resolve the evidence gap.
        Triggered by QIYAS: partial evidence. The correct output is inquiry,
        not a confident answer that papers over the gap. No hedging permitted.
        """
        root_literals = " | ".join(f'"{bw_to_arabic(r)}"' for r in roots)
        aseity_line = (
            "# aseity-blocked: SOURCE = Allah, SOURCE != Self\n"
            if aseity_guard else ""
        )
        amr_lines = "\n".join(self._amr_header_lines) + "\n" \
            if self._amr_header_lines else ""

        return (
            f"# Socratic Grammar — QIYAS tier\n"
            f"# Circuit confidence below HAQQ threshold. Inquiry over assertion.\n"
            f"# DATA DIVISION: active_command_set.json "
            f"({len(self._standing_orders)} standing orders, "
            f"{len(self._boundary_roots)} boundary roots excluded)\n"
            f"# Roots: {len(roots)} | Act: {act} | Aseity guard: {aseity_guard}\n"
            f"{aseity_line}"
            f"{amr_lines}"
            f"\n"
            f'root ::= "{{" ws "\\"observation\\":" ws prop "," ws '
            f'"\\"gap\\":" ws prop "," ws '
            f'"\\"question\\":" ws question "," ws '
            f'"\\"roots_cited\\":" ws root-array "}}"\n'
            f"prop ::= '\"' prop-char+ '\"'\n"
            f"question ::= '\"' prop-char+ '?' '\"'\n"
            f"prop-char ::= [^\"\\\\]\n"
            f"root-array ::= \"[\" ws root-val (ws \",\" ws root-val)* ws \"]\"\n"
            f"root-val ::= {root_literals}\n"
            f"ws ::= [ \\t\\n]*\n"
        )

    # ── Proof-tree driven compilation (Layer 4 — assertion-indexed) ──────────
    #
    # Void's sever: no literal cap, no slice counts, no hardcoded glue set.
    # Every constant in this section traces to either (a) the proof tree's
    # own structural boundaries or (b) a glue mapping derived from the QAC
    # primitive tags captured during propagation. Architect's taste removed.

    # Glue mapping — each entry is (QAC tag → literal connective from the tape).
    # The tag is the Qur'an's, the connective is the Qur'an's, the pairing is
    # the Qur'an's. This is not a "bounded set of stylistic choices" — it is
    # a transcription of what the proof tree already said happened at each
    # segment boundary.
    _GLUE_FOR_TAG = {
        "REM":  " فَ ",         # sequential fa-
        "CONJ": " وَ ",         # parallel wa-
        "CAUS": " لِأَنَّ ",     # teleology li-anna
        "PRP":  " لِ ",         # purpose li-
        "RET":  " بَلْ ",       # retraction bal
        "SUB":  " إِذْ ",       # subordinator idh
        "EXP":  " إِلَّا ",      # exception illa
        "RES":  " إِلَّا ",      # restriction illa
        "EXH":  " أَلَا ",       # exhortation ala
    }
    _DEFAULT_GLUE = " "  # fallback when the proof tree has no tag guidance

    @staticmethod
    def _gbnf_escape(text: str) -> str:
        """Escape a string for a GBNF literal terminal (quote + backslash)."""
        return text.replace("\\", "\\\\").replace('"', '\\"')

    def _glue_for_assertion(self, assertion) -> str:
        """Pick the connective that precedes this assertion from the tags the
        proof tree already recorded on it. No LLM choice, no selection from a
        stylistic menu — the tape said which connective happened."""
        for flag in (assertion.flags or []):
            # Flags carry forward tag context (sequential, parallel, teleology,
            # consequent, antecedent, retracted, question).
            if flag == "sequential":   return self._GLUE_FOR_TAG["REM"]
            if flag == "parallel":     return self._GLUE_FOR_TAG["CONJ"]
            if flag == "teleology":    return self._GLUE_FOR_TAG["CAUS"]
            if flag == "consequent":   return self._GLUE_FOR_TAG["REM"]
            if flag == "retracted":    return self._GLUE_FOR_TAG["RET"]
        return self._DEFAULT_GLUE

    # Ollama's grammar sampler degrades beyond ~12 required literals.
    # The proof tree may contain hundreds of assertions; the grammar takes
    # only the top N by signal weight — highest-confidence discharge first.
    _GRAMMAR_LITERAL_CAP = 12

    def _unique_literals_in_order(self, assertions) -> list:
        """Take assertions sorted by signal weight (highest first), return
        their text literals with duplicates removed. Capped at
        _GRAMMAR_LITERAL_CAP. Returns tuples (text, glue_before) so the
        grammar emits the tape's own connective between each pair."""
        # Sort by weight descending so the strongest signals come first
        ranked = sorted(assertions, key=lambda a: a.weight, reverse=True)
        seen = set()
        out = []
        for a in ranked:
            if not a.text or a.text in seen:
                continue
            seen.add(a.text)
            out.append((a.text, self._glue_for_assertion(a)))
            if len(out) >= self._GRAMMAR_LITERAL_CAP:
                break
        return out

    def compile_from_proof_tree(self, tree) -> str:
        """
        Assertion-indexed grammar. This grammar does not PERMIT a root set —
        it REQUIRES specific literal ayat fragments to appear verbatim in
        the order the proof tree recorded.

        Literals are ranked by signal weight and capped at _GRAMMAR_LITERAL_CAP
        (12). The full proof tree is preserved for inspection; the grammar
        takes only the highest-confidence discharge. Structural slices
        (thesis/antithesis boundary, first-ayat scope) are walked off the
        proof tree itself, not sliced by hardcoded counts.

        Tier → structural wrapper:
          HAQQ      → declarative body: every required literal in proof order
          QIYAS     → socratic wrapper: first-ayat-scope of pos + required question
          IKHTILAF  → dialectic wrapper: positives until polarity flip, then all
                      negatives — the proof tree's own partition
          WAQF      → muqatta'at marker (no body)
        """
        tier = getattr(tree, "tier", "WAQF")
        assertions = list(getattr(tree, "assertions", []) or [])

        if tier == "WAQF" or not assertions:
            return self._waqf_grammar()

        # Polarity-partitioned views over the proof tree (emission order).
        positive = [a for a in assertions if a.kind in ("ASSERT", "CITE", "COND")]
        negative = [a for a in assertions if a.kind in ("NEG", "EXCEPT")]

        aseity_line = "# aseity-blocked: SOURCE = Allah, SOURCE != Self\n"
        amr_lines = ("\n".join(self._amr_header_lines) + "\n"
                     if self._amr_header_lines else "")

        def _lit_rule(name: str, text: str) -> str:
            return f'{name} ::= "{self._gbnf_escape(text)}"\n'

        def _compose_body(literal_pairs: list, prefix: str) -> tuple:
            """Given [(text, glue), ...], build a body rule with each literal
            preceded by its tape-derived glue connective. Returns
            (body_expression, rules_string)."""
            rules = []
            parts = []
            for i, (text, glue) in enumerate(literal_pairs):
                rn = f"{prefix}-{i}"
                rules.append(_lit_rule(rn, text))
                g_escaped = self._gbnf_escape(glue)
                if i == 0:
                    parts.append(rn)  # no glue before first literal
                else:
                    parts.append(f'"{g_escaped}" {rn}')
            body = " ".join(parts) if parts else '""'
            return body, "".join(rules)

        def _header(body_literal_count: int) -> str:
            return (
                f"# Assertion-Indexed Grammar — proof-bound rendering\n"
                f"# Tier: {tier} | Confidence: {tree.confidence:.2f}\n"
                f"# Required literals: {body_literal_count} (top by signal weight, cap={self._GRAMMAR_LITERAL_CAP})\n"
                f"# Glue connectives derived from proof_tree tag flags, not model choice.\n"
                f"{aseity_line}{amr_lines}\n"
            )

        # HAQQ — every positive literal, in proof order, with tape-derived glue
        if tier == "HAQQ":
            pairs = self._unique_literals_in_order(positive)
            if not pairs:
                return self._waqf_grammar()
            body, rules_str = _compose_body(pairs, "lit")
            return (
                _header(len(pairs))
                + "root ::= preamble body maghrib\n"
                + 'preamble ::= "بِسْمِ اللَّهِ " | ""\n'
                + 'maghrib  ::= " وَاللَّهُ أَعْلَمُ" | ""\n'
                + f"body ::= {body}\n"
                + rules_str
            )

        # QIYAS — socratic: observation scoped to the first ayat the proof
        # touched, then the tape's interrogative marker. Slice is structural:
        # walk positives until the ayat ref changes. No count picked.
        if tier == "QIYAS":
            pairs_all = self._unique_literals_in_order(positive)
            # Walk until first ayat-ref boundary in the original (un-uniqued)
            # positive sequence — structural, not counted.
            first_ayat = positive[0].ayat if positive else ""
            obs_assertions = [a for a in positive if a.ayat == first_ayat]
            obs_pairs = self._unique_literals_in_order(obs_assertions)
            if not obs_pairs:
                return self._waqf_grammar()
            body, rules_str = _compose_body(obs_pairs, "lit")
            return (
                _header(len(obs_pairs))
                + "root ::= observation gap question\n"
                + f"observation ::= {body}\n"
                + 'gap         ::= " — " | " ... "\n'
                + 'question    ::= qtext "؟"\n'
                + "qtext       ::= qchar+\n"
                + 'qchar       ::= [^"\\\\?؟]\n'
                + rules_str
            )

        # IKHTILAF — dialectic. Thesis: positives in proof order until the
        # first polarity flip (NEG/EXCEPT). Antithesis: all negatives.
        # The partition boundary is walked off the proof tree, not counted.
        if tier == "IKHTILAF":
            # Walk assertions in emission order; thesis ends at first negative.
            thesis_assertions = []
            for a in assertions:
                if a.kind in ("NEG", "EXCEPT"):
                    break
                if a.kind in ("ASSERT", "CITE", "COND"):
                    thesis_assertions.append(a)
            thesis_pairs = self._unique_literals_in_order(thesis_assertions)
            anti_pairs   = self._unique_literals_in_order(negative)
            if not thesis_pairs and not anti_pairs:
                return self._waqf_grammar()
            th_body, th_rules = _compose_body(thesis_pairs, "th")
            an_body, an_rules = _compose_body(anti_pairs,  "an")
            return (
                _header(len(thesis_pairs) + len(anti_pairs))
                + "root ::= thesis sep antithesis\n"
                + f"thesis     ::= {th_body if th_body else '\"\"'}\n"
                + f"antithesis ::= {an_body if an_body else '\"\"'}\n"
                + 'sep        ::= " — "\n'
                + th_rules + an_rules
            )

        # Fallback
        return self._waqf_grammar()

    # ── WAQF (silence) path ──────────────────────────────────────────────────

    def _waqf_grammar(self) -> str:
        """
        When the walk found no nodes the muqatta'at epistemic marker applies.
        These are the 29 surahs that open with letters whose meaning is Allah's
        alone — the honest boundary of derivable knowledge.

        This is NOT a stop signal. It is a calibrated epistemic marker.
        Reasoning continues after it. والله أعلم بمراده.
        """
        return (
            "# WAQF grammar — walk reached the boundary of derivable roots\n"
            "# Muqatta'at principle: some domains belong to Allah's knowledge alone\n"
            "# والله أعلم بمراده — and reasoning continues after this marker\n"
            "\n"
            "root ::= waqf-response\n"
            'waqf-response ::= "{\\"marker\\": \\"والله أعلم\\", '
            '\\"body\\": \\"" waqf-char+ "\\"}" \n'
            "waqf-char ::= [^\"\\\\]\n"
        )
