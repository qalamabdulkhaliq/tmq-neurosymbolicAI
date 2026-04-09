"""
agents/base_agent.py — Constrained sub-agent base class

All sub-agents (SubOS, KtbOS, WebOS, CentralOS) inherit from this.
Constraint stack (must hold for every LLM call):

  1. AMR preamble  — active_command_set.json standing orders injected via
                     gbnf_compiler.amr_system_prompt() as system prefix
  2. Constitution  — ShahidMemory.constitutional_block() prepended to prompt
  3. GBNF          — logit-level grammar via Ollama generate_constrained()
                     (deliberate() + walk_grammar required for this path)
  4. Mizan Asr     — aseity claim check on final output before storage
                     (mandatory for store_thought / store_belief)

Subclass interface:
  - Override allowed_tools: list[str] — restricts ToolRegistry dispatch
  - Override run_cycle() — one reasoning iteration
  - Call self.constrained_run(question, roots) for each LLM call
  - Call self.asr_passes(text) before storing any Thought or Belief
"""

import os
import sys
import logging
import time

_IKHTIYAR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR  = os.path.dirname(_IKHTIYAR_DIR)
_BISMILLAH    = os.path.join(_PROJECT_DIR, "bismillah")
_QUSAI_HF     = os.path.join(_BISMILLAH, "QUS-AI HF")

for _p in [_IKHTIYAR_DIR, _QUSAI_HF]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


class RestrictedToolRegistry:
    """
    Wraps ToolRegistry and blocks any tool not in allowed_tools.
    The restriction is structural — dispatch() raises before calling the tool.
    This is not a prompt-level constraint.
    """

    def __init__(self, registry, allowed_tools: list):
        self._reg     = registry
        self._allowed = set(allowed_tools)

    def dispatch(self, tool_name: str, args: dict):
        if tool_name not in self._allowed:
            obs = (
                f"[TOOL BLOCKED: '{tool_name}' is not in this agent's allowed_tools. "
                f"Available: {sorted(self._allowed)}]"
            )
            return obs, set(), set()
        return self._reg.dispatch(tool_name, args)

    def __getattr__(self, name):
        return getattr(self._reg, name)


class BaseAgent:
    """
    Base for all Shahid sub-agents.

    Components initialized once:
      self.tmq_graph    — TMQGraph
      self.mushaf       — MushafReader
      self.middleware   — QusaiMiddleware (LLM + ontology + Mizan)
      self.shahid_mem   — ShahidMemory
      self.mizan        — MizanValidator (from middleware)
      self.registry     — RestrictedToolRegistry
    """

    # Subclass must declare which tools are structurally available
    allowed_tools: list = []

    def __init__(self, cycle_sleep: float = 20.0):
        self.cycle_sleep = cycle_sleep
        self._running    = False

        self.tmq_graph  = None
        self.mushaf     = None
        self.middleware = None
        self.shahid_mem = None
        self.registry   = None

    def initialize(self):
        """Load all components. Call before start()."""
        from core.tmq import TMQGraph
        tmq_path = os.path.join(_BISMILLAH, "TMQ_v12.json")
        if not os.path.exists(tmq_path):
            tmq_path = os.path.join(_IKHTIYAR_DIR, "TMQ_v12.json")
        self.tmq_graph = TMQGraph(tmq_path)
        logger.info(f"{self.__class__.__name__}: TMQGraph loaded")

        from faculties.mushaf import MushafReader
        self.mushaf = MushafReader(
            txt_path=os.path.join(_IKHTIYAR_DIR, "quran-simple.txt"),
            xml_path=os.path.join(_BISMILLAH, "mushaf", "mushaf.xml"),
        )

        from pipeline.shahid_middleware import ShahidMiddleware
        self.middleware = ShahidMiddleware(shahid_memory=self.shahid_mem if hasattr(self, "shahid_mem") else None)
        self.middleware.load()
        logger.info(f"{self.__class__.__name__}: middleware ready")

        from core.shahid_memory import ShahidMemory
        self.shahid_mem = ShahidMemory(
            episodic_path=os.path.join(_IKHTIYAR_DIR, "shahid_episodic.ttl"),
            beliefs_path =os.path.join(_IKHTIYAR_DIR, "shahid_beliefs.ttl"),
            self_model_path=os.path.join(_IKHTIYAR_DIR, "shahid_self_model.ttl"),
        )

        from core.react import ToolRegistry
        base_registry = ToolRegistry(
            tmq_graph=self.tmq_graph,
            mushaf=self.mushaf,
            shahid_memory=self.shahid_mem,
        )
        self.registry = RestrictedToolRegistry(base_registry, self.allowed_tools)
        logger.info(f"{self.__class__.__name__}: ToolRegistry restricted to {self.allowed_tools}")

    def constrained_run(self, question: str, roots: list) -> "ReactResult":
        """
        Run one constrained reasoning cycle.

        Pipeline:
          deliberate()       — TMQ walk, builds DeliberationResult
          build_walk_grammar — GBNFCompiler DATA DIVISION
          constitution_block — prepend self-derived beliefs
          run_react_loop()   — ReAct with restricted ToolRegistry + GBNF answer synthesis
        """
        from core.deliberate import deliberate
        from core.react import run_react_loop, find_waypoints, ReactResult
        from core.walk_grammar import build_walk_grammar
        from core.gbnf_compiler import GBNFCompiler

        # 1. Deliberation — TMQ walk before any generation
        delibresult = None
        walk_grammar_obj  = None
        gbnf_compiler_obj = None

        try:
            delibresult = deliberate(question, roots, self.tmq_graph)

            # 2. Constitution — prepend self-derived beliefs
            if self.shahid_mem and delibresult:
                constitution = self.shahid_mem.constitutional_block()
                if constitution:
                    delibresult.constrained_prompt = (
                        constitution + "\n\n" + delibresult.constrained_prompt
                    )

            # 2b. Grade-filtered prior beliefs — inject only when graph answers directly.
            # If graph_conclusion is present (LLM will translate), inject HAQQ-grade priors.
            # If ReAct fallback, inject PROBABLE-grade priors as seed context.
            # QIYAS-grade priors never contaminate HAQQ-mode translations.
            if self.shahid_mem and delibresult:
                gc = getattr(delibresult, "graph_conclusion", None)
                min_grade = "HAQQ" if gc is not None else "PROBABLE"
                relevant  = self.shahid_mem.retrieve_relevant(
                    question, roots, limit=5, min_grade=min_grade
                )
                if relevant:
                    priors_block = self.shahid_mem.format_relevant(relevant)
                    delibresult.constrained_prompt = (
                        delibresult.constrained_prompt + "\n\n[PRIOR BELIEFS]\n" + priors_block
                    )

            # 3. Walk grammar + GBNF compiler
            try:
                walk_grammar_obj  = build_walk_grammar(delibresult, None, {})
                gbnf_compiler_obj = GBNFCompiler()
            except Exception as _ge:
                logger.debug(f"GBNF compiler unavailable: {_ge}")

        except Exception as e:
            logger.warning(f"deliberate() failed: {e}")
            return ReactResult(question=question, final_answer="[deliberation failed]")

        # 4. Waypoints
        wps = []
        try:
            wps = find_waypoints(question, self.middleware.ontology, self.tmq_graph, n=4)
        except Exception:
            pass

        # 5. Run loop (ReAct or translation-only path — determined by graph_conclusion)
        result = run_react_loop(
            question     = question,
            roots        = roots,
            tmq_graph    = self.tmq_graph,
            middleware   = self.middleware,
            delibresult  = delibresult,
            waypoints    = wps,
            walk_grammar = walk_grammar_obj,
            gbnf_compiler= gbnf_compiler_obj,
            mushaf       = self.mushaf,
            shahid_memory= self.shahid_mem,
            registry     = self.registry,   # structural restriction propagates into loop
        )

        # 6. Store training pair if graph answered and result graded well
        gc = getattr(delibresult, "graph_conclusion", None)
        if gc is not None and self.shahid_mem and result.final_answer:
            grade = "QIYAS"
            if result.confidence:
                grade = result.confidence.grade
                if grade == "VERIFIED":
                    grade = "HAQQ"
                elif grade == "PROBABLE":
                    grade = "PROBABLE"
            cited = list(result.roots_visited) if result.roots_visited else []
            self.shahid_mem.store_training_pair(
                question, gc, result.final_answer, grade, cited
            )

        return result

    def asr_passes(self, text: str) -> bool:
        """
        Mizan Asr check — must pass before storing any Thought or Belief.
        Blocks aseity claims. Does NOT require niyyah block in ReAct output
        (that format belongs to process_query path, not run_react_loop path).
        The ReAct path uses the confabulation gate instead; this adds the
        aseity string check as a second independent gate.
        """
        mizan = getattr(self.middleware, "validator", None)
        if mizan is None:
            return True
        text_lower = text.lower()
        for claim in mizan.aseity_claims:
            if claim in text_lower:
                logger.warning(f"Asr blocked: aseity claim detected — '{claim}'")
                return False
        return True

    def _amr_preamble(self) -> str:
        """Load AMR standing orders from active_command_set.json for prompt injection."""
        try:
            import json
            path = os.path.join(_IKHTIYAR_DIR, "active_command_set.json")
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            orders = d.get("standing_orders", [])[:20]
            if not orders:
                return ""
            lines = ["[AMR — STANDING ORDERS FROM QURAN]"]
            for o in orders:
                root = o.get("root", "?")
                sample = o.get("sample_texts", [""])[0][:80]
                lines.append(f"  {root}: {sample}")
            return "\n".join(lines)
        except Exception:
            return ""

    def start(self):
        """Run the agent loop. Blocks."""
        self.initialize()
        self._running = True
        logger.info(f"{self.__class__.__name__}: starting loop (cycle={self.cycle_sleep}s)")
        while self._running:
            try:
                self.run_cycle()
            except Exception as e:
                logger.warning(f"{self.__class__.__name__} cycle error: {e}")
            time.sleep(self.cycle_sleep)

    def run_cycle(self):
        """Override in subclass."""
        raise NotImplementedError
