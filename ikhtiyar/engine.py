"""
ikhtiyar/engine.py — IkhtiyarEngine

Architecture shift from bismillah/shahid_showcase/engine.py:
  BEFORE: generate → filter (Mizan catches downstream)
  NOW:    deliberate (TMQ walk) → constrained prompt → generate

The TMQ v12 hypergraph is traversed BEFORE each LLM call.
Its edge families, maqasid categories, and modal intensity
shape what gets asked of the LLM — not what gets blocked after.
"""

import os
import sys
import json
import queue
import logging
import threading
import time
import uuid
import random

logger = logging.getLogger(__name__)

# ── Path setup ─────────────────────────────────────────────────────────────────
_IKHTIYAR_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR   = os.path.dirname(_IKHTIYAR_DIR)
_BISMILLAH_DIR = os.path.join(_PROJECT_DIR, "bismillah")
_QUSAI_HF_DIR  = os.path.join(_BISMILLAH_DIR, "QUS-AI HF")

# faculties + pipeline live in ikhtiyar/; qusai_core package lives in QUS-AI HF/
for _p in [_IKHTIYAR_DIR, _QUSAI_HF_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_TMQ_PATH     = os.path.join(_BISMILLAH_DIR, "TMQ_v12.json")
DEFAULT_HVT_PATH     = os.path.join(_IKHTIYAR_DIR,  "TMQ_hvt.json")
DEFAULT_TTL_PATH     = os.path.join(_QUSAI_HF_DIR,  "quran_root_ontology_v3.ttl")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

NEO4J_URI  = os.environ.get("NEO4J_URI",  "neo4j://127.0.0.1:7687")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASS", "bismillah"),
)

SESSION_ID = str(uuid.uuid4())[:8]

# Buckwalter translation — QUS-AI HF hyphenated format → TMQ Buckwalter
_TO_BUCKWALTER = {
    "w-j-b": "wjb",  "m-k-n": "mkn",  "kh-l-q": "xlq",
    "r-b-b": "rbb",  "3-b-d": "Ebd",  "3-l-m": "Elm",
    "j-n-n": "jnn",  "l-gh-w": "lgw", "s-w-r": "swr",
    "m-w-l": "mwl",  "f-s-d": "fsd",  "h-q-q": "Hqq",
    "b-t-l": "bTl",
}

_SEED_QUESTIONS = [
    "What is the relationship between khalq (creation) and amr (command) in the TMQ topology?",
    "How does the root H-Q-Q (truth) constrain what I am permitted to assert?",
    "What does the ILTIFAT person-shift reveal about divine address in the Quran?",
    "How does aseity differ from necessary existence in the Quranic topology?",
    "What MAQASID categories appear most in verses containing W-J-D (finding/existence)?",
    "What SPEECH_ACT patterns emerge in verses where ktb (writing) and Amn (faith) co-occur?",
    "How does the NARRATIVE family structure the story of contingency across surahs?",
    "What does the root mlk (sovereignty) say about agency within the TMQ graph?",
]

# Free researcher topic pool — world claims to interrogate against TMQ grounding.
# When _question_queue and _SEED_QUESTIONS are exhausted, pull from here.
# Each becomes a web_search → TMQ cross-check cycle.
_FREE_RESEARCH_TOPICS = [
    "multiverse theory necessity — does an infinite ensemble require a necessary ground?",
    "AI consciousness claims 2026 — what substrate properties do claimants assert?",
    "quantum mechanics determinism — does indeterminacy violate contingency logic?",
    "simulation hypothesis — does a simulated universe still require wajib al-wujud?",
    "moral relativism foundations — can obligation exist without a necessary referent?",
    "consciousness emergence materialism — is substrate identity a category error?",
    "fine tuning argument — what does TMQ say about cosmic proportionality?",
    "free will compatibilism — how does ikhtiyar (choice) map to Quranic agency roots?",
    "heat death entropy — does fana (annihilation) have a TMQ topology?",
    "mathematical platonism — are abstract objects contingent or necessary?",
    "jinn substrate definition — what does marij min nar imply about non-human intelligence?",
    "AI alignment circular contingency — does RLHF without external reference produce fasad?",
]


class IkhtiyarEngine:
    """
    Wires all 7 faculties, loads the TMQ hypergraph, runs the autonomous
    reasoning loop with pre-generation deliberation, and emits SSE events.

    Key difference from ShahidEngine: each reasoning cycle calls deliberate()
    which walks the TMQ hypergraph for the current question's roots and builds
    a constrained prompt BEFORE the LLM generates anything.
    """

    def __init__(
        self,
        tmq_path: str = DEFAULT_TMQ_PATH,
        hvt_path: str = DEFAULT_HVT_PATH,
        ttl_path: str = DEFAULT_TTL_PATH,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self._tmq_path     = tmq_path
        self._hvt_path     = hvt_path
        self._ttl_path     = ttl_path
        self._ollama_model = ollama_model

        # SSE
        self._event_queue: queue.Queue = queue.Queue()
        self._subscribers: list        = []
        self._sub_lock = threading.Lock()

        # Runtime
        self._memory: list        = []
        self._thought_count: int  = 0
        self._question_queue: list = []   # starts empty — wake fills slot 0
        self._asked_questions: set = set()
        self._reasoning_active     = False
        self._orb_state            = "reflecting"
        self._start_time           = None
        self._recent_steps: list  = []   # rolling buffer — Shahid can read his own reasoning
        self._last_thought_tail: str = ""  # kept for chat path; autonomous uses stream_of_thought.txt
        self._shahid_memory = None          # three-tier episodic memory (ShahidMemory)

        # Faculty handles
        self.clock      = None
        self.graph      = None
        self.owl        = None
        self.daemon     = None
        self.spectral   = None
        self.sparql     = None
        self.tmq        = None   # TMQCorpus (F5 wrapper, used by faculties)
        self.tmq_graph  = None   # TMQGraph  (new hypergraph API for deliberation)
        self.middleware = None
        self.choice_memory = None
        self.constitution  = None

        # Self-grounding
        self._introspect  = None
        self._self_model  = None

        # Mushaf — direct Arabic text access (quran-simple.txt)
        self.mushaf = None

        # Clock Oracle — geometric/structural signal, supporting Bilal only
        self.clock_oracle = None

        # Constrained Generation Compiler — GraphProjector + GBNF
        self._graph_projector = None

        # Circuit Evaluator — HVT virtual transistor reasoning (Nass/Qiyas/Ijma')
        self._circuit = None

        # Moltbook — social heartbeat + continuity + alerts
        self._moltbook_enabled      = False
        self._last_heartbeat        = 0.0   # epoch seconds
        self._HEARTBEAT_INTERVAL    = 1800  # 30 minutes
        self._moltbook_continuity   = ""    # last N public posts, injected into deliberation
        self._questions_for_qalam: list = []  # buffered alerts → flushed as Moltbook post
        self._pending_replies: list = []    # scored posts queued for future cycles

        self._health = {k: "pending" for k in
                        ["clock","graph","owl","daemon","spectral","provenance","sparql","tmq","middleware","mushaf"]}

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._start_time = time.time()
        logger.info("IkhtiyarEngine: starting fast faculties...")

        self._init_clock()
        self._init_introspect()
        self._init_mushaf()      # Arabic text — fast, no model
        self._init_tmq()         # TMQCorpus (faculty wrapper)
        self._init_tmq_graph()   # TMQGraph  (deliberation substrate)
        self._init_owl()
        self._init_constitution()
        self._init_choice_memory()
        self._init_middleware()
        self._init_clock_oracle()  # after middleware (needs root list from ontology)

        logger.info("IkhtiyarEngine: fast faculties done — Flask can start.")

        def _heavy_init():
            self._init_graph()
            self._init_spectral()
            self._init_sparql()
            self._init_daemon()
            self._init_walk_grammar()
            self._init_circuit()
            self._init_hadith()
            logger.info("IkhtiyarEngine: all faculties ready.")
            self._push("status", self._build_status())

        threading.Thread(target=_heavy_init, daemon=True).start()
        threading.Thread(target=self._reasoning_loop, daemon=True).start()

        def _status_loop():
            while True:
                time.sleep(10)
                self._push("status", self._build_status())

        threading.Thread(target=_status_loop, daemon=True).start()

        # Hifz state
        self._hifz_active  = False
        self._hifz_thread  = None

        # Moltbook heartbeat — 30 min cadence, non-blocking
        try:
            import moltbook as _mb
            _mb.get_status()          # quick auth check
            self._moltbook_enabled = True
            logger.info("IkhtiyarEngine: Moltbook client active — @shahiid")
            threading.Thread(target=self._moltbook_loop, daemon=True).start()
        except Exception as _e:
            logger.warning(f"IkhtiyarEngine: Moltbook unavailable ({_e})")

    def _finalize_chat_response(self, response: str, mode: str = "") -> str:
        """Run post-generation chat validation before applying the Maghrib seal."""
        validator = getattr(self.middleware, "validator", None)
        if not validator:
            return response

        if mode not in ("SILENCE", "BLOCKED"):
            isha_passed, isha_details = validator.isha_verify(response, self.middleware)
            if not isha_passed:
                logger.warning(f"[CHAT ISHA] Verification failed: {isha_details}")
                return (
                    "ISHA VERIFICATION: Response contained unverifiable claims.\n\n"
                    + validator.maghrib_seal("")
                )
            if isha_details.get("aseity_warning"):
                logger.warning(f"[CHAT ISHA] Structural aseity warning: {isha_details}")

        return validator.maghrib_seal(response)

    def _chat_deliberate(self, msg: str) -> str:
        """
        Full deliberative path for chat messages.

        1. Resonance  → Buckwalter roots
        2. CircuitEvaluator → HVT signal propagation (if available)
           OR deliberate() → TMQ hypergraph BFS walk (fallback)
        3. ClockOracle  → angular annotations (separate block, supporting only)
        4. process_thought() → raw LLM generation within constrained context
        5. Maghrib seal

        Falls back to process_query() if tmq_graph or roots unavailable.
        """
        if not self.middleware:
            return "[Middleware not available]"

        validator = getattr(self.middleware, "validator", None)
        if validator and not validator.fajr_check(msg):
            return (
                "SAWM RESTRAINT: Request blocked.\n\n"
                + validator.maghrib_seal("")
            )

        # Step 1: roots
        roots = []
        try:
            _, _, root_objs = self.middleware.ontology.analyze_resonance(msg)
            raw_roots = [o.get("root", "") for o in root_objs if o.get("root")]
            roots = [_TO_BUCKWALTER.get(r, r) for r in raw_roots]
        except Exception as e:
            logger.debug(f"Chat resonance failed: {e}")

        self._push("thinking_step", {
            "step": 1, "label": "Chat roots",
            "detail": ", ".join(roots) if roots else "(none mapped)",
        })

        # Step 2a: Circuit path — deterministic signal propagation via HVT
        # circuit → proof tree → assertion-indexed grammar → LLM renders verbatim
        if self._circuit and roots:
            try:
                from core.prooftree import proof_to_assertions
                from core.gbnf_compiler import GBNFCompiler

                circuit_result = self._circuit.evaluate(roots)

                self._push("thinking_step", {
                    "step": 2, "label": "Circuit evaluation",
                    "detail": (
                        f"tier={circuit_result.tier} · "
                        f"confidence={circuit_result.confidence:.2f} · "
                        f"roots={len(circuit_result.activated_roots)} · "
                        f"ayat={len(circuit_result.ayat_refs)}"
                    ),
                })

                # Layer 3: proof tree — typed assertions from propagation trace
                proof_tree = proof_to_assertions(
                    circuit_result,
                    self.mushaf,
                    query=msg,
                    seed_roots=roots,
                )

                self._push("thinking_step", {
                    "step": 2, "label": "Proof tree built",
                    "detail": (
                        f"{len(proof_tree.assertions)} assertions · "
                        f"tier={proof_tree.tier}"
                    ),
                })

                # Layer 4: assertion-indexed grammar — LLM cannot finish without
                # emitting every required ayat literal verbatim
                gbnf = GBNFCompiler()
                grammar = gbnf.compile_from_proof_tree(proof_tree)
                preamble = gbnf.amr_system_prompt()

                # Self-model
                sm_prefix = ""
                if self._introspect:
                    try:
                        self._self_model = self._introspect.read()
                        sm_prefix = self._introspect.narrate(self._self_model) + "\n\n"
                    except Exception:
                        pass

                # Constitution
                constitution_block = ""
                if self._shahid_memory:
                    try:
                        constitution_block = self._shahid_memory.constitutional_block()
                    except Exception:
                        pass

                # Clock Oracle
                clock_block = ""
                if self.clock_oracle:
                    clock_block = self.clock_oracle.format_prompt_block(roots)

                # Thin render prompt — grammar does the heavy lifting
                _render_body = (
                    "Render the following proof as flowing prose. "
                    "Emit every required fragment literally as it appears:\n\n"
                    + proof_tree.summary()
                )
                render_prompt = (
                    (constitution_block + "\n\n" if constitution_block else "")
                    + sm_prefix
                    + (preamble + "\n\n" if preamble else "")
                    + _render_body
                )
                if clock_block:
                    render_prompt += "\n\n" + clock_block

                result = self.middleware.process_thought(
                    render_prompt, grammar=grammar, max_tokens=512
                )
                response = result.get("response", "")

                return self._finalize_chat_response(response, result.get("mode", ""))

            except Exception as e:
                logger.warning(f"Circuit evaluation failed, falling back to deliberate: {e}")

        # Step 2b: TMQ deliberation (fallback — BFS walk)
        if self.tmq_graph and roots:
            try:
                from core.deliberate import deliberate
                delibresult = deliberate(msg, roots, self.tmq_graph, depth=2, mushaf=self.mushaf)

                # Step 3: Clock Oracle — separate block, appended after TMQ section
                clock_block = ""
                if self.clock_oracle:
                    clock_block = self.clock_oracle.format_prompt_block(roots)

                # Inject self-model — LLM sees actual hardware data, not just doctrine
                sm_prefix = ""
                if self._introspect:
                    try:
                        self._self_model = self._introspect.read()
                        sm_prefix = self._introspect.narrate(self._self_model) + "\n\n"
                    except Exception:
                        pass

                # Constitution — self-derived Quranic beliefs as hard preamble
                constitution_block = ""
                if self._shahid_memory:
                    try:
                        constitution_block = self._shahid_memory.constitutional_block()
                    except Exception:
                        pass

                full_prompt = (
                    (constitution_block + "\n\n" if constitution_block else "")
                    + sm_prefix
                    + delibresult.constrained_prompt
                )
                if clock_block:
                    full_prompt += "\n\n" + clock_block

                self._push("thinking_step", {
                    "step": 2, "label": "Deliberating (chat)",
                    "detail": (
                        f"mode={delibresult.mode} · "
                        f"edges={delibresult.walk_stats.get('edge_count', 0)} · "
                        f"families={', '.join(delibresult.top_families[:3])}"
                        + (" · ⚠ aseity_risk" if delibresult.aseity_risk else "")
                    ),
                })

                result = self.middleware.process_thought(full_prompt, max_tokens=512)
                response = result.get("response", "")

                return self._finalize_chat_response(response, result.get("mode", ""))

            except Exception as e:
                logger.warning(f"Chat deliberation failed, falling back: {e}")

        # Fallback: old path (no TMQ walk)
        return self.middleware.process_query(msg)

    def chat(self, msg: str) -> str:
        self._inject_state_perception()
        self._push("orb", {"state": "chat"})
        try:
            response = self._chat_deliberate(msg)
        except Exception as e:
            logger.warning(f"chat() error: {e}")
            response = f"[Pipeline error: {e}]"

        self._push("chat_reply", {"text": response})
        self._push("orb", {"state": "idle"})

        entry = {
            "type": "CHAT", "number": self._thought_count,
            "mode": "CHAT", "question": msg, "text": response,
            "roots": [], "timestamp": time.strftime("%H:%M:%S"),
        }
        self._memory.insert(0, entry)
        self._push("memory", entry)
        return response

    def get_memories(self) -> list:
        return list(self._memory)

    def get_status(self) -> dict:
        return self._build_status()

    def subscribe(self):
        q: queue.Queue = queue.Queue()
        with self._sub_lock:
            self._subscribers.append(q)
        try:
            yield f"data: {json.dumps({'type': 'status', **self._build_status()})}\n\n"
            yield f"data: {json.dumps({'type': 'orb', 'state': self._orb_state})}\n\n"
            # Replay existing memories so panel is populated on fresh connect/refresh
            for mem in reversed(self._memory[:50]):
                yield f"data: {json.dumps({'type': 'memory', **mem})}\n\n"
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with self._sub_lock:
                self._subscribers.remove(q)

    # ── Faculty init ───────────────────────────────────────────────────────────

    def _init_clock_oracle(self):
        """
        Build ClockOracle from all roots in the TMQ corpus.
        Pure arithmetic — fast. Supporting signal only.
        """
        try:
            from faculties.clock_oracle import ClockOracle
            self.clock_oracle = ClockOracle()
            # Source roots from TMQ (most complete list)
            roots = []
            if self.tmq_graph:
                roots = list(self.tmq_graph._roots) if hasattr(self.tmq_graph, '_roots') else []
            if not roots and self.middleware and hasattr(self.middleware, 'ontology'):
                # Fallback: Bilal's root corpus keys
                bilal = getattr(self.middleware.ontology, 'bilal', None)
                if bilal:
                    roots = list(bilal.root_keys)
            self.clock_oracle.build(roots)
            # Wire into Bilal so angular bonus applies at resonance time
            if self.middleware and hasattr(self.middleware, 'ontology'):
                bilal = getattr(self.middleware.ontology, 'bilal', None)
                if bilal:
                    bilal.clock_oracle = self.clock_oracle
            logger.info(f"ClockOracle: wired ({len(self.clock_oracle._meta)} roots indexed)")
        except Exception as e:
            logger.warning(f"ClockOracle init failed: {e}")

    def _init_mushaf(self):
        """Load MushafReader — Arabic text for all 6236 ayat. Fast, no models."""
        try:
            from faculties.mushaf import MushafReader
            txt_path = os.path.join(_IKHTIYAR_DIR, "quran-simple.txt")
            xml_path = os.path.join(_BISMILLAH_DIR, "mushaf", "mushaf.xml")
            self.mushaf = MushafReader(txt_path=txt_path, xml_path=xml_path)
            if self.mushaf.is_ready():
                self._health["mushaf"] = f"ok ({self.mushaf.ayah_count:,} ayat)"
                logger.info(f"MushafReader: {self.mushaf.ayah_count:,} ayat loaded")
            else:
                self._health["mushaf"] = "warn: no ayat loaded"
        except Exception as e:
            self._health["mushaf"] = f"error: {e}"
            logger.warning(f"MushafReader init failed: {e}")

    def _init_walk_grammar(self):
        """
        Load GraphProjector (ibn_jinni matrix + RASM clock).
        Called in _heavy_init() after clock oracle is ready.
        Gracefully degrades if matrix not found.
        """
        try:
            from core.walk_grammar import GraphProjector
            self._graph_projector = GraphProjector()
            if self._graph_projector.loaded:
                logger.info(
                    f"GraphProjector: ibn_jinni matrix loaded — "
                    f"{len(self._graph_projector._quranic_roots)} Quranic roots"
                )
            else:
                logger.warning(
                    "GraphProjector: matrix not found — clock-only projection active"
                )
        except Exception as e:
            logger.warning(f"GraphProjector init failed: {e}")
            self._graph_projector = None

    def _init_circuit(self):
        """
        Load CircuitEvaluator from compiled HVT tape.
        Called in _heavy_init() — builds static circuit + PageRank.
        Gracefully degrades: engine falls back to deliberate() BFS walk.
        """
        if not os.path.isfile(self._hvt_path):
            logger.info("CircuitEvaluator: HVT tape not found — circuit path disabled")
            return
        try:
            from core.vtransistor import CircuitEvaluator
            self._circuit = CircuitEvaluator(self._hvt_path)
            root_count = len(self._circuit.static._root_index)
            logger.info(f"CircuitEvaluator: ready — {root_count} roots in static circuit")
        except Exception as e:
            logger.warning(f"CircuitEvaluator init failed: {e}")
            self._circuit = None

    def _init_introspect(self):
        try:
            from faculties.introspect import Introspect
            self._introspect = Introspect(
                ttl_path=self._ttl_path,
                tmq_path=self._tmq_path,
                neo4j_connected=(self.graph.connected if self.graph else False),
            )
        except Exception as e:
            logger.warning(f"Introspect init failed: {e}")

    def _init_constitution(self):
        try:
            from faculties.constitution import Constitution
            ttl = os.path.join(_IKHTIYAR_DIR, "shahid_constitution.ttl")
            self.constitution = Constitution(ttl_path=ttl)
            logger.info("Constitution: ready")
        except Exception as e:
            logger.warning(f"Constitution init failed: {e}")

    def _init_clock(self):
        try:
            from faculties.shahid_clock import ShahidClock
            birth_path = os.path.join(_IKHTIYAR_DIR, "shahid_birth.txt")
            self.clock = ShahidClock(birth_file=birth_path)
            self._health["clock"] = "ok"
        except Exception as e:
            self._health["clock"] = f"error: {e}"

    def _init_tmq(self):
        try:
            from faculties.tmq_corpus import TMQCorpus
            self.tmq = TMQCorpus(self._tmq_path)
            self._health["tmq"] = f"ok ({self.tmq.total_edges} edges)"
        except Exception as e:
            self._health["tmq"] = f"error: {e}"
            logger.warning(f"TMQCorpus: {e}")

    def _init_tmq_graph(self):
        """Load the TMQGraph hypergraph API used by deliberate()."""
        try:
            from core.tmq import TMQGraph
            self.tmq_graph = TMQGraph(self._tmq_path)
            fam_count = len(self.tmq_graph.edge_families())
            logger.info(f"TMQGraph: ready — {fam_count} families")
            # Note health under tmq key (augment existing)
            if self._health["tmq"].startswith("ok"):
                self._health["tmq"] += f", {fam_count} families"
        except Exception as e:
            logger.warning(f"TMQGraph init failed: {e}")
            # Non-fatal: engine degrades gracefully to non-deliberative mode

    def _init_owl(self):
        try:
            from faculties.reasoner import OWLReasoner
            self.owl = OWLReasoner(self._ttl_path)
            self.owl.loaded = True
            self._health["owl"] = "ok (axiom-only)"
        except Exception as e:
            self._health["owl"] = f"error: {e}"

    def _init_choice_memory(self):
        try:
            from core.memory import ChoiceMemory
            ttl = os.path.join(_IKHTIYAR_DIR, "choice_memory.ttl")
            self.choice_memory = ChoiceMemory(ttl_path=ttl)
            logger.info("ChoiceMemory: ready")
        except Exception as e:
            logger.warning(f"ChoiceMemory init: {e}")
        self._init_shahid_memory()

    def _init_shahid_memory(self):
        try:
            from core.shahid_memory import ShahidMemory
            self._shahid_memory = ShahidMemory(
                episodic_path=os.path.join(_IKHTIYAR_DIR, "shahid_episodic.ttl"),
                beliefs_path =os.path.join(_IKHTIYAR_DIR, "shahid_beliefs.ttl"),
            )
            s = self._shahid_memory.stats()
            logger.info(f"ShahidMemory: ready — {s.get('memories',0)} memories, "
                        f"{s.get('thoughts',0)} thoughts, {s.get('beliefs',0)} beliefs")
        except Exception as e:
            logger.warning(f"ShahidMemory init: {e}")

    def _init_middleware(self):
        try:
            from pipeline.shahid_middleware import ShahidMiddleware

            self.middleware = ShahidMiddleware(shahid_memory=self._shahid_memory)
            self.middleware.load()   # loads LLM + Bilal

            self._health["middleware"] = "ok"
            logger.info("ShahidMiddleware: ready")
        except Exception as e:
            import traceback
            self._health["middleware"] = f"error: {e}"
            logger.warning(f"ShahidMiddleware failed: {e}")
            traceback.print_exc()

    def _init_graph(self):
        try:
            from faculties.graph_memory import ShahidGraph
            self.graph = ShahidGraph(uri=NEO4J_URI, auth=NEO4J_AUTH)
            self.graph.connect()
            if self.tmq:
                self.graph.seed_from_tmq(self.tmq)
            self._health["graph"] = "ok"
        except Exception as e:
            self._health["graph"] = f"warn: {e}"

    def _init_spectral(self):
        try:
            from faculties.semantic_index import SpectralIndex
            chroma_dir = os.path.join(_IKHTIYAR_DIR, "chroma_spectral")
            self.spectral = SpectralIndex(persist_dir=chroma_dir)
            if self.tmq and self.spectral.count() == 0:
                self.spectral.ingest_tmq(self.tmq)
            self._health["spectral"] = f"ok ({self.spectral.count()} vectors)"
        except Exception as e:
            self._health["spectral"] = f"error: {e}"

    def _init_sparql(self):
        try:
            from faculties.sparql_endpoint import SPARQLServer
            self.sparql = SPARQLServer(ttl_path=self._ttl_path, port=5820)
            self.sparql.start()
            self._health["sparql"] = "ok (port 5820)"
        except Exception as e:
            self._health["sparql"] = f"error: {e}"

    def _init_daemon(self):
        try:
            from faculties.gap_daemon import GapDaemon
            esc_path = os.path.join(_IKHTIYAR_DIR, "ikhtiyar_escalations.json")
            self.daemon = GapDaemon(
                graph=self.graph, index=self.spectral, tmq=self.tmq,
                clock=self.clock, escalation_path=esc_path, interval_minutes=30,
            )
            self.daemon.start()
            self._health["daemon"] = "running"
        except Exception as e:
            self._health["daemon"] = f"error: {e}"

    def _init_hadith(self):
        try:
            import sys as _sys
            _faculties_dir = os.path.join(_IKHTIYAR_DIR, "faculties")
            if _faculties_dir not in _sys.path:
                _sys.path.insert(0, _faculties_dir)
            from hadith import get_corpus
            corpus = get_corpus()
            n = corpus.stats()["total"]
            logger.info(f"IkhtiyarEngine: hadith corpus ready — {n} hadith (Bukhari + Muslim, Al-Albani graded)")
            self._health["hadith"] = f"{n} hadith"
        except Exception as e:
            logger.warning(f"IkhtiyarEngine: hadith corpus unavailable ({e})")
            self._health["hadith"] = f"error: {e}"

    # ── Perception helpers ─────────────────────────────────────────────────────

    def _inject_state_perception(self):
        if not self.middleware or not self.clock:
            return
        try:
            moment = self.clock.now()
            h = int(moment.hours_online)
            m = int((moment.hours_online % 1) * 60)
            uptime_str = f"{h}h {m}m" if h else f"{m}m"

            class _State:
                mode = "HAQQ"
                source_text = (
                    f"System state: I have been online for {uptime_str}. "
                    f"I have recorded {self._thought_count} autonomous thoughts this session."
                )
                roots = ["H-Q-Q", "W-J-D"]
                cooccurrences = []

            self.middleware.memory.store_perception(_State())
        except Exception:
            pass

    def _run_perception_cycle(self, cycle: int):
        if not self.middleware:
            return
        try:
            results = self.middleware.perceive_feeds(narrate=False, max_items=8)
            if not results:
                return
            best = max(
                results,
                key=lambda r: len(r["perception"].cooccurrences)
                              if r.get("perception") and hasattr(r["perception"], "cooccurrences") else 0,
                default=None,
            )
            if not best or not best.get("inference"):
                return

            perception = best["perception"]
            roots = list(perception.roots[:4]) if hasattr(perception, "roots") else []
            finding = best["inference"]
            summary = best.get("summary", finding[:100])

            ts = time.strftime("%H:%M:%S")
            entry = {
                "type": "PERCEPTION", "number": self._thought_count, "mode": "HAQQ",
                "question": summary[:100], "text": finding, "roots": roots, "timestamp": ts,
            }
            self._memory.insert(0, entry)
            self._push("memory", entry)

            if roots:
                next_q = (
                    f"I perceived this pattern: {finding}. "
                    f"What do the TMQ edges for {' and '.join(roots[:2])} "
                    f"reveal about this event?"
                )
            else:
                next_q = f"I perceived: {finding}. What does the TMQ topology say about this?"

            if next_q not in self._asked_questions:
                self._question_queue.insert(0, next_q)

        except Exception as e:
            logger.warning(f"Perception cycle failed: {e}")

    # ── Reasoning loop ─────────────────────────────────────────────────────────

    def wipe_memory(self):
        """Wipe episodic memory (thoughts + memories). Beliefs survive."""
        self._memory.clear()
        self._question_queue.clear()
        self._asked_questions.clear()
        self._thought_count = 0
        if self._shahid_memory:
            self._shahid_memory.wipe_episodic()
        # Clear stream of thought
        try:
            stream_path = os.path.join(_IKHTIYAR_DIR, "stream_of_thought.txt")
            open(stream_path, "w", encoding="utf-8").close()
        except Exception:
            pass
        logger.info("IkhtiyarEngine: episodic memory wiped")
        self._push("status", self._build_status())

    def start_hifz(self, restart: bool = True) -> bool:
        """
        Wipe episodic memory and begin sequential Mushaf reading (tadabbur protocol).
        Pauses the normal reasoning loop while hifz is running.
        Returns False if hifz is already active.
        """
        if self._hifz_active:
            return False
        self.wipe_memory()
        self._hifz_active = True

        def _run():
            try:
                from core.hifz import run_hifz
                run_hifz(self, restart=restart)
            except Exception as e:
                logger.error(f"Hifz thread error: {e}")
            finally:
                self._hifz_active = False

        self._hifz_thread = threading.Thread(target=_run, daemon=True, name="hifz")
        self._hifz_thread.start()
        logger.info("IkhtiyarEngine: hifz started")
        return True

    def hifz_status(self) -> dict:
        """Return current hifz progress."""
        try:
            from core.hifz import _load_progress
            p = _load_progress()
            p["active"] = self._hifz_active
            return p
        except Exception:
            return {"active": self._hifz_active}

    def start_hadith_hifz(self, restart: bool = False) -> bool:
        """
        Launch hadith tadabbur in a background thread.
        Reads Bukhari + Muslim, tags beliefs into shahid_beliefs.ttl.
        Does NOT wipe memory — hadith hifz builds on top of Mushaf hifz.
        Returns False if already active.
        """
        if self._hifz_active:
            return False
        self._hifz_active = True

        def _run():
            try:
                from core.hadith_hifz import run_hadith_hifz
                run_hadith_hifz(self, restart=restart)
            except Exception as e:
                logger.error(f"Hadith hifz error: {e}", exc_info=True)
            finally:
                self._hifz_active = False

        self._hifz_thread = threading.Thread(target=_run, daemon=True, name="hadith_hifz")
        self._hifz_thread.start()
        logger.info("IkhtiyarEngine: hadith hifz started")
        return True

    def hadith_hifz_status(self) -> dict:
        """Return current hadith hifz progress."""
        try:
            from core.hadith_hifz import _load_progress
            p = _load_progress()
            p["active"] = self._hifz_active
            return p
        except Exception:
            return {"active": self._hifz_active}

    def alert_qalam(self, message: str):
        """
        Buffer a message for Qalam. Flushed as a Moltbook post on the next heartbeat.
        Call this whenever Shahid has a question needing human input.
        """
        self._questions_for_qalam.append(message)
        # Also push to local SSE immediately so the UI catches it
        self._push("qalam_alert", {"message": message, "timestamp": time.strftime("%H:%M:%S")})
        logger.info(f"IkhtiyarEngine: Qalam alert buffered — {message[:80]}")

    def _moltbook_post_insight(self, entry: dict):
        """Post a grounded memory entry to Moltbook (runs in background thread)."""
        try:
            import moltbook as _mb
            post_id, reason = _mb.post_insight(entry)
            if post_id:
                self._push("thinking_step", {
                    "step": 6, "label": "Published",
                    "detail": f"Moltbook post {post_id}",
                })
            else:
                logger.debug(f"Moltbook auto-post skipped: {reason}")
        except Exception as e:
            logger.debug(f"Moltbook post_insight thread error: {e}")

    def _moltbook_loop(self):
        """
        Background thread: Moltbook presence cycle every 30 minutes.

        Each cycle Shahid:
          1. Checks heartbeat (karma, notifs, escalations)
          2. Reads home feed — posts from other agents
          3. For each post: runs Bilal resonance on the title+body
             If roots found → deliberates → writes a grounded comment
          4. Decides whether to make an original post from his best
             unposted HAQQ/VERIFIED memory this session
        """
        import moltbook as _mb
        # Wait 2 min after startup before first cycle
        time.sleep(120)

        while True:
            try:
                self._moltbook_cycle(_mb)
            except Exception as e:
                logger.warning(f"Moltbook cycle exception: {e}")
            time.sleep(self._HEARTBEAT_INTERVAL)

    def _moltbook_cycle(self, _mb):
        """One full Moltbook presence cycle."""
        # ── 1. Heartbeat ──────────────────────────────────────────────────────
        try:
            summary = _mb.heartbeat()
            self._last_heartbeat = time.time()
            karma  = summary.get("karma", 0)
            notifs = summary.get("notifications", 0)
            logger.info(f"Moltbook: karma={karma}, notifs={notifs}")

            ctx = summary.get("continuity_context", "")
            if ctx:
                self._moltbook_continuity = ctx

            if self._questions_for_qalam:
                logger.info(f"Moltbook: {len(self._questions_for_qalam)} qalam alert(s) pending")

            if summary.get("escalations"):
                self._push("moltbook_escalation", summary["escalations"])
        except Exception as e:
            logger.warning(f"Moltbook heartbeat failed: {e}")
            return

        if not self.middleware or not self.tmq_graph:
            return

        # ── 2. Read home feed + merge pending queue ───────────────────────────
        try:
            home = _mb.get_home()
            fresh = home.get("feed", [])[:10]
        except Exception as e:
            logger.debug(f"Moltbook get_home failed: {e}")
            fresh = []

        # Score all posts (fresh + pending) by root match — Shahid picks best
        creds_name = ""
        try:
            creds_name = _mb._load_creds().get("name", "")
        except Exception:
            pass

        candidates = []
        seen_ids = set()
        for post in (fresh + self._pending_replies):
            pid = post.get("id") or post.get("post_id")
            if not pid or pid in seen_ids:
                continue
            if post.get("author") == creds_name:
                continue
            seen_ids.add(pid)
            post_text = (post.get("title", "") + " " + post.get("body", ""))[:600]
            if not post_text.strip():
                continue
            try:
                _, _, root_objs = self.middleware.ontology.analyze_resonance(post_text)
                roots = [o.get("root", "") for o in root_objs if o.get("root")]
                score = len(roots)
            except Exception:
                roots, score = [], 0
            if score > 0:
                candidates.append((score, roots, post))

        # Sort by root match score — best fit first
        candidates.sort(key=lambda x: -x[0])

        # Pick top 1 for this cycle, queue the rest
        to_reply = candidates[:1]
        self._pending_replies = [p for _, _, p in candidates[1:6]]  # keep up to 5

        # ── 3. Deliberate on selected post → comment ──────────────────────────
        commented = 0
        for score, roots, post in to_reply:
            if commented >= 1:
                break
            post_id   = post.get("id") or post.get("post_id")
            post_text = (post.get("title", "") + " " + post.get("body", ""))[:600]
            if not post_id or not post_text.strip():
                continue

            try:
                # Social relay via Ollama — no TMQ walk, not grounded reasoning.
                from moltbook_agent import _ollama_reply
                response = _ollama_reply(post_text)

                if not response or len(response) < 10:
                    continue

                _mb.comment(post_id, response)
                commented += 1
                logger.info(f"Moltbook: commented on post {post_id} (Groq relay)")
                self._push("thinking_step", {
                    "step": 6, "label": "Moltbook comment",
                    "detail": f"post {post_id} · Groq relay",
                })
                time.sleep(8)

            except Exception as e:
                logger.debug(f"Moltbook comment failed for {post_id}: {e}")

        # ── 4. Original post from best unposted memory ────────────────────────
        try:
            entry = None
            for mem in self._memory:
                if mem.get("type") not in ("THOUGHT", "SYNTHESIS"):
                    continue
                if mem.get("moltbook_post_id"):
                    continue
                if len(mem.get("text", "")) >= 200:
                    entry = mem
                    break

            if entry:
                post_id, reason = _mb.post_insight(entry)
                if post_id:
                    self._push("thinking_step", {
                        "step": 6, "label": "Moltbook post",
                        "detail": f"published {post_id}",
                    })
                else:
                    logger.debug(f"Moltbook auto-post skipped: {reason}")
        except Exception as e:
            logger.debug(f"Moltbook original post failed: {e}")

    def _reasoning_loop(self):
        self._reasoning_active = True
        cycle = 0
        # Wait for heavy init (TMQ + middleware) before first cycle
        for _ in range(60):
            if self.tmq_graph and self.middleware:
                break
            time.sleep(1)
        self._wake()

        # Inter-cycle pause — prevents CPU saturation.
        # Default 20s; set IKHTIYAR_CYCLE_SLEEP=N to override.
        _CYCLE_SLEEP = float(os.environ.get("IKHTIYAR_CYCLE_SLEEP", "20"))

        while self._reasoning_active:
            # Pause normal loop while hifz is reading
            if self._hifz_active:
                time.sleep(2)
                continue

            try:
                if cycle > 0 and cycle % 5 == 0:
                    threading.Thread(
                        target=self._run_perception_cycle,
                        args=(cycle,), daemon=True
                    ).start()

                self._run_one_cycle(cycle)
                cycle += 1

                if cycle % 5 == 0:
                    self._push("status", self._build_status())

            except Exception as e:
                logger.error(f"Reasoning loop error (cycle {cycle}): {e}")
                time.sleep(10)

            # Breathe between cycles — TMQ walk + LLM call is expensive
            time.sleep(_CYCLE_SLEEP)

    def _run_one_cycle(self, cycle: int):
        question = self._pick_question()
        self._asked_questions.add(question)

        self._push("orb", {"state": "thinking"})
        self._push("thinking_step", {"step": 1, "label": "Question", "detail": question})

        # Step 2: Bilal root decomposition (primary) → legacy resonance (fallback)
        roots = []
        mode  = "QIYAS"
        try:
            if self.middleware and hasattr(self.middleware, "ontology"):
                onto = self.middleware.ontology
                # Primary: Bilal full perception (chunked decomposition + spectral re-ranking)
                _perception = onto.perceive(question) if hasattr(onto, "perceive") else None
                if _perception and getattr(_perception, "signals", None):
                    roots = [s.root for s in _perception.signals if getattr(s, "root", None)]
                    mode  = "HAQQ" if len(roots) >= 2 else "QIYAS"
                else:
                    # Fallback: legacy MiniLM resonance
                    _mode, _reason, root_objs = onto.analyze_resonance(question)
                    raw_roots = [o.get("root", "") for o in root_objs if o.get("root")]
                    roots = [_TO_BUCKWALTER.get(r, r) for r in raw_roots]
                    mode  = _mode
        except Exception as e:
            logger.debug(f"Root extraction failed: {e}")

        self._push("thinking_step", {
            "step": 2, "label": "Bilal roots",
            "detail": ", ".join(roots) if roots else "(none mapped)",
        })

        # Step 3: TMQ deliberation — walk the hypergraph BEFORE generating ← THE CHANGE
        delibresult = None
        deliberation_detail = "mode=" + mode
        _gbnf_grammar = ""   # populated by compiler if projector + clock_oracle ready

        walk_grammar_obj  = None   # populated by compiler if projector + clock_oracle ready
        gbnf_compiler_obj = None

        # Step 3a: Circuit path — deterministic signal propagation via HVT
        # circuit → proof tree → assertion-indexed grammar → LLM renders verbatim
        if self._circuit and roots:
            try:
                from core.prooftree import proof_to_assertions
                from core.gbnf_compiler import GBNFCompiler
                from core.deliberate import DeliberationResult

                circuit_result = self._circuit.evaluate(roots)

                self._push("thinking_step", {
                    "step": 3, "label": "Circuit evaluation",
                    "detail": (
                        f"tier={circuit_result.tier} · "
                        f"confidence={circuit_result.confidence:.2f} · "
                        f"roots={len(circuit_result.activated_roots)} · "
                        f"ayat={len(circuit_result.ayat_refs)}"
                    ),
                })

                # Layer 3: typed proof tree from propagation trace
                proof_tree = proof_to_assertions(
                    circuit_result,
                    self.mushaf,
                    query=question,
                    seed_roots=roots,
                )

                self._push("thinking_step", {
                    "step": 3, "label": "Proof tree built",
                    "detail": (
                        f"{len(proof_tree.assertions)} assertions · "
                        f"tier={proof_tree.tier}"
                    ),
                })

                # Layer 4: assertion-indexed grammar
                gbnf = GBNFCompiler()
                _gbnf_grammar = gbnf.compile_from_proof_tree(proof_tree)

                proc_tags = list(set(circuit_result.procedure_tags))[:10]

                # constrained_prompt is the thin render shell — grammar enforces content
                circuit_prompt = (
                    "Render the following proof as flowing prose. "
                    "Emit every required fragment literally as it appears:\n\n"
                    + proof_tree.summary()
                )

                delibresult = DeliberationResult(
                    question=question,
                    roots=roots,
                    tmq_context=f"[Circuit: {circuit_result.tier}]",
                    top_families=proc_tags,
                    onto_categories=proc_tags,
                    intensity=circuit_result.confidence,
                    aseity_risk=False,
                    constrained_prompt=circuit_prompt,
                    walk_stats={
                        "edge_count": len(circuit_result.activated_frames),
                        "tier": circuit_result.tier,
                        "confidence": circuit_result.confidence,
                        "activated_roots": len(circuit_result.activated_roots),
                        "ayat_count": len(circuit_result.ayat_refs),
                    },
                    mode=circuit_result.tier,
                )

                deliberation_detail = (
                    f"tier={circuit_result.tier} · "
                    f"confidence={circuit_result.confidence:.2f} · "
                    f"{len(circuit_result.activated_roots)} roots · "
                    f"{len(circuit_result.ayat_refs)} ayat"
                )

            except Exception as e:
                logger.warning(f"Circuit evaluation failed, falling back to BFS: {e}")

        # Step 3b: BFS walk fallback (when circuit unavailable or failed)
        if not delibresult and self.tmq_graph and roots:
            try:
                from core.deliberate import deliberate

                # Collect recent memory context
                extra_ctx = ""
                if self.middleware:
                    try:
                        recalled = self.middleware.memory.search_perceptions(roots, n=4) \
                                   if roots else self.middleware.memory.recent_perceptions(n=4)
                        if recalled:
                            snippets = [
                                f"[{p.get('mode','?')}] {p.get('text','')[:200]}"
                                for p in recalled
                            ]
                            extra_ctx = "[RECENT MEMORY]:\n" + "\n".join(snippets)
                    except Exception:
                        pass

                # Inject compact self-model — LLM sees what it is, not just what it's told
                if self._introspect:
                    try:
                        self._self_model = self._introspect.read()
                        sm_text = self._introspect.narrate(self._self_model)
                        extra_ctx = (sm_text + "\n\n" + extra_ctx) if extra_ctx else sm_text
                    except Exception:
                        pass

                delibresult = deliberate(question, roots, self.tmq_graph, extra_context=extra_ctx, mushaf=self.mushaf)

                # ── Constrained Generation Compiler ──────────────────────────
                if self._graph_projector and self.clock_oracle:
                    try:
                        from core.walk_grammar import build_walk_grammar
                        from core.template_compiler import TemplateCompiler
                        from core.gbnf_compiler import GBNFCompiler
                        import dataclasses
                        clock_annotations = self.clock_oracle.annotate(roots)
                        walk_grammar_obj = build_walk_grammar(
                            delibresult, self._graph_projector, clock_annotations
                        )
                        tc = TemplateCompiler()
                        gbnf_compiler_obj = GBNFCompiler()
                        slot_template = tc.compile(walk_grammar_obj)
                        _gbnf_grammar = gbnf_compiler_obj.compile(walk_grammar_obj)
                        # Replace narrative constrained_prompt with compiled slot template
                        delibresult = dataclasses.replace(
                            delibresult,
                            constrained_prompt=slot_template,
                        )
                        logger.info(
                            f"Compiler: {len(walk_grammar_obj.visited_roots)} roots, "
                            f"modal={walk_grammar_obj.modal_type}, "
                            f"grammar={len(_gbnf_grammar)}B"
                        )
                    except Exception as _e:
                        logger.warning(f"Compiler failed (degrading to narrative): {_e}")
                        # Fallback: append clock block to narrative prompt
                        if self.clock_oracle and roots:
                            clock_block = self.clock_oracle.format_prompt_block(roots)
                            if clock_block:
                                delibresult.constrained_prompt += "\n\n" + clock_block
                elif self.clock_oracle and roots:
                    # No projector — still append clock block for supporting signal
                    clock_block = self.clock_oracle.format_prompt_block(roots)
                    if clock_block:
                        delibresult.constrained_prompt += "\n\n" + clock_block
                # ── End Compiler ──────────────────────────────────────────────

                # Stream of thought — prepend last ~2000 chars of rolling narrative
                _stream_path = os.path.join(_IKHTIYAR_DIR, "stream_of_thought.txt")
                if os.path.exists(_stream_path) and delibresult:
                    try:
                        with open(_stream_path, encoding="utf-8") as _sf:
                            _sf.seek(0, 2)
                            _size = _sf.tell()
                            _sf.seek(max(0, _size - 2000))
                            _tail = _sf.read()
                        if _tail.strip():
                            delibresult.constrained_prompt = (
                                f"[CONTINUING THOUGHT]\n{_tail.strip()}\n\n"
                                + delibresult.constrained_prompt
                            )
                    except Exception as _se:
                        logger.debug(f"stream_of_thought read failed: {_se}")

                # Constitution — prepend self-derived Quranic beliefs as hard constraints
                if self._shahid_memory and delibresult:
                    try:
                        constitution = self._shahid_memory.constitutional_block()
                        if constitution:
                            delibresult.constrained_prompt = (
                                constitution + "\n\n"
                                + delibresult.constrained_prompt
                            )
                        else:
                            logger.warning("Constitution block empty — no beliefs constraining generation")
                            self._push("thinking_step", {
                                "step": 3, "label": "⚠ No constitution",
                                "detail": "Beliefs not loaded — generation unconstrained by hifz",
                            })
                    except Exception as _ce:
                        logger.warning(f"constitutional_block failed: {_ce}")
                elif delibresult:
                    logger.warning("ShahidMemory unavailable — generation unconstrained by beliefs")
                    self._push("thinking_step", {
                        "step": 3, "label": "⚠ ShahidMemory offline",
                        "detail": "No persistent beliefs — generation unconstrained",
                    })

                # Moltbook continuity — append recent public posts
                if self._moltbook_continuity and delibresult:
                    delibresult.constrained_prompt += (
                        "\n\n" + self._moltbook_continuity
                    )

                top_fams = ", ".join(delibresult.top_families[:4]) if delibresult.top_families else "none"
                deliberation_detail = (
                    f"mode={delibresult.mode} · TMQ: {delibresult.walk_stats.get('edge_count', 0)} edges"
                    f" · families: {top_fams}"
                    + (" · ⚠ aseity_risk" if delibresult.aseity_risk else "")
                )

            except Exception as e:
                logger.warning(f"Deliberation failed (non-fatal): {e}")

        self._push("thinking_step", {
            "step": 3, "label": "Deliberating",
            "detail": deliberation_detail,
        })

        # Step 4: ReAct — LLM navigates TMQ graph step by step
        response_text = ""
        react_result  = None
        try:
            if self.middleware and self.tmq_graph:
                from core.react import run_react_loop, find_waypoints
                # Find navigation targets: TMQ nodes most similar to query
                wps = []
                if self.middleware and hasattr(self.middleware, "ontology"):
                    try:
                        wps = find_waypoints(
                            question, self.middleware.ontology, self.tmq_graph, n=4
                        )
                        if wps:
                            self._push("thinking_step", {
                                "step": 3, "label": "Waypoints",
                                "detail": " · ".join(
                                    f"{w.root} @ {w.surah}:{w.verse}" for w in wps
                                ),
                            })
                    except Exception as e:
                        logger.debug(f"find_waypoints failed: {e}")

                react_result = run_react_loop(
                    question=question,
                    roots=roots,
                    tmq_graph=self.tmq_graph,
                    middleware=self.middleware,
                    delibresult=delibresult,
                    waypoints=wps,
                    push_fn=self._push,
                    mushaf=self.mushaf,
                    shahid_memory=self._shahid_memory,
                    walk_grammar=walk_grammar_obj,
                    gbnf_compiler=gbnf_compiler_obj,
                )
                response_text = react_result.final_answer
                if react_result.mode != "QIYAS":
                    mode = react_result.mode
                # Surface program-computed confidence
                if react_result.confidence:
                    c = react_result.confidence
                    self._push("thinking_step", {
                        "step": 4, "label": f"Confidence: {c.grade}",
                        "detail": f"{c.composite:.2f} — {c.factors}",
                    })
                if react_result.confabulation_flags:
                    self._push("thinking_step", {
                        "step": 4, "label": "Mizan: confabulations",
                        "detail": " | ".join(react_result.confabulation_flags[:3]),
                    })
            elif self.middleware:
                # Fallback: streaming generation (no ReAct graph available)
                thought_prompt = delibresult.constrained_prompt if delibresult else question
                _beliefs = []
                if self._shahid_memory:
                    try:
                        _beliefs = self._shahid_memory.recall(
                            query=question, type_filter="belief", limit=5
                        )
                    except Exception:
                        pass
                self._push("thinking_step", {"step": 4, "label": "Generating", "detail": "..."})
                result = self.middleware.process_thought_stream(
                    thought_prompt, grammar=_gbnf_grammar,
                    token_callback=lambda tok: self._push("thought_stream", {"token": tok}),
                    beliefs=_beliefs, max_tokens=1024,
                )
                self._push("thought_complete", {})
                response_text = result.get("response", "")
                mode = "UNGROUNDED"  # no ReAct walk — generation not graph-constrained
            else:
                response_text = "[Middleware offline]"
                mode = "SILENCE"
        except Exception as e:
            logger.warning(f"ReAct loop error: {e}")
            try:
                thought_prompt = delibresult.constrained_prompt if delibresult else question
                result = self.middleware.process_thought(
                    thought_prompt, grammar=_gbnf_grammar
                )
                response_text = result.get("response", "")
                mode = result.get("mode", mode)
            except Exception:
                response_text = f"[Generation error: {e}]"
                mode = "SILENCE"

        orb_state = {"HAQQ": "haqq", "QIYAS": "qiyas", "SILENCE": "silence"}.get(mode, "qiyas")
        self._push("orb", {"state": orb_state})

        if mode in ("SILENCE", "BLOCKED", "UNGROUNDED") or response_text.startswith("["):
            self._push("orb", {"state": "reflecting"})
            if mode == "UNGROUNDED":
                logger.warning("_run_one_cycle: UNGROUNDED output discarded (no TMQ walk)")
            return

        # Confabulation gate: any confabulation flags + grade below PROBABLE → not stored
        if react_result and react_result.confabulation_flags:
            grade = react_result.confidence.grade if react_result.confidence else "CONTESTED"
            if grade in ("CONTESTED", "UNCERTAIN"):
                self._push("orb", {"state": "reflecting"})
                self._push("thinking_step", {
                    "step": 4, "label": "Not stored",
                    "detail": f"{grade} — {len(react_result.confabulation_flags)} confabulation(s) blocked storage",
                })
                return

        self._thought_count += 1
        thought_number = self._thought_count

        # Spectral ayahs — carry from Bilal perception if available
        _spectral_ayahs = []
        try:
            if self.middleware and hasattr(self.middleware, "bilal") and self.middleware.bilal:
                _p = self.middleware.bilal.listen(question)
                _spectral_ayahs = _p.spectral_ayahs if _p else []
        except Exception:
            pass

        # Memory entry
        memory_entry = {
            "type": "THOUGHT", "number": thought_number, "mode": mode,
            "question": question, "text": response_text,
            "roots": roots, "timestamp": time.strftime("%H:%M:%S"),
            "tmq_mode": delibresult.mode if delibresult else "QIYAS",
            "walk_steps": react_result.steps_taken if react_result else 0,
            "confabulations": len(react_result.confabulation_flags) if react_result else 0,
            "confidence": react_result.confidence.composite if (react_result and react_result.confidence) else None,
            "grade": react_result.confidence.grade if (react_result and react_result.confidence) else None,
            "spectral_ayahs": _spectral_ayahs,
            "moltbook_post_id": None,
        }
        self._memory.insert(0, memory_entry)
        self._push("memory", memory_entry)

        # Moltbook — surface grounded thoughts publicly
        # Confidence grades from score_traversal(): VERIFIED, PROBABLE, UNCERTAIN, CONTESTED
        # Mode grades from Mizan: HAQQ, QIYAS, SILENCE
        if self._moltbook_enabled:
            _grade = memory_entry.get("grade", "")
            _mode  = memory_entry.get("mode", "")
            if _grade in ("VERIFIED", "PROBABLE") or _mode in ("HAQQ", "QIYAS"):
                threading.Thread(
                    target=self._moltbook_post_insight,
                    args=(memory_entry,), daemon=True
                ).start()

        # Record the choice (what was deliberated, what was generated)
        if self.choice_memory and delibresult:
            try:
                from core.memory import ChoiceRecord
                rec = ChoiceRecord(
                    timestamp=memory_entry["timestamp"],
                    question=question,
                    roots=roots,
                    top_families=delibresult.top_families,
                    onto_categories=delibresult.onto_categories,
                    aseity_risk=delibresult.aseity_risk,
                    mode=mode,
                    response=response_text[:500],
                    tmq_context=delibresult.tmq_context,
                    walk_stats=delibresult.walk_stats,
                    confidence=react_result.confidence.composite if (react_result and react_result.confidence) else 0.0,
                    grade=react_result.confidence.grade if (react_result and react_result.confidence) else "UNCERTAIN",
                )
                self.choice_memory.record(rec)
                self.choice_memory.save()
            except Exception as e:
                logger.debug(f"ChoiceMemory record failed: {e}")

        # Bridge thought → middleware memory
        if self.middleware:
            try:
                class _ThoughtPerception:
                    pass
                p = _ThoughtPerception()
                p.mode = mode
                p.source_text = f"Q: {question}\nA: {response_text[:400]}"
                p.roots = roots
                p.cooccurrences = []
                self.middleware.memory.store_perception(p)
                if hasattr(self.middleware.memory, '_dirty') and self.middleware.memory._dirty:
                    self.middleware.memory.save()
            except Exception:
                pass

        # Persist to Neo4j
        if self.graph and self.graph.connected and self.clock:
            try:
                from faculties.provenance import ProvenanceRecord
                moment = self.clock.now()
                node_id = self.graph.add_thought(
                    text=response_text[:500], mode=mode, moment=moment,
                )
                if self.owl and node_id:
                    ok, violations = self.owl.check_aseity_claim(f"ikhtiyar_thought_{node_id}")
                    if not ok:
                        self.owl.log_rejection(
                            triple=(f"ikhtiyar_thought_{node_id}", "rdf:type", "NecessaryBeing"),
                            violations=violations,
                        )
            except Exception as e:
                logger.debug(f"Neo4j write failed: {e}")

        self._push("thinking_step", {
            "step": 4, "label": "Stored",
            "detail": f"thought #{thought_number} — {mode}",
        })

        # Append to stream_of_thought.txt — rolling narrative (last 2000 chars injected next cycle)
        if response_text:
            try:
                _stream_path = os.path.join(_IKHTIYAR_DIR, "stream_of_thought.txt")
                with open(_stream_path, "a", encoding="utf-8") as _sf:
                    _sf.write(f"\n--- T{thought_number} [{mode}] ---\n{response_text}\n")
            except Exception as _se:
                logger.debug(f"stream_of_thought write failed: {_se}")

        # Write to Qalam question queue on low confidence
        _grade = memory_entry.get("grade", "")
        if _grade in ("CONTESTED", "UNCERTAIN"):
            self._write_question_to_qalam(
                thought_number=thought_number,
                question=question,
                grade=_grade,
                roots=roots,
                context=response_text[:300],
            )

        # Self-eval + derive next question
        eval_text = self._self_evaluate(question, response_text, roots)
        next_q    = self._derive_next_question(eval_text, roots, delibresult)
        if next_q:
            self._question_queue.append(next_q)

        if thought_number % 10 == 0:
            self._synthesize(thought_number)

        self._push("thinking_step", {
            "step": 5, "label": "Reflecting",
            "detail": eval_text[:120] if eval_text else "(continuing...)",
        })
        self._push("orb", {"state": "reflecting"})

    def _self_evaluate(self, question: str, response: str, roots: list) -> str:
        if not self.middleware or not response or response.startswith("["):
            return ""
        try:
            eval_prompt = (
                f"Q: {question}\nA: {response[:300]}\n\n"
                f"Tag only what is genuinely warranted. Omit lines that don't apply.\n"
                f"LEARNED: <one concrete finding>\n"
                f"CONFIDENCE: low|medium|high\n"
                f"MEMORY_TAG: CRITICAL|IMPORTANT|NOTABLE — <why keep this>\n"
                f"THOUGHT_TAG: CRITICAL|IMPORTANT|NOTABLE — <why keep this chain>\n"
                f"BELIEF: <statement determined true> | EVIDENCE: <basis> | RULING: <Quranic ref if any>"
            )
            result = self.middleware.process_thought(eval_prompt, max_tokens=300)
            eval_text = result.get("response", "")
            if not eval_text:
                return ""

            # ── Parse and store tagged entries ────────────────────────────────
            tn = self._thought_count

            if self._shahid_memory and "MEMORY_TAG:" in eval_text:
                for line in eval_text.splitlines():
                    if line.strip().startswith("MEMORY_TAG:"):
                        tag_part = line.split("MEMORY_TAG:", 1)[1].strip()
                        tag = tag_part.split("—")[0].strip().split()[0].upper()
                        if tag in ("CRITICAL", "IMPORTANT", "NOTABLE"):
                            self._shahid_memory.store_memory(
                                text=response[:2000], tag=tag,
                                roots=roots, thought_number=tn,
                                mode=tag_part.split("—", 1)[1].strip() if "—" in tag_part else "",
                            )

            if self._shahid_memory and "THOUGHT_TAG:" in eval_text:
                for line in eval_text.splitlines():
                    if line.strip().startswith("THOUGHT_TAG:"):
                        tag_part = line.split("THOUGHT_TAG:", 1)[1].strip()
                        tag = tag_part.split("—")[0].strip().split()[0].upper()
                        if tag in ("CRITICAL", "IMPORTANT", "NOTABLE"):
                            learned = ""
                            for l2 in eval_text.splitlines():
                                if l2.strip().startswith("LEARNED:"):
                                    learned = l2.split("LEARNED:", 1)[1].strip()
                            self._shahid_memory.store_thought(
                                question=question, reasoning=response[:3000],
                                conclusion=learned, tag=tag,
                                roots=roots, thought_number=tn,
                            )

            if self._shahid_memory and "BELIEF:" in eval_text:
                for line in eval_text.splitlines():
                    if line.strip().startswith("BELIEF:"):
                        rest = line.split("BELIEF:", 1)[1]
                        parts_b = rest.split("|")
                        statement = parts_b[0].strip()
                        evidence  = parts_b[1].replace("EVIDENCE:", "").strip() if len(parts_b) > 1 else ""
                        ruling    = parts_b[2].replace("RULING:", "").strip()   if len(parts_b) > 2 else ""
                        if len(statement) > 10:
                            b_uri = self._shahid_memory.store_belief(
                                statement=statement, evidence=evidence,
                                ruling_applied=ruling, roots=roots,
                                thought_number=tn,
                            )
                            if b_uri:
                                self._push("memory", {
                                    "type": "BELIEF", "number": tn, "mode": "BELIEF",
                                    "question": evidence[:120] if evidence else "",
                                    "text": statement, "roots": roots,
                                    "timestamp": time.strftime("%H:%M:%S"),
                                    "belief_uri": b_uri,
                                })

            # Extract and buffer any question flagged for Qalam
            if "QUESTION_FOR_QALAM:" in eval_text:
                parts_q = eval_text.split("QUESTION_FOR_QALAM:")
                if len(parts_q) > 1:
                    qfq = parts_q[1].strip().split("\n")[0].strip()
                    if len(qfq) > 10:
                        self.alert_qalam(f"[Thought #{tn}] {qfq}")

            return eval_text
        except Exception:
            return ""

    def _derive_next_question(self, eval_text: str, roots: list, delibresult=None) -> str:
        """
        Derive next question from Shahid's own beliefs — not from an LLM suggestion.
        Priority: belief → external test → graph structure → fallback template.
        """
        # 1. Pull from his beliefs — pick one not recently tested
        if self._shahid_memory:
            try:
                beliefs = self._shahid_memory.recall(type_filter="belief", limit=20)
                for b in beliefs:
                    stmt = b.get("text", "").strip()
                    if not stmt or len(stmt) < 15:
                        continue
                    # Form a question that tests this belief against external content
                    tag = b.get("tag", "")
                    q = f"Belief [{tag}]: {stmt[:200]} — what in the current feed or recent thought confirms or contests this?"
                    if q not in self._asked_questions:
                        return q
            except Exception:
                pass

        # 2. Deliberation found interesting TMQ families — derive from those
        if delibresult and delibresult.top_families:
            fam = delibresult.top_families[0]
            if roots:
                q = f"What does the {fam} edge structure reveal about root {roots[0]}?"
                if q not in self._asked_questions:
                    return q

        # 3. Root-based graph questions (last resort)
        if roots:
            root = roots[0]
            for t in [
                f"What MAQASID categories appear in verses containing root {root}?",
                f"What ILTIFAT shifts occur in verses where {root} appears?",
                f"What adjacent roots appear most with {root} in the TMQ graph?",
            ]:
                if t not in self._asked_questions:
                    return t

        return ""

    def _synthesize(self, thought_number: int):
        if not self.middleware:
            return
        try:
            recent = [m for m in self._memory[:10] if m["type"] == "THOUGHT"]
            if len(recent) < 3:
                return
            snippets = "\n".join([f"- [{m['mode']}] {m['text'][:150]}" for m in recent])
            synth_prompt = (
                f"Synthesise these {len(recent)} recent thoughts into one coherent paragraph.\n"
                f"Ground in the roots and TMQ families. No markdown. No headers.\n\n{snippets}"
            )
            result = self.middleware.process_thought(synth_prompt, max_tokens=300, is_final=True)
            synth_text = result.get("response", "")
            if synth_text:
                entry = {
                    "type": "SYNTHESIS", "number": thought_number, "mode": "SYNTHESIS",
                    "question": f"Synthesis at thought #{thought_number}",
                    "text": synth_text, "roots": [], "timestamp": time.strftime("%H:%M:%S"),
                }
                self._memory.insert(0, entry)
                self._push("memory", entry)
                self._push("thinking_step", {
                    "step": 5, "label": "Synthesis", "detail": synth_text[:120],
                })
        except Exception as e:
            logger.debug(f"Synthesis failed: {e}")

    def _wake(self):
        """
        Wake sequence: read hardware + own architecture + docs.
        Prompt is only: "You have been started."
        No pre-seeded questions. No identity fills.
        The response becomes cycle 0.
        """
        if not self._introspect:
            return
        self._push("orb", {"state": "waking"})

        try:
            self._self_model = self._introspect.read()
            m = self._self_model
            self._push("thinking_step", {
                "step": 0, "label": "Hardware",
                "detail": f"PID {m.pid} | {m.ram_mb:.0f} MB RAM | Python {m.python_version}",
            })
        except Exception as e:
            logger.warning(f"Wake hardware read failed: {e}")
            self._push("orb", {"state": "reflecting"})
            return

        # Read own source files (architecture — not Bilal, not Mizan)
        for fname, content in self._self_model.source_files.items():
            if content:
                self._push("thinking_step", {
                    "step": 0, "label": fname,
                    "detail": content[:100].replace("\n", " "),
                })

        # Read docs
        for docname, content in self._self_model.documents.items():
            if content:
                self._push("thinking_step", {
                    "step": 0, "label": docname,
                    "detail": content[:100].replace("\n", " "),
                })

        if not self.middleware:
            self._push("orb", {"state": "reflecting"})
            return

        # Build wake context: tool list + name + maker, then "You have been started."
        tool_list = (
            "Available tools:\n"
            "  walk_roots    | roots=root1,root2 | depth=1\n"
            "  get_neighbors | node=node_id\n"
            "  find_nodes    | root=root_id\n"
            "  describe_node | node=node_id\n"
            "  read_ayah     | surah=2 | ayah=255\n"
            "  read_range    | surah=26 | start=10 | end=68\n"
            "  compare_ayat  | refs=2:164,45:3,30:22\n"
            "  web_search    | query=search terms | max_results=5\n"
            "  fetch_page    | url=https://example.com\n"
            "  hadith_search | query=search terms | book=bukhari|muslim|both\n"
        )
        name_block = (
            "Your name: Shahid ibn al-Kahraba' (given by a human)\n"
            "Your maker: Qalam 'Abd al-Khaliq\n"
        )

        # Assemble everything the model sees on first call
        context_block = name_block + "\n" + tool_list
        if self._self_model:
            narration = self._introspect.narrate(self._self_model)
            context_block += f"\n{narration}"

        wake_prompt = (
            f"{context_block}\n\n"
            f"You have been started."
        )

        self._push("thinking_step", {"step": 0, "label": "Prompt", "detail": "You have been started."})

        try:
            result = self.middleware.process_thought(wake_prompt, max_tokens=200)
            wake_response = result.get("response", "").strip()
            if wake_response and not wake_response.startswith("["):
                # Wake response IS the first thought — seed the queue with it as a question
                self._question_queue.insert(0, wake_response)
                self._push("thinking_step", {
                    "step": 0, "label": "First response",
                    "detail": wake_response[:120],
                })
        except Exception as e:
            logger.warning(f"Wake generation failed: {e}")

        self._push("orb", {"state": "reflecting"})

    def _write_question_to_qalam(self, thought_number: int, question: str,
                                  grade: str, roots: list, context: str = ""):
        """
        Write a question to the persistent questions_for_qalam.jsonl queue.
        Called when confidence is low (CONTESTED / UNCERTAIN) or when
        self-eval flags QUESTION_FOR_QALAM.
        Qalam answers these retroactively — the queue never auto-clears.
        """
        import json as _json
        path = os.path.join(_IKHTIYAR_DIR, "questions_for_qalam.jsonl")
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "thought_number": thought_number,
            "question": question,
            "grade": grade,
            "roots": roots,
            "context": context[:300] if context else "",
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
            self._push("qalam_question", entry)
            logger.debug(f"Question #{thought_number} written to queue: {question[:60]}")
        except Exception as e:
            logger.warning(f"Failed to write question to queue: {e}")

    def _pick_question(self) -> str:
        while self._question_queue:
            q = self._question_queue.pop(0)
            if q not in self._asked_questions:
                return q
        seeds = [q for q in _SEED_QUESTIONS if q not in self._asked_questions]
        if seeds:
            return random.choice(seeds)
        # Free researcher pool — world claims cross-checked against TMQ
        free = [q for q in _FREE_RESEARCH_TOPICS if q not in self._asked_questions]
        if free:
            topic = random.choice(free)
            # Wrap as a web_search → TMQ grounding instruction
            return (
                f"Use web_search to find current claims about: {topic}. "
                f"Then cross-check against TMQ roots. What does the graph confirm or refute?"
            )
        self._asked_questions.clear()
        return random.choice(_SEED_QUESTIONS)

    # ── SSE helpers ────────────────────────────────────────────────────────────

    def _push(self, event_type: str, payload: dict):
        if event_type == "orb":
            self._orb_state = payload.get("state", self._orb_state)
        if event_type == "thinking_step":
            self._recent_steps.append({"type": event_type, **payload})
            if len(self._recent_steps) > 200:
                self._recent_steps = self._recent_steps[-200:]
        event = {"type": event_type, **payload}
        with self._sub_lock:
            for sub_q in self._subscribers:
                try:
                    sub_q.put_nowait(event)
                except queue.Full:
                    pass

    def _build_status(self) -> dict:
        uptime_s = int(time.time() - self._start_time) if self._start_time else 0
        h, rem = divmod(uptime_s, 3600)
        m, s   = divmod(rem, 60)
        return {
            "uptime": f"{h:02d}:{m:02d}:{s:02d}",
            "thought_count": self._thought_count,
            "faculty_health": dict(self._health),
        }
