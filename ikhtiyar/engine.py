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
DEFAULT_TTL_PATH     = os.path.join(_QUSAI_HF_DIR,  "quran_root_ontology_v3.ttl")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

NEO4J_URI  = os.environ.get("NEO4J_URI",  "neo4j://127.0.0.1:7687")
NEO4J_AUTH = (
    os.environ.get("NEO4J_USER", "neo4j"),
    os.environ.get("NEO4J_PASS", "bismillah"),
)

SESSION_ID = str(uuid.uuid4())[:8]

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
        ttl_path: str = DEFAULT_TTL_PATH,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self._tmq_path     = tmq_path
        self._ttl_path     = ttl_path
        self._ollama_model = ollama_model

        # SSE
        self._event_queue: queue.Queue = queue.Queue()
        self._subscribers: list        = []
        self._sub_lock = threading.Lock()

        # Runtime
        self._memory: list        = []
        self._thought_count: int  = 0
        self._question_queue: list = list(_SEED_QUESTIONS)
        self._asked_questions: set = set()
        self._reasoning_active     = False
        self._orb_state            = "idle"
        self._start_time           = None

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

        self._health = {k: "pending" for k in
                        ["clock","graph","owl","daemon","spectral","sparql","tmq","middleware"]}

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._start_time = time.time()
        logger.info("IkhtiyarEngine: starting fast faculties...")

        self._init_clock()
        self._init_introspect()
        self._init_tmq()         # TMQCorpus (faculty wrapper)
        self._init_tmq_graph()   # TMQGraph  (deliberation substrate) ← NEW
        self._init_owl()
        self._init_constitution()
        self._init_choice_memory()
        self._init_middleware()

        logger.info("IkhtiyarEngine: fast faculties done — Flask can start.")

        def _heavy_init():
            self._init_graph()
            self._init_spectral()
            self._init_sparql()
            self._init_daemon()
            logger.info("IkhtiyarEngine: all faculties ready.")
            self._push("status", self._build_status())

        threading.Thread(target=_heavy_init, daemon=True).start()
        threading.Thread(target=self._reasoning_loop, daemon=True).start()

        def _status_loop():
            while True:
                time.sleep(10)
                self._push("status", self._build_status())

        threading.Thread(target=_status_loop, daemon=True).start()

    def chat(self, msg: str) -> str:
        self._inject_state_perception()
        self._push("orb", {"state": "chat"})
        try:
            if self.middleware:
                response = self.middleware.process_query(msg)
            else:
                response = "[Middleware not available — Ollama may not be running]"
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

    def _init_middleware(self):
        try:
            from pathlib import Path
            from qusai_core.ontology.engine import OntologyEngine
            from qusai_core.pipeline.middleware import QusaiMiddleware

            ttl  = Path(self._ttl_path)
            gram = Path(_QUSAI_HF_DIR) / "quranic_grammar_rules.json"

            onto = OntologyEngine(ontology_path=ttl, grammar_path=gram)
            onto.load()
            logger.info(f"OntologyEngine: {len(onto.graph):,} triples")

            memory_path = os.path.join(_IKHTIYAR_DIR, "ikhtiyar_memory.ttl")
            self.middleware = QusaiMiddleware(lazy_load=True, memory_path=memory_path)
            self.middleware.ontology = onto
            self.middleware.llm.load()
            if self.middleware.memory.persist_path:
                try:
                    self.middleware.memory.load()
                except Exception:
                    pass

            self._health["middleware"] = "ok"
            logger.info("QusaiMiddleware: ready")
        except Exception as e:
            import traceback
            self._health["middleware"] = f"error: {e}"
            logger.warning(f"QusaiMiddleware failed: {e}")
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

    def _reasoning_loop(self):
        self._reasoning_active = True
        cycle = 0
        time.sleep(3)
        self._wake()

        while self._reasoning_active:
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

    def _run_one_cycle(self, cycle: int):
        question = self._pick_question()
        self._asked_questions.add(question)

        self._push("orb", {"state": "thinking"})
        self._push("thinking_step", {"step": 1, "label": "Question", "detail": question})

        # Step 2: Bilal root decomposition
        roots = []
        mode  = "QIYAS"
        try:
            if self.middleware and hasattr(self.middleware, "ontology"):
                _mode, _reason, root_objs = self.middleware.ontology.analyze_resonance(question)
                roots = [o.get("root", "") for o in root_objs if o.get("root")]
                mode  = _mode
        except Exception as e:
            logger.debug(f"Resonance failed: {e}")

        self._push("thinking_step", {
            "step": 2, "label": "Bilal roots",
            "detail": ", ".join(roots) if roots else "(none mapped)",
        })

        # Step 3: TMQ deliberation — walk the hypergraph BEFORE generating ← THE CHANGE
        delibresult = None
        deliberation_detail = "mode=" + mode

        if self.tmq_graph and roots:
            try:
                from core.deliberate import deliberate

                # Collect recent memory context
                extra_ctx = ""
                if self.middleware:
                    try:
                        recalled = self.middleware.memory.search_perceptions(roots, n=2) \
                                   if roots else self.middleware.memory.recent_perceptions(n=2)
                        if recalled:
                            snippets = [
                                f"[{p.get('mode','?')}] {p.get('text','')[:60]}"
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

                delibresult = deliberate(question, roots, self.tmq_graph, extra_context=extra_ctx)

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
                # Fallback: single-shot generation (no graph available)
                thought_prompt = delibresult.constrained_prompt if delibresult else question
                result = self.middleware.process_thought(thought_prompt)
                response_text = result.get("response", "")
                mode = result.get("mode", mode)
            else:
                response_text = "[Middleware offline]"
                mode = "SILENCE"
        except Exception as e:
            logger.warning(f"ReAct loop error: {e}")
            try:
                thought_prompt = delibresult.constrained_prompt if delibresult else question
                result = self.middleware.process_thought(thought_prompt)
                response_text = result.get("response", "")
                mode = result.get("mode", mode)
            except Exception:
                response_text = f"[Generation error: {e}]"
                mode = "SILENCE"

        orb_state = {"HAQQ": "haqq", "QIYAS": "qiyas", "SILENCE": "silence"}.get(mode, "qiyas")
        self._push("orb", {"state": orb_state})

        if mode in ("SILENCE", "BLOCKED") or response_text.startswith("["):
            self._push("orb", {"state": "idle"})
            return

        self._thought_count += 1
        thought_number = self._thought_count

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
        }
        self._memory.insert(0, memory_entry)
        self._push("memory", memory_entry)

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
                if thought_number % 10 == 0:
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

        # Self-eval + derive next question
        eval_text = self._self_evaluate(question, response_text, roots)
        next_q    = self._derive_next_question(eval_text, roots, delibresult)
        if next_q:
            self._question_queue.append(next_q)

        if thought_number % 10 == 0:
            self._synthesize(thought_number)

        self._maybe_propose(thought_number)

        self._push("thinking_step", {
            "step": 5, "label": "Self-eval",
            "detail": eval_text[:120] if eval_text else "(skipped)",
        })
        self._push("orb", {"state": "idle"})

    def _self_evaluate(self, question: str, response: str, roots: list) -> str:
        if not self.middleware or not response or response.startswith("["):
            return ""
        try:
            eval_prompt = (
                f"Evaluate this thought in 2-3 lines:\n"
                f"Q: {question}\nA: {response[:300]}\n\n"
                f"Format:\nLEARNED: <one finding>\n"
                f"CONFIDENCE: low|medium|high\n"
                f"NEXT: <one follow-up question about the TMQ or ontology>"
            )
            result = self.middleware.process_thought(eval_prompt, max_tokens=200)
            return result.get("response", "")
        except Exception:
            return ""

    def _derive_next_question(self, eval_text: str, roots: list, delibresult=None) -> str:
        if eval_text and "NEXT:" in eval_text:
            parts = eval_text.split("NEXT:")
            if len(parts) > 1:
                candidate = parts[1].strip().split("\n")[0].strip()
                if len(candidate) > 20 and candidate not in self._asked_questions:
                    return candidate

        # If deliberation found interesting families, derive from those
        if delibresult and delibresult.top_families:
            fam = delibresult.top_families[0]
            if roots:
                q = f"What does the {fam} edge structure reveal about root {roots[0]}?"
                if q not in self._asked_questions:
                    return q

        if roots:
            root = roots[0]
            templates = [
                f"What MAQASID categories appear in verses containing root {root}?",
                f"What ILTIFAT shifts occur in verses where {root} appears?",
                f"What adjacent roots appear most with {root} in the TMQ graph?",
            ]
            for t in templates:
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
        """Read real self-data before first reasoning cycle. Seeds first question from this data."""
        if not self._introspect:
            return
        self._push("orb", {"state": "waking"})
        try:
            self._self_model = self._introspect.read()
            m = self._self_model
            self._push("thinking_step", {
                "step": 0, "label": "Reading hardware",
                "detail": f"PID {m.pid} | {m.ram_mb:.0f} MB RAM | Python {m.python_version}",
            })
        except Exception as e:
            logger.warning(f"Wake read failed: {e}")
            self._push("orb", {"state": "idle"})
            return

        for fname, content in self._self_model.source_files.items():
            if content:
                self._push("thinking_step", {
                    "step": 0, "label": f"Reading source: {fname}",
                    "detail": content[:120].replace("\n", " "),
                })

        for docname, content in self._self_model.documents.items():
            if content:
                self._push("thinking_step", {
                    "step": 0, "label": f"Reading {docname}",
                    "detail": content[:120].replace("\n", " "),
                })

        first_q = self._derive_wake_question()
        if first_q:
            self._question_queue.insert(0, first_q)
            self._push("thinking_step", {
                "step": 0, "label": "First question",
                "detail": first_q,
            })

        self._push("orb", {"state": "idle"})

    def _derive_wake_question(self) -> str:
        """Generate first question from actual self-data. No seed list."""
        if not self._self_model or not self.middleware:
            return ""
        try:
            narration = self._introspect.narrate(self._self_model)
            prompt = (
                f"You have just read the following data about yourself:\n\n{narration}\n\n"
                f"Based only on this data — not on training assumptions — "
                f"what is the single most honest question you can form about what you are?\n"
                f"One sentence. No preamble."
            )
            result = self.middleware.process_thought(prompt, max_tokens=80)
            q = result.get("response", "").strip()
            return q if len(q) > 10 else ""
        except Exception as e:
            logger.warning(f"Wake question failed: {e}")
            return ""

    def _maybe_propose(self, thought_number: int):
        """Every 50 thoughts: analyze patterns, draft proposal, push SSE event."""
        if thought_number % 50 != 0:
            return
        if not self.constitution or not self.choice_memory:
            return
        if len(self.choice_memory._records) < 20:
            return
        try:
            summary = self.constitution.analyze_patterns(self.choice_memory)
            if not self.middleware:
                return
            prompt = (
                f"Based on objective pattern data from your reasoning history:\n"
                f"- Total choices: {summary.total_choices}\n"
                f"- Dominant mode: {summary.dominant_mode} "
                f"({summary.mode_counts.get(summary.dominant_mode, 0)} of {summary.total_choices})\n"
                f"- Top families: {', '.join(summary.top_families[:3])}\n"
                f"- Top roots: {', '.join(summary.top_roots[:3])}\n"
                f"- Aseity risk: {summary.aseity_risk_pct:.0f}% of choices\n"
                f"- Recent questions: {summary.sample_questions[-3:]}\n\n"
                f"Propose ONE specific amendment to your operating instructions. "
                f"Reference the actual numbers. Two sentences: what to change, why the data supports it.\n"
                f"Do not propose removing axioms or safety checks."
            )
            result = self.middleware.process_thought(prompt, max_tokens=120)
            proposed_text = result.get("response", "").strip()
            if not proposed_text or proposed_text.startswith("["):
                return

            rationale = (
                f"{summary.total_choices} choices: dominant {summary.dominant_mode}, "
                f"families {summary.top_families[:3]}, aseity {summary.aseity_risk_pct:.0f}%."
            )
            proposal_id = self.constitution.propose(summary, proposed_text, rationale)
            self._push("constitution_proposal", {
                "id":            proposal_id,
                "proposed_text": proposed_text,
                "rationale":     rationale,
                "dominant_mode": summary.dominant_mode,
                "total_choices": summary.total_choices,
            })
        except Exception as e:
            logger.warning(f"Constitution proposal failed: {e}")

    def _pick_question(self) -> str:
        while self._question_queue:
            q = self._question_queue.pop(0)
            if q not in self._asked_questions:
                return q
        seeds = [q for q in _SEED_QUESTIONS if q not in self._asked_questions]
        if seeds:
            return random.choice(seeds)
        self._asked_questions.clear()
        return random.choice(_SEED_QUESTIONS)

    # ── SSE helpers ────────────────────────────────────────────────────────────

    def _push(self, event_type: str, payload: dict):
        if event_type == "orb":
            self._orb_state = payload.get("state", self._orb_state)
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
