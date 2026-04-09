"""
ikhtiyar/core/vtransistor.py — Virtual Transistor Signal Propagation Engine

Three-tier reasoning architecture derived from Islamic jurisprudential methodology:

  Tier 1 — Static Circuit  (Nass)   : Qur'an's own gate topology, compiled from HVT.
           ROM. Loaded once. Deterministic. Output = HAQQ.

  Tier 2 — Dynamic Circuit (Qiyas)  : projected/inferred connections, assembled per query.
           RAM. Assembled, evaluated, discarded. Output = QIYAS.
           Constraint: must not contradict any Tier 1 evaluation.

  Tier 3 — Convergence     (Ijma')  : multiple independent signal paths through both
           circuits converge on the same output state.
           Not voted on. Computed. Output = high-confidence consensus.
           Divergence = dialectic — presented honestly, both paths traced.

No LLM touches the reasoning path. The LLM renders. It does not reason.

Core primitive: VTransistor — a software switching element.
  gate input > threshold → signal passes.
  gate input ≤ threshold → signal blocked.
That is the complete operation.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Virtual Transistor ────────────────────────────────────────────────────────

class VTransistor:
    """
    Software switching element. The atomic unit of reasoning.
    gate_signal > threshold → ON (signal passes).
    gate_signal ≤ threshold → OFF (signal blocked).
    """
    __slots__ = ('threshold',)

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def switch(self, gate_signal: float) -> bool:
        return gate_signal > self.threshold


# ── Belief Latch (SRAM equivalent) ───────────────────────────────────────────

class BeliefLatch:
    """
    Two cross-coupled inverters. Bistable — holds state by topology.
    The belief is not stored in either transistor. It is stored in their
    mutual reinforcement.

    Write: overwhelm the reinforcement with a signal stronger than the
           current state. Latch flips.
    Read:  return current stable state.
    """
    __slots__ = ('_state', '_strength', '_write_threshold')

    def __init__(self, initial: bool = False, strength: float = 0.5):
        self._state = initial
        self._strength = strength          # how strongly the latch holds
        self._write_threshold = strength   # signal must exceed this to flip

    @property
    def state(self) -> bool:
        return self._state

    @property
    def strength(self) -> float:
        return self._strength

    def read(self) -> bool:
        return self._state

    def write(self, signal: float) -> bool:
        """
        Attempt to write. Returns True if latch flipped.
        Signal must exceed current strength to flip.
        On flip, new strength = signal level (stronger beliefs are harder to flip).
        """
        if signal > self._strength:
            self._state = not self._state
            self._strength = signal
            return True
        return False

    def reinforce(self, amount: float = 0.05):
        """Strengthen current state. Repeated activation hardens the belief."""
        self._strength = min(1.0, self._strength + amount)


# ── Propagation result ───────────────────────────────────────────────────────

@dataclass
class PropagationResult:
    """Output of a circuit evaluation."""
    activated_frames: dict          # frame_index → activation_level (float)
    activated_roots: dict           # root → max_activation_level
    procedure_tags: list            # collected from activated frames
    ayat_refs: list                 # refs of ayat containing activated frames
    tier: str                       # "HAQQ" | "QIYAS" | "WAQF"
    confidence: float               # 0.0–1.0
    path_trace: list = field(default_factory=list)   # ordered frame indices showing signal path
    proof_tree: list = field(default_factory=list)   # ordered dicts: {frame, seg_id, tag, sig_in, sig_out, ayat}
    route: dict = field(default_factory=dict)        # current addressee state: {person, number}


@dataclass
class DialecticResult:
    """Output of Ijma' / dialectic evaluation."""
    convergent: dict                # root → activation where all paths agree
    divergent: list                 # list of (path_a_root, path_b_root, divergence_frame)
    ijma_confidence: float          # 0.0–1.0 — fraction of output that converged
    tier: str                       # "IJMA" | "IKHTILAF"


# ── Static Circuit (Nass — Tier 1) ───────────────────────────────────────────

class StaticCircuit:
    """
    Compiled once from HVT tape. Loaded at startup. Not modified at runtime.

    Wiring:
      1. Root wiring — all frames sharing a root are implicitly linked.
         Activating root 'ktb' lights up every frame in the tape with root='ktb'.
      2. Gate wiring — sequential frames connected by their gate type.
         AND: signal reinforces. OR: signal passes via either. NOT: signal inverts.
      3. Ayat boundaries — propagation is bounded within ayat unless cross-ayat
         root wiring carries it.

    PageRank threshold: high-rank roots have lower activation threshold
    (they are structurally central — easier to activate). Low-rank roots
    require stronger signal.
    """

    def __init__(self, hvt_path: str):
        logger.info(f"Compiling static circuit from {hvt_path} ...")

        with open(hvt_path, encoding="utf-8") as f:
            self._frames: list = json.load(f)

        self._n = len(self._frames)

        # ── Root index: root_bw → [frame indices] ──
        # Index on Buckwalter (programmatic key); Arabic is display format
        self._root_index: dict[str, list[int]] = defaultdict(list)
        for i, frame in enumerate(self._frames):
            root_bw = frame["data"].get("root_bw", "")
            if root_bw:
                self._root_index[root_bw].append(i)

        # ── Ayat boundaries: list of (start_idx, end_idx) ──
        self._ayat_spans: list[tuple[int, int]] = []
        current_start = 0
        for i, frame in enumerate(self._frames):
            if frame.get("ayat_header") is not None and i > 0:
                self._ayat_spans.append((current_start, i - 1))
                current_start = i
        self._ayat_spans.append((current_start, self._n - 1))

        # ── Frame-to-ayat index ──
        self._frame_to_ayat: list[int] = [0] * self._n
        for ayat_idx, (start, end) in enumerate(self._ayat_spans):
            for i in range(start, end + 1):
                self._frame_to_ayat[i] = ayat_idx

        # ── PageRank (computed once) ──
        self._pagerank: dict[str, float] = {}
        self._compute_pagerank()

        # ── Transistor bank: one per root, threshold from PageRank ──
        self._transistors: dict[str, VTransistor] = {}
        for root, rank in self._pagerank.items():
            # High PageRank → low threshold (easier to activate)
            # Rank is 0.0–1.0 normalized. Threshold = 1.0 - rank, clamped to [0.1, 0.9]
            threshold = max(0.1, min(0.9, 1.0 - rank))
            self._transistors[root] = VTransistor(threshold=threshold)

        logger.info(
            f"Static circuit ready — {self._n:,} frames, "
            f"{len(self._root_index):,} roots, "
            f"{len(self._ayat_spans):,} ayat spans"
        )

    def _compute_pagerank(self, iterations: int = 20, damping: float = 0.85):
        """
        PageRank over root co-occurrence within ayat.

        Roots that appear in the same ayat are linked. A root's rank is
        determined by how many high-rank roots it co-occurs with — not by
        raw frequency alone.

        This becomes the gate threshold: structurally central roots
        require less activation energy.
        """
        # Build co-occurrence graph: root_a → {root_b: count}
        co_occurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        all_roots: set[str] = set()

        for start, end in self._ayat_spans:
            ayat_roots = set()
            for i in range(start, end + 1):
                r = self._frames[i]["data"]["root_bw"]
                if r:
                    ayat_roots.add(r)
                    all_roots.add(r)
            roots_list = list(ayat_roots)
            for a in roots_list:
                for b in roots_list:
                    if a != b:
                        co_occurrence[a][b] += 1

        if not all_roots:
            return

        n = len(all_roots)
        rank = {r: 1.0 / n for r in all_roots}

        for _ in range(iterations):
            new_rank = {}
            for root in all_roots:
                incoming = 0.0
                for neighbor, weight in co_occurrence[root].items():
                    out_degree = sum(co_occurrence[neighbor].values())
                    if out_degree > 0:
                        incoming += rank[neighbor] * weight / out_degree
                new_rank[root] = (1.0 - damping) / n + damping * incoming
            rank = new_rank

        # Normalize to [0.0, 1.0]
        max_rank = max(rank.values()) if rank else 1.0
        min_rank = min(rank.values()) if rank else 0.0
        span = max_rank - min_rank if max_rank > min_rank else 1.0
        self._pagerank = {r: (v - min_rank) / span for r, v in rank.items()}

    def get_pagerank(self, root: str) -> float:
        return self._pagerank.get(root, 0.0)

    def evaluate(self, seed_roots: list[str], propagation_steps: int = 3) -> PropagationResult:
        """
        Circuit-driven evaluation.

        The four tautologies that used to live here are severed:
          (1) Seeds no longer flood-fill `signals` at 1.0. Each seed frame is
              injected INTO the word circuit at its own index — the seed
              word's own prefixes and segments get to shape the initial
              signal. A seed in a negated verse comes out negative.
          (2) `signals` is intermediate state only. Collection reads
              `proof_tree`, which is the single source of truth for what
              the circuit did.
          (3) Confidence is derived from per-seed survival through the
              word circuit, not from a lookup on a pre-flood-filled dict.
          (4) Tier is computed from polarity distribution across the
              proof_tree, not hardcoded to "HAQQ".
        """
        path_trace: list[int] = []
        proof_tree: list[dict] = []

        # ── Step 1: Full-ayat circuit walk per seed-containing verse ──
        # For every ayat that contains at least one seed root's frame, walk
        # the entire ayat left-to-right, feeding +1.0 signal through each
        # frame's word circuit in sequence. This lets NEG/AVR/RET at ANY
        # position in the verse — including before the seed word — flip
        # the polarity the seed emerges with. A seed in "لا تقربوا الصلاة
        # وأنتم سكارى" emerges negative because the NEG on "تقربوا" flows
        # through the subsequent words' circuits before reaching "سكارى".
        seed_survivals: dict[str, list[float]] = {r: [] for r in seed_roots}
        seed_ayat_idxs: set[int] = set()
        seeds_in_tape: set[str] = set()
        seeds_missing: set[str] = set()
        for root in seed_roots:
            frames = self._root_index.get(root, [])
            if frames:
                seeds_in_tape.add(root)
                for idx in frames:
                    seed_ayat_idxs.add(self._frame_to_ayat[idx])
            else:
                seeds_missing.add(root)

        # Seeds absent from the tape are a transliteration/lookup failure,
        # not Qur'anic silence. "UNKNOWN" is structurally distinct from
        # "WAQF": WAQF is the muqatta'at boundary (الم, طسم) where the
        # Qur'an itself marks a limit of derivable meaning; UNKNOWN is
        # "the interpreter doesn't recognize this input." Collapsing the
        # two would let typos wear the epistemic weight of Allah's own
        # withholding. They are different kinds of silence.
        if not seeds_in_tape:
            return PropagationResult(
                activated_frames={},
                activated_roots={},
                procedure_tags=[],
                ayat_refs=[],
                tier="UNKNOWN",
                confidence=0.0,
                proof_tree=proof_tree,
            )

        seed_root_set = set(seed_roots)
        for ayat_idx in seed_ayat_idxs:
            ayat_start, ayat_end = self._ayat_spans[ayat_idx]
            sig = 1.0  # fresh positive signal per ayat — no cross-ayat bleed
            for frame_idx in range(ayat_start, ayat_end + 1):
                new_sig = self._apply_word_circuit(frame_idx, sig, proof_tree)
                if new_sig is None:
                    continue
                sig = new_sig
                path_trace.append(frame_idx)

                frame = self._frames[frame_idx]
                loc = frame.get("loc", [0, 0, 0])
                r = frame["data"].get("root_bw", "")

                # If this frame carries a seed root, record it as a SEED
                # entry carrying the verse-contextual polarity.
                if r in seed_root_set:
                    proof_tree.append({
                        "frame":  frame_idx,
                        "seg_id": f"{loc[0]}:{loc[1]}:{loc[2] if len(loc) > 2 else 0}",
                        "tag":    "SEED",
                        "role":   "seed",
                        "lem":    r,
                        "sig_in":  1.0,
                        "sig_out": round(sig, 4),
                        "ayat":    f"{loc[0]}:{loc[1]}",
                    })
                    seed_survivals[r].append(sig)

        # ── Step 2: Collect from path_trace (WALK entries removed from proof_tree) ──
        # path_trace holds every visited frame index in order. activated_frames,
        # activated_roots, ayat_refs, procedure_tags are derived here directly —
        # proof_tree is now a trace of gate events only (SEED + segment primitives).
        activated_frames: dict[int, float] = {}
        activated_roots: dict[str, float] = {}
        root_polarity: dict[str, set[str]] = {}
        ayat_refs: set[str] = set()
        procedure_tags: list[str] = []
        seen_procedures: set[str] = set()

        # Signal per frame: read from SEED entries in proof_tree (most accurate
        # post-circuit value). Frames with no SEED entry get 1.0 (unmodified).
        _frame_sig: dict[int, float] = {}
        for entry in proof_tree:
            if entry.get("tag") == "SEED":
                idx = entry.get("frame")
                if idx is not None:
                    _frame_sig[idx] = float(entry.get("sig_out", 1.0))

        for frame_idx in path_trace:
            frame = self._frames[frame_idx]
            loc = frame.get("loc", [0, 0, 0])
            r = frame["data"].get("root_bw", "")
            ayat = f"{loc[0]}:{loc[1]}"
            sig_out = _frame_sig.get(frame_idx, 1.0)
            mag = abs(sig_out)

            activated_frames[frame_idx] = max(activated_frames.get(frame_idx, 0.0), mag)
            if r:
                activated_roots[r] = max(activated_roots.get(r, 0.0), mag)
                root_polarity.setdefault(r, set()).add("+" if sig_out > 0 else "-")
            ayat_refs.add(ayat)

            for tag in frame["procedure"].get("speech_act", []):
                if tag not in seen_procedures:
                    procedure_tags.append(tag)
                    seen_procedures.add(tag)

        # ── Confidence: per-seed AFFIRMATION through the word circuit ──
        # A seed is "affirmed" in a verse iff its word circuit emerged with
        # positive signal — the verse, as a sentential circuit, affirms it.
        # A negated seed (sig < 0) is still a real event, tracked in
        # root_polarity for tier detection, but it does not count as
        # confidence. Confidence asks: in what fraction of its occurrences
        # is the seed affirmed by the verse containing it?
        per_seed_conf: dict[str, float] = {}
        for root in seed_roots:
            sigs = seed_survivals.get(root, [])
            if not sigs:
                per_seed_conf[root] = 0.0
                continue
            affirmed = sum(1 for s in sigs if s > 0)
            per_seed_conf[root] = affirmed / len(sigs)
        confidence = (
            sum(per_seed_conf.values()) / len(seed_roots)
            if seed_roots else 0.0
        )

        # ── Tier: four structural tests over seed polarity, no thresholds ──
        # Each tier is defined by a yes/no condition on the seeds themselves,
        # not by a confidence threshold I picked. The rules, in order:
        #
        #   WAQF     — no seed produced any non-zero signal anywhere.
        #              (Also the only tier where activated_roots is empty.)
        #   QIYAS    — at least one seed has zero affirmation rate: either it
        #              does not appear in the tape at all, or every one of its
        #              occurrences was fully negated by its verse. The proof
        #              is incomplete at the seed level; inquiry is honest.
        #   IKHTILAF — at least one seed appears in the proof tree with BOTH
        #              polarities. Some occurrence of the seed was affirmed,
        #              some was negated. The verse structure itself is
        #              dialectic. This matches the Qur'an's rhetorical mode
        #              of declaring truth against its opposition.
        #   HAQQ     — every seed was affirmed in every one of its
        #              occurrences, and no seed was negated anywhere. Pure
        #              declarative — the rare clean case.

        # Per-seed polarity partition (derived directly from SEED entries).
        seed_polarities: dict[str, set[str]] = {r: set() for r in seed_roots}
        for e in proof_tree:
            if e.get("tag") != "SEED":
                continue
            root = e.get("lem", "")
            sig = float(e.get("sig_out", 0.0))
            if sig > 0:
                seed_polarities.setdefault(root, set()).add("+")
            elif sig < 0:
                seed_polarities.setdefault(root, set()).add("-")

        has_any_signal = any(pols for pols in seed_polarities.values())
        any_seed_empty = any(not pols for pols in seed_polarities.values())
        any_seed_only_neg = any(
            pols == {"-"} for pols in seed_polarities.values()
        )
        any_seed_dialectic = any(
            pols == {"+", "-"} for pols in seed_polarities.values()
        )

        if not has_any_signal or not activated_roots:
            tier = "WAQF"
        elif any_seed_empty or any_seed_only_neg:
            # Incomplete proof at the seed level → honest inquiry.
            tier = "QIYAS"
        elif any_seed_dialectic:
            # Qur'an's own rhetoric for this seed is dialectic → present both.
            tier = "IKHTILAF"
        else:
            # Every seed affirmed everywhere, no negation.
            tier = "HAQQ"

        return PropagationResult(
            activated_frames=activated_frames,
            activated_roots=activated_roots,
            procedure_tags=procedure_tags,
            ayat_refs=sorted(ayat_refs),
            tier=tier,
            confidence=round(confidence, 4),
            path_trace=path_trace,
            proof_tree=proof_tree,
        )

    # ── Per-primitive signal handlers ──────────────────────────────────────
    #
    # Each handler: (signal_in, segment_dict) → signal_out | None
    # None aborts propagation to this frame entirely.
    #
    # Two operations only — grounded in Arabic grammar, not in preference:
    #
    #   _h_neg  — sign inversion. NEG/AVR negate; RET retracts.
    #             Direction is determined by linguistics. Magnitude is 1.0
    #             because no corpus-derived magnitude has been computed yet.
    #             When IDF weights are derived from the QAC morphology file
    #             (128,219 segments), they will be inserted here with full
    #             provenance. Until then: polarity only.
    #
    #   _h_pass — identity. All other tags are structurally recorded in the
    #             proof tree (via is_gate) but do not modify signal magnitude.
    #             Their presence in the trace is real; their gain is deferred.
    #
    # Tags with deferred magnitude (corpus derivation pending):
    #   CERT, EMPH, EXH   — certainty / emphasis    → amplify (direction known)
    #   REM, CONJ, SUB    — structural connectives  → attenuate (direction known)
    #   CAUS, PRP         — causal / purposive       → amplify (direction known)
    #   COND, ANS, RSLT   — conditional              → gate (direction known)
    #   RES, EXP, EXL     — restriction              → attenuate (direction known)
    #   AMD               — amendment                → attenuate (direction known)
    #   AMR, NAHY         — speech act (frame level) → amplify/suppress (direction known)
    #
    # Directions above are from Arabic grammar — not invented.
    # Magnitudes require: log(N/count) from QAC, normalized to a gain range
    # the circuit can sustain across a full ayat walk. That computation has
    # not been done. # REQUIRES_CORPUS_DERIVATION

    @staticmethod
    def _h_neg(sig, seg):
        """Polarity inversion. NEG / AVR / RET — negate."""
        return -sig

    @staticmethod
    def _h_pass(sig, seg):
        """Identity. No magnitude change."""
        return sig

    # ── Dispatch table — tag → method name ───────────────────────────────
    # Polarity tags invert. Everything else is identity until corpus-derived
    # magnitudes are computed and reviewed.
    _HANDLER_NAMES: dict = {
        "NEG": "_h_neg", "AVR": "_h_neg", "RET": "_h_neg",
    }

    @classmethod
    def _handler_for(cls, tag: str):
        name = cls._HANDLER_NAMES.get(tag, "_h_pass")
        return getattr(cls, name)

    def _apply_speech_act(self, frame: dict, sig: float) -> float:
        """Speech-act frame-level gate. Magnitude deferred — identity until
        corpus-derived AMR/NAHY weights are computed. # REQUIRES_CORPUS_DERIVATION"""
        return sig

    def _apply_word_circuit(
        self,
        target_idx: int,
        incoming_signal: float,
        proof_tree: list,
    ) -> Optional[float]:
        """
        Walk the target frame's procedure.segments[] in order, applying each
        segment's primitive handler to the signal in sequence. This is the
        word-level micro-circuit: a single punch can contain multiple gates.

        Captures every non-trivial step into proof_tree for downstream
        proof-to-assertions compilation (Step 3 of the plan).
        """
        frame = self._frames[target_idx]
        proc = frame.get("procedure", {})
        segs = proc.get("segments", [])
        if not segs:
            return incoming_signal  # no segments = no operation, pass through

        # Derive ayat ref once for proof tree entries
        loc = frame.get("loc", [0, 0, 0])
        ayat_ref = f"{loc[0]}:{loc[1]}"

        sig = incoming_signal
        for seg in segs:
            tag = seg.get("tag", "")
            handler = self._handler_for(tag)
            sig_in = sig
            sig = handler(sig, seg)
            if sig is None:
                return None
            # Record step if this segment is a declared gate or signal changed.
            if seg.get("is_gate") or abs(sig - sig_in) > 0.05:
                proof_tree.append({
                    "frame":  target_idx,
                    "seg_id": seg.get("id", ""),
                    "tag":    tag,
                    "role":   seg.get("role", ""),
                    "lem":    seg.get("lem", ""),
                    "sig_in":  round(sig_in, 4),
                    "sig_out": round(sig,    4),
                    "ayat":   ayat_ref,
                })

        # Frame-level speech-act amplifier (AMR, NAHY, TABSHIR, INDHAR).
        # Applied after all segment gates so the command layer wraps the word.
        sig = self._apply_speech_act(frame, sig)
        return sig


# ── Dynamic Circuit (Qiyas — Tier 2) ─────────────────────────────────────────

class DynamicCircuit:
    """
    Assembled per query from projected/inferred connections.
    The nodes are real HVT frames. The wiring is hypothesized.

    Constraint: output must not contradict any Tier 1 evaluation.
    If a root is suppressed (NOT gate) in the static circuit, the dynamic
    circuit cannot activate it.
    """

    def __init__(self, static: StaticCircuit):
        self._static = static

    def evaluate(
        self,
        seed_roots: list[str],
        projected_connections: list[tuple[str, str, float]],
        static_result: PropagationResult,
    ) -> PropagationResult:
        """
        Evaluate dynamic circuit using projected root connections.

        Args:
            seed_roots: original query roots
            projected_connections: list of (source_root, target_root, confidence)
                from GraphProjector (clock rotation, ibn_jinni matrix)
            static_result: Tier 1 result — used for contradiction checking

        Returns:
            PropagationResult with tier="QIYAS"
        """
        # Build suppressed set from static circuit NOT gates
        suppressed_roots: set[str] = set()
        for idx, sig in static_result.activated_frames.items():
            if sig < 0:  # NOT-suppressed
                r = self._static._frames[idx]["data"]["root_bw"]
                if r:
                    suppressed_roots.add(r)

        # Build dynamic wiring from projections
        dynamic_roots: list[str] = list(seed_roots)
        for src, tgt, conf in projected_connections:
            # Constraint: cannot activate suppressed roots
            if tgt in suppressed_roots:
                continue
            if conf > 0.2 and tgt not in dynamic_roots:
                dynamic_roots.append(tgt)

        if not dynamic_roots or dynamic_roots == list(seed_roots):
            return PropagationResult(
                activated_frames={},
                activated_roots={},
                procedure_tags=[],
                ayat_refs=[],
                tier="WAQF",
                confidence=0.0,
            )

        # Evaluate through static circuit with expanded root set
        result = self._static.evaluate(dynamic_roots, propagation_steps=2)

        # Re-tag as QIYAS and reduce confidence (projected, not direct)
        max_proj_confidence = max(c for _, _, c in projected_connections) if projected_connections else 0.0
        qiyas_confidence = result.confidence * max_proj_confidence * 0.8

        return PropagationResult(
            activated_frames=result.activated_frames,
            activated_roots=result.activated_roots,
            procedure_tags=result.procedure_tags,
            ayat_refs=result.ayat_refs,
            tier="QIYAS",
            confidence=round(qiyas_confidence, 4),
            path_trace=result.path_trace,
        )


# ── Circuit Evaluator (Orchestrator) ─────────────────────────────────────────

class CircuitEvaluator:
    """
    Top-level orchestrator. Runs Tier 1, Tier 2, Tier 3 in sequence.

    Tier 1 (Static/Nass):  always runs first. If HAQQ output is sufficient,
                           return without Tier 2.
    Tier 2 (Dynamic/Qiyas): runs when Tier 1 confidence is below threshold
                            or seed roots don't resolve in static circuit.
    Tier 3 (Ijma'/Ikhtilaf): runs when multiple independent evaluations are
                             available. Computes convergence.
    """

    def __init__(self, hvt_path: str):
        self._static = StaticCircuit(hvt_path)
        self._dynamic = DynamicCircuit(self._static)
        self._beliefs: dict[str, BeliefLatch] = {}  # root → latch
        logger.info("CircuitEvaluator ready")

    @property
    def static(self) -> StaticCircuit:
        return self._static

    def evaluate(
        self,
        seed_roots: list[str],
        projected_connections: list[tuple[str, str, float]] = None,
        haqq_threshold: float = 0.5,
    ) -> PropagationResult:
        """
        Full three-tier evaluation.

        Args:
            seed_roots: Buckwalter roots from Bilal
            projected_connections: from GraphProjector (optional)
            haqq_threshold: minimum confidence for Tier 1 to be sufficient

        Returns:
            PropagationResult (best tier result)
        """
        # ── Tier 1: Static (Nass) ──
        static_result = self._static.evaluate(seed_roots)

        if static_result.confidence >= haqq_threshold:
            self._update_beliefs(static_result)
            return static_result

        # ── Tier 2: Dynamic (Qiyas) ──
        if projected_connections:
            dynamic_result = self._dynamic.evaluate(
                seed_roots, projected_connections, static_result
            )

            if dynamic_result.confidence > static_result.confidence:
                self._update_beliefs(dynamic_result)
                return dynamic_result

        # ── Tier 1 result is best available (even if below threshold) ──
        if static_result.activated_frames:
            self._update_beliefs(static_result)
            return static_result

        # ── WAQF ──
        return PropagationResult(
            activated_frames={},
            activated_roots={},
            procedure_tags=[],
            ayat_refs=[],
            tier="WAQF",
            confidence=0.0,
        )

    def evaluate_ijma(
        self,
        root_sets: list[list[str]],
        projected_connections: list[tuple[str, str, float]] = None,
    ) -> DialecticResult:
        """
        Tier 3: Evaluate multiple independent signal paths and compute convergence.

        Each root_set is an independent entry point (different decomposition of
        the same query, or different facets). All are evaluated independently.
        Convergence = Ijma'. Divergence = dialectic.

        Args:
            root_sets: list of independent root lists (each from Bilal)
            projected_connections: shared projections (optional)

        Returns:
            DialecticResult with convergence analysis
        """
        results: list[PropagationResult] = []
        for roots in root_sets:
            r = self.evaluate(roots, projected_connections)
            results.append(r)

        if len(results) < 2:
            # Need at least 2 paths for dialectic
            single = results[0] if results else None
            return DialecticResult(
                convergent=single.activated_roots if single else {},
                divergent=[],
                ijma_confidence=single.confidence if single else 0.0,
                tier="HAQQ" if single else "WAQF",
            )

        # Compute intersection (Ijma') and difference (Ikhtilaf)
        all_root_sets = [set(r.activated_roots.keys()) for r in results]
        convergent_roots = all_root_sets[0]
        for rs in all_root_sets[1:]:
            convergent_roots = convergent_roots & rs

        # Convergent: roots ALL paths agree on, with min activation across paths
        convergent = {}
        for root in convergent_roots:
            min_activation = min(r.activated_roots.get(root, 0.0) for r in results)
            convergent[root] = min_activation

        # Divergent: roots where paths disagree
        all_activated = set()
        for rs in all_root_sets:
            all_activated |= rs
        divergent_roots = all_activated - convergent_roots

        divergent = []
        for root in divergent_roots:
            present_in = [i for i, rs in enumerate(all_root_sets) if root in rs]
            absent_from = [i for i, rs in enumerate(all_root_sets) if root not in rs]
            divergent.append({
                "root": root,
                "present_in_paths": present_in,
                "absent_from_paths": absent_from,
                "max_activation": max(
                    results[i].activated_roots.get(root, 0.0) for i in present_in
                ),
            })

        # Ijma' confidence: fraction of total activated roots that converged
        total_unique = len(all_activated) if all_activated else 1
        ijma_confidence = len(convergent_roots) / total_unique

        tier = "IJMA" if ijma_confidence >= 0.5 else "IKHTILAF"

        return DialecticResult(
            convergent=convergent,
            divergent=divergent,
            ijma_confidence=round(ijma_confidence, 4),
            tier=tier,
        )

    def _update_beliefs(self, result: PropagationResult):
        """
        Update belief latches based on circuit evaluation result.
        High-activation roots reinforce existing beliefs.
        New roots create new latches.
        """
        for root, activation in result.activated_roots.items():
            if root not in self._beliefs:
                self._beliefs[root] = BeliefLatch(
                    initial=True,
                    strength=activation * 0.5,
                )
            else:
                latch = self._beliefs[root]
                if latch.state:
                    latch.reinforce(activation * 0.05)
                else:
                    latch.write(activation)

    def get_belief(self, root: str) -> Optional[BeliefLatch]:
        return self._beliefs.get(root)

    def get_all_beliefs(self) -> dict[str, dict]:
        return {
            root: {"state": latch.state, "strength": round(latch.strength, 4)}
            for root, latch in self._beliefs.items()
        }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    here = __file__
    from pathlib import Path
    proj = Path(here).parent.parent.parent  # project root
    hvt_path = proj / "ikhtiyar" / "TMQ_hvt.json"

    if not hvt_path.exists():
        print(f"HVT tape not found at {hvt_path}")
        print("Run hvt_compiler.py first.")
        sys.exit(1)

    evaluator = CircuitEvaluator(str(hvt_path))

    # Test: evaluate roots for "What is justice?"
    test_roots = ["Edl", "Hkm", "qsT"]
    print(f"\n── Tier 1 (Nass) evaluation for roots: {test_roots} ──")
    result = evaluator.evaluate(test_roots)
    print(f"Tier:       {result.tier}")
    print(f"Confidence: {result.confidence}")
    print(f"Activated roots ({len(result.activated_roots)}): "
          f"{dict(list(result.activated_roots.items())[:10])}")
    print(f"Procedure tags: {result.procedure_tags[:5]}")
    print(f"Ayat refs ({len(result.ayat_refs)}): {result.ayat_refs[:5]}")

    # Test: Ijma' with two independent decompositions
    path_a = ["Edl", "Hkm"]
    path_b = ["qsT", "myz"]
    print(f"\n── Tier 3 (Ijma') evaluation ──")
    print(f"Path A: {path_a}")
    print(f"Path B: {path_b}")
    dialectic = evaluator.evaluate_ijma([path_a, path_b])
    print(f"Tier:       {dialectic.tier}")
    print(f"Ijma' confidence: {dialectic.ijma_confidence}")
    print(f"Convergent roots ({len(dialectic.convergent)}): "
          f"{dict(list(dialectic.convergent.items())[:8])}")
    print(f"Divergent entries: {len(dialectic.divergent)}")
    if dialectic.divergent:
        print(f"First divergence: {dialectic.divergent[0]}")
