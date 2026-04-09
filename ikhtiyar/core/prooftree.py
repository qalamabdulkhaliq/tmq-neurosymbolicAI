"""
ikhtiyar/core/prooftree.py — Proof tree / assertions bridge (Layer 3)

Converts a CircuitEvaluator PropagationResult.proof_tree (ordered per-segment
signal transformations) into a set of typed Assertions the grammar layer can
require verbatim. This is the Lean-proof equivalent: deterministic, replayable,
machine-checkable.

Walk semantics (one pass over proof_tree, stable order preserved):
  CERT / EMPH           → ASSERT    (latch-write: belief discharged)
  NEG / AVR             → NEG       (suppressed root within ayat)
  COND                  → COND      (paired with following ANS/RSLT as consequent)
  RES / EXP / EXL       → EXCEPT    (excluded scope)
  REM                   → ordering flag on next emitted assertion
  CONJ                  → parallel flag on next emitted assertion
  VOC                   → ROUTE update (addressee register)
  INTG                  → QIYAS elevator (question forced)
  CAUS / PRP            → teleology flag on next ASSERT
  RET                   → flip most recent ASSERT for same root
  CITE                  → always emitted per unique ayat visited

The output ProofTree is ordered by visit order (the tape's order = Qur'an order),
never by score. Duplicates are collapsed per (kind, root, ayat, text). Each
assertion carries literal mushaf text — the grammar layer will require it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Typed assertions ─────────────────────────────────────────────────────────

@dataclass
class Assertion:
    kind: str                 # ASSERT | NEG | COND | EXCEPT | ROUTE | WEIGHT | CITE
    ayat: str                 # "s:v"
    text: str                 # literal Arabic fragment from mushaf
    root: Optional[str] = None
    force: str = "NEUTRAL"    # AMR | NAHY | TABSHIR | INDHAR | ISTIFHAM | NEUTRAL
    weight: float = 1.0       # signal strength at emission
    flags: list = field(default_factory=list)   # ordering/parallel/teleology markers
    seg_id: str = ""          # originating segment id for replay

    def key(self) -> tuple:
        return (self.kind, self.root or "", self.ayat, self.text)


@dataclass
class ProofTree:
    query: str
    seed_roots: list
    tier: str                         # HAQQ | QIYAS | IKHTILAF | WAQF
    confidence: float
    assertions: list                  # list[Assertion]
    route: dict = field(default_factory=dict)
    ayat_refs: list = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable single-line summary for prompt rendering."""
        lines = [f"Query: {self.query}",
                 f"Tier: {self.tier}  Confidence: {self.confidence:.2f}",
                 f"Seed: {', '.join(self.seed_roots)}",
                 f"Route: {self.route or '(unset)'}",
                 f"Assertions ({len(self.assertions)}):"]
        for i, a in enumerate(self.assertions, 1):
            lines.append(f"  {i}. [{a.kind}] {a.ayat} — {a.text}")
        return "\n".join(lines)

    def required_literals(self) -> list:
        """Literal terminals the grammar must require."""
        seen, out = set(), []
        for a in self.assertions:
            if a.text and a.text not in seen:
                seen.add(a.text)
                out.append(a.text)
        return out


# ── Signal-force mapping ─────────────────────────────────────────────────────

def _force_from_tag(tag: str, sig: float) -> str:
    """
    Map a segment tag + current signal polarity to an illocutionary force.
    Tag lookup is exact (QAC provenance). Polarity is sign-only — no
    magnitude thresholds, because magnitude was architect's taste masquerading
    as calibration.
    """
    if tag in ("AMR", "IMPV"):
        return "AMR"
    if tag in ("PROH", "NAHY"):
        return "NAHY"
    if tag == "INTG":
        return "ISTIFHAM"
    if sig > 0:
        return "TABSHIR"
    if sig < 0:
        return "INDHAR"
    return "NEUTRAL"


# ── Walk / build ─────────────────────────────────────────────────────────────

def proof_to_assertions(
    result,              # PropagationResult
    mushaf,              # MushafReader (or None → text stays empty)
    query: str = "",
    seed_roots: Optional[list] = None,
) -> ProofTree:
    """
    Walk PropagationResult.proof_tree in order and emit typed assertions.

    The walk is order-preserving; assertion list reflects recitation order.
    """
    steps = list(result.proof_tree or [])
    seeds = list(seed_roots or [])

    assertions: list[Assertion] = []
    seen_keys: set = set()
    cited_ayat: set = set()
    pending_flags: list = []
    route: dict = dict(getattr(result, "route", {}) or {})

    def _mushaf_text(ayat: str) -> str:
        if not mushaf or ":" not in ayat:
            return ""
        try:
            s, v = ayat.split(":", 1)
            return mushaf.get_ayah(int(s), int(v)) or ""
        except Exception:
            return ""

    def _emit(a: Assertion):
        if pending_flags:
            a.flags.extend(pending_flags)
            pending_flags.clear()
        k = a.key()
        if k in seen_keys:
            return
        seen_keys.add(k)
        assertions.append(a)

    def _frame_root(frame_idx: int) -> Optional[str]:
        try:
            frame = result  # not used — handler below resolves via step
        except Exception:
            pass
        return None

    # Pre-scan: collect ayat that contain a SEED-tagged step.
    # CITE fires only on these — not every ayat the walk touches.
    seed_ayat: set = {s["ayat"] for s in steps if s.get("tag") == "SEED" and s.get("ayat")}

    # Fast root lookup per step via captured frame in proof_tree entry (if any).
    for step in steps:
        tag = step.get("tag", "")
        ayat = step.get("ayat", "")
        seg_id = step.get("seg_id", "")
        sig_in = float(step.get("sig_in", 0.0))
        sig_out = float(step.get("sig_out", 0.0))
        root = step.get("lem") or step.get("root") or None
        force = _force_from_tag(tag, sig_out)
        text = _mushaf_text(ayat)

        # CITE — one per unique *seed* ayat only (not every touched frame)
        if ayat and ayat in seed_ayat and ayat not in cited_ayat:
            cited_ayat.add(ayat)
            _emit(Assertion(
                kind="CITE", ayat=ayat, text=text,
                force=force, weight=abs(sig_out), seg_id=seg_id,
            ))

        # ASSERT — CERT and EMPH only; ACC fires on every accusative noun (bloat)
        if tag in ("CERT", "EMPH"):
            _emit(Assertion(
                kind="ASSERT", ayat=ayat, text=text, root=root,
                force=force, weight=abs(sig_out), seg_id=seg_id,
            ))

        elif tag in ("NEG", "AVR"):
            _emit(Assertion(
                kind="NEG", ayat=ayat, text=text, root=root,
                force="NAHY" if force == "NEUTRAL" else force,
                weight=abs(sig_out), seg_id=seg_id,
            ))

        elif tag == "COND":
            _emit(Assertion(
                kind="COND", ayat=ayat, text=text, root=root,
                force=force, weight=abs(sig_out), seg_id=seg_id,
                flags=["antecedent"],
            ))

        elif tag in ("ANS", "RSLT"):
            # Consequent of a prior COND: flag and emit as ASSERT
            _emit(Assertion(
                kind="ASSERT", ayat=ayat, text=text, root=root,
                force=force, weight=abs(sig_out), seg_id=seg_id,
                flags=["consequent"],
            ))

        elif tag in ("RES", "EXP", "EXL"):
            _emit(Assertion(
                kind="EXCEPT", ayat=ayat, text=text, root=root,
                force=force, weight=abs(sig_out), seg_id=seg_id,
            ))

        elif tag == "REM":
            pending_flags.append("sequential")

        elif tag == "CONJ":
            pending_flags.append("parallel")

        elif tag in ("CAUS", "PRP"):
            pending_flags.append("teleology")

        elif tag == "VOC":
            addr = step.get("addr") or step.get("lem") or ""
            if addr:
                route["addressee"] = addr
            _emit(Assertion(
                kind="ROUTE", ayat=ayat, text=text,
                force=force, weight=abs(sig_out), seg_id=seg_id,
            ))

        elif tag == "INTG":
            # Question — elevates to socratic rendering
            _emit(Assertion(
                kind="ASSERT", ayat=ayat, text=text, root=root,
                force="ISTIFHAM", weight=abs(sig_out), seg_id=seg_id,
                flags=["question"],
            ))

        elif tag == "RET":
            # Flip most recent ASSERT for same ayat (retraction)
            for a in reversed(assertions):
                if a.kind == "ASSERT" and a.ayat == ayat:
                    a.kind = "NEG"
                    a.flags.append("retracted")
                    break

    # Sort deterministically: ayat order (surah, verse), then original index
    def _ayat_key(a: Assertion) -> tuple:
        if ":" in a.ayat:
            try:
                s, v = a.ayat.split(":", 1)
                return (int(s), int(v))
            except Exception:
                return (9999, 9999)
        return (9999, 9999)

    # Preserve emission order but group CITE first per ayat
    assertions.sort(key=lambda a: (_ayat_key(a), 0 if a.kind == "CITE" else 1))

    return ProofTree(
        query=query,
        seed_roots=seeds,
        tier=getattr(result, "tier", "WAQF"),
        confidence=float(getattr(result, "confidence", 0.0)),
        assertions=assertions,
        route=route,
        ayat_refs=list(getattr(result, "ayat_refs", []) or []),
    )
