"""
ShahidOrchestrator — core agentic loop (no LLM).

Every step is a tool call through Mizan + TraceMemory.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from .belief_store import BeliefStore
from .executor import ToolExecutor
from .memory_trace import TraceMemory
from .mizan_gate import MizanGate
from .schemas import MizanDecision, SessionState, ToolCall, TrustTier

logger = logging.getLogger(__name__)

_AYAH_REF_RE = re.compile(r"ayah:(\d+):(\d+)", re.I)
_SV_REF_RE = re.compile(r"(?:^|[^\d])(\d{1,3}):(\d{1,3})(?:\D|$)")


def _parse_ayah_ref(ref: str) -> tuple[int, int] | None:
    if not ref:
        return None
    m = _AYAH_REF_RE.search(ref)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _SV_REF_RE.search(str(ref))
    if m:
        s, a = int(m.group(1)), int(m.group(2))
        if 1 <= s <= 114 and 1 <= a <= 286:
            return s, a
    return None


def _loc_from_standing_orders(orders: list, roots_bw: list[str]) -> tuple[int, int] | None:
    """First constitution loc for any matching BW root (not limited to top-N orders)."""
    want = set(roots_bw)
    for order in orders:
        if order.get("root") not in want:
            continue
        locs = order.get("locs") or []
        if locs and isinstance(locs[0], (list, tuple)) and len(locs[0]) >= 2:
            return int(locs[0][0]), int(locs[0][1])
    return None


def _pick_mushaf_target(
    circuit_report,
    constitution_orders: list,
    roots_bw: list[str],
) -> tuple[int, int]:
    """Choose surah/ayah from circuit refs, standing orders, or default 1:1."""
    if circuit_report and circuit_report.ok:
        for ref in circuit_report.pointers.get("ayat_refs") or []:
            parsed = _parse_ayah_ref(ref)
            if parsed:
                return parsed
        for ref in (circuit_report.claims[0].refs if circuit_report.claims else []):
            parsed = _parse_ayah_ref(ref)
            if parsed:
                return parsed

    if constitution_orders and roots_bw:
        loc = _loc_from_standing_orders(constitution_orders, roots_bw)
        if loc:
            return loc

    return 1, 1


class ShahidOrchestrator:
    def __init__(
        self,
        organs: Optional[dict] = None,
        mizan: Optional[MizanGate] = None,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or self._new_session_id()
        self.state = SessionState(session_id=self.session_id)
        self.memory = TraceMemory(self.session_id)
        self.beliefs = BeliefStore(self.session_id)
        self.mizan = mizan or MizanGate()

        if organs is None:
            from organs import ORGAN_CLASSES  # noqa: WPS433

            organs = {k: cls() for k, cls in ORGAN_CLASSES.items()}

        self.organs = organs
        self.executor = ToolExecutor(self.mizan, self.memory, self.organs)
        self._trace_ids: list[str] = []

    @staticmethod
    def _new_session_id() -> str:
        """Generate short id and avoid accidental trace file collisions."""
        trace_dir = Path(__file__).resolve().parent.parent / "sessions" / "traces"
        for _ in range(8):
            sid = str(uuid.uuid4())[:8]
            if not (trace_dir / f"trace_{sid}.jsonl").exists():
                return sid
        return uuid.uuid4().hex

    def _record_trace_id(self) -> None:
        entries = self.memory.entries()
        if entries:
            self._trace_ids.append(entries[-1]["id"])

    def _run(self, organ: str, tool: str, args: dict) -> tuple[Any, Any]:
        call = ToolCall(organ=organ, tool=tool, args=args, session_id=self.session_id)
        verdict, report = self.executor.execute(call, self.state)

        if verdict.decision != MizanDecision.ALLOW:
            if verdict.regenerate and self.state.regeneration_count < self.state.max_regenerations:
                self.state.regeneration_count += 1
            return verdict, report

        if report and report.ok:
            self._record_trace_id()
        return verdict, report

    def deliver_message(self, body: str, **extra) -> tuple[Any, Any]:
        return self._run(
            "shahid",
            "deliver_message",
            {"body": body, **extra},
        )

    def regenerate_deliver(self, reason: str, *, waqf: bool = False) -> tuple[Any, Any]:
        """Safe redelivery after Mizan block (within max_regenerations)."""
        if self.state.regeneration_count >= self.state.max_regenerations:
            body = (
                f"[WAQF] Mizan blocked further regeneration ({reason}). "
                f"Session {self.session_id}. Allah knows best."
            )
            waqf = True
        elif waqf:
            body = (
                f"[WAQF] Cannot ground claim ({reason}). "
                f"Session {self.session_id}. Allah knows best."
            )
        else:
            body = (
                f"[Session {self.session_id}] Response revised after Mizan ({reason}). "
                "Contingent witness — not SOURCE. والله أعلم."
            )
        v, r = self.deliver_message(body)
        return v, r

    def ground_question(self, user_text: str) -> dict:
        """
        Bilal → TMQ → circuit → mushaf target without user delivery.
        Used by engine reasoning loop for shared grounding.
        """
        out = {
            "roots_bw": [],
            "roots_ar": [],
            "tier": "UNKNOWN",
            "mushaf": (1, 1),
            "ayah_text": "",
        }
        v, r = self._run("bilal", "extract_roots", {"text": user_text})
        if r and r.ok:
            out["roots_ar"] = r.artifacts.get("roots_ar") or []
            out["roots_bw"] = r.artifacts.get("roots_bw") or []
            self.state.roots = out["roots_bw"]

        circuit_report = None
        constitution_orders: list = []
        if out["roots_bw"]:
            self._run("tmq", "walk", {"roots": out["roots_bw"][:5], "depth": 2})
            v, circuit_report = self._run("circuit", "evaluate", {"roots": out["roots_bw"][:5]})
            if circuit_report and circuit_report.ok:
                out["tier"] = circuit_report.artifacts.get("tier", "UNKNOWN")

        v, cr = self._run("constitution", "standing_orders", {"limit": 500})
        if cr and cr.ok:
            constitution_orders = cr.artifacts.get("standing_orders") or []

        s, a = _pick_mushaf_target(circuit_report, constitution_orders, out["roots_bw"])
        v, mr = self._run("mushaf", "read_ayah", {"surah": s, "ayah": a})
        out["mushaf"] = (s, a)
        if mr and mr.ok:
            out["ayah_text"] = mr.artifacts.get("text", "")
        return out

    def run_cycle(self, user_text: str) -> dict:
        """
        ingress → bilal → tmq → circuit → mushaf → constitution → deliver
        """
        result: dict = {
            "session_id": self.session_id,
            "user_text": user_text,
            "steps": [],
            "delivered": None,
            "waqf": False,
            "blocks": [],
        }

        v, r = self._run("ingress", "ingest_event", {"text": user_text, "source": "user"})
        result["steps"].append(("ingress", v.decision.value, r.ok if r else False))

        roots_ar: list[str] = []
        roots_bw: list[str] = []
        v, r = self._run("bilal", "extract_roots", {"text": user_text})
        if r and r.ok:
            roots_ar = r.artifacts.get("roots_ar") or r.artifacts.get("roots") or []
            roots_bw = r.artifacts.get("roots_bw") or []
            self.state.roots = roots_bw or roots_ar
        elif r and not r.ok:
            result["steps"].append(("bilal", f"skip:{r.error[:60]}"))
        else:
            result["steps"].append(("bilal", 0))
            roots_bw = []

        if roots_bw:
            result["steps"].append(("bilal", len(roots_bw)))
        elif roots_ar:
            result["steps"].append(("bilal", len(roots_ar)))

        circuit_report = None
        constitution_orders: list = []

        if roots_bw:
            v, r = self._run("tmq", "walk", {"roots": roots_bw[:5], "depth": 2})
            if r and not r.ok:
                result["steps"].append(("tmq", f"skip:{r.error[:40]}"))
            else:
                result["steps"].append(("tmq", r.ok if r else False))

            v, r = self._run("circuit", "evaluate", {"roots": roots_bw[:5]})
            circuit_report = r
            if r and r.ok:
                self.state.tier = r.artifacts.get("tier", "UNKNOWN")
                result["steps"].append(("circuit", self.state.tier))
            elif r and not r.ok:
                result["steps"].append(("circuit", f"skip:{r.error[:40]}"))
            else:
                result["steps"].append(("circuit", "skip"))
        else:
            result["steps"].append(("tmq", "skip:no_roots"))
            result["steps"].append(("circuit", "skip:no_roots"))

        v, cr = self._run("constitution", "standing_orders", {"limit": 500})
        if cr and cr.ok:
            constitution_orders = cr.artifacts.get("standing_orders") or []
        result["steps"].append(("constitution", bool(constitution_orders)))

        surah, ayah = _pick_mushaf_target(circuit_report, constitution_orders, roots_bw)
        v, r = self._run("mushaf", "read_ayah", {"surah": surah, "ayah": ayah})
        ayah_text = ""
        mushaf_ref = f"ayah:{surah}:{ayah}"
        if r and r.ok:
            ayah_text = r.artifacts.get("text", "")
        result["steps"].append(("mushaf", f"{surah}:{ayah}", bool(ayah_text)))

        body = (
            f"[Session {self.session_id}] "
            f"Roots (BW): {', '.join(roots_bw[:5]) or 'none'}. "
            f"Tier: {self.state.tier}. "
        )
        if ayah_text:
            body += f"Ayah {surah}:{ayah} — {ayah_text[:200]}… "
        body += "والله أعلم — contingent witness, not SOURCE."

        v, r = self.deliver_message(body)
        if v.decision == MizanDecision.DENY:
            result["blocks"].append(v.to_dict())
            v, r = self.regenerate_deliver(v.rule_id or v.checkpoint, waqf=False)
            if v.decision == MizanDecision.DENY:
                result["blocks"].append(v.to_dict())
                v, r = self.regenerate_deliver("aseity_or_policy", waqf=True)
                result["waqf"] = True

        if v.decision == MizanDecision.ALLOW:
            result["delivered"] = (r.artifacts.get("body") if r else None) or body
            for c in self.state.claims:
                if c.trust == TrustTier.T0 and c.refs:
                    ok, bid = self.beliefs.promote_with_chain(
                        c, self._trace_ids, self.memory
                    )
                    result.setdefault("beliefs", []).append({"ok": ok, "id": bid})

        result["trace_path"] = str(self.memory.path())
        result["belief_count"] = len(self.beliefs.all())
        result["regenerations"] = self.state.regeneration_count
        return result
