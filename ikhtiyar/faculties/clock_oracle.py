"""
faculties/clock_oracle.py — Abjad Clock Oracle

Computes geometric coordinates for Quranic roots in the 28-position Abjad
clock space. Provides angular proximity scoring as a SUPPORTING signal for
Bilal — never conflated with TMQ hyperedge data.

Two independent axes per root:
  angle_deg  — structural: clock-face position derived from consonant order
  theta0     — modal: angle from real axis in discriminant complex plane

The four-quadrant entity mapping (HUMAN/JINN/ADVERSARIAL/ANGELIC) is a
ROTATION property of the full Ayat al-Kursi, not a fixed property of
individual roots. ClockOracle does not assert quadrant membership for
individual roots — it reports D_class (REAL/CMPLX) as the modal indicator.

Usage:
    oracle = ClockOracle()
    oracle.build(root_list)           # precompute — call once at startup
    meta = oracle.get("ktb")          # RootClockMeta or None
    bonus = oracle.angular_bonus("ktb", "Amn")  # 0.0–0.15, supporting only
    annotations = oracle.annotate(["ktb", "Amn"])  # list of dicts for prompt
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 28 Arabic consonants in Abjad sequential order (Buckwalter)
ABJAD_SEQ = [
    'A','b','j','d','h','w','z','H','T','y',
    'k','l','m','n','s','E','f','S','q','r',
    '$','t','v','x','*','D','Z','g'
]
ABJAD_POS = {c: i + 1 for i, c in enumerate(ABJAD_SEQ)}   # letter → 1-based position

# Eastern Abjad numerical values (for discriminant)
ABJAD_VAL = {
    'A':1,  'b':2,  'j':3,  'd':4,  'h':5,  'w':6,  'z':7,  'H':8,  'T':9,  'y':10,
    'k':20, 'l':30, 'm':40, 'n':50, 's':60, 'E':70, 'f':80, 'S':90, 'q':100,'r':200,
    '$':300,'t':400,'v':500,'x':600,'*':700,'D':800,'Z':900,'g':1000,
}

# Maximum angular bonus applied to Bilal semantic scores — clock is supporting only
MAX_ANGULAR_BONUS = 0.15


@dataclass
class RootClockMeta:
    root:      str
    clock_pos: float          # average Abjad position (1–28)
    angle_deg: float          # structural clock-face angle (0–360°)
    D:         Optional[int]  # discriminant b²−4ac (Eastern Abjad values)
    D_class:   str            # REAL (D>0) / CMPLX (D<0) / ZERO / WAQF
    theta0:    Optional[float]  # modal angle from real axis (°), only when D<0


class ClockOracle:
    """
    Geometric coordinate system for Quranic roots.

    Supporting signal only.
    TMQ hyperedge data lives in TMQGraph / deliberate().
    Clock data lives here.
    They do not mix.
    """

    def __init__(self) -> None:
        self._meta: dict[str, RootClockMeta] = {}

    # ── Public API ──────────────────────────────────────────────────────────────

    def build(self, roots: list[str]) -> None:
        """Precompute clock metadata for every root. O(n), pure arithmetic."""
        built = 0
        for root in roots:
            m = self._compute(root)
            if m:
                self._meta[root] = m
                built += 1
        logger.info(f"ClockOracle: {built}/{len(roots)} roots indexed")

    def get(self, root: str) -> Optional[RootClockMeta]:
        return self._meta.get(root)

    def annotate(self, roots: list[str]) -> list[dict]:
        """
        Return clock coordinates for a list of roots.
        Format suitable for constrained_prompt injection (separate labeled section).
        """
        out = []
        for r in roots:
            m = self._meta.get(r)
            if not m:
                continue
            entry = {
                "root":      r,
                "angle_deg": round(m.angle_deg, 1),
                "D_class":   m.D_class,
            }
            if m.theta0 is not None:
                entry["theta0"] = round(m.theta0, 1)
            out.append(entry)
        return out

    def angular_bonus(self, root: str, reference_root: str) -> float:
        """
        Additive proximity bonus for Bilal scoring.

        Returns 0.0–MAX_ANGULAR_BONUS based on angular closeness on the clock face.
        0° apart → MAX_ANGULAR_BONUS.  180° apart → 0.0.

        Never replaces semantic score — always additive.
        """
        m1 = self._meta.get(root)
        m2 = self._meta.get(reference_root)
        if not m1 or not m2:
            return 0.0
        diff = abs(m1.angle_deg - m2.angle_deg)
        diff = min(diff, 360.0 - diff)   # shortest arc
        return MAX_ANGULAR_BONUS * (1.0 - diff / 180.0)

    def k_nearest(
        self,
        root: str,
        k: int = 5,
        candidates: list = None,
    ) -> list[dict]:
        """
        Return k roots from candidates (or all indexed roots) nearest to root
        on the clock face, sorted by shortest arc ascending.

        Works even if root is not in _meta — computes its angle dynamically.

        Args:
            root:       Query root (Buckwalter). May not be in the indexed set.
            k:          Number of results to return.
            candidates: If provided, restrict the search pool to these roots.
                        Each must be in _meta. Unknown candidates are skipped.

        Returns:
            list of dicts: [{"root", "angle_deg", "arc_deg", "D_class"}, ...]
            sorted by arc_deg ascending (0° = same position, 180° = antipodal).
        """
        # Get or compute the target root's metadata
        target_meta = self._meta.get(root)
        if not target_meta:
            target_meta = self._compute(root)
        if not target_meta:
            return []

        pool = candidates if candidates is not None else list(self._meta.keys())

        results = []
        for r in pool:
            m = self._meta.get(r)
            if not m:
                continue
            diff = abs(target_meta.angle_deg - m.angle_deg)
            arc = min(diff, 360.0 - diff)   # shortest arc on the clock face
            results.append({
                "root":      r,
                "angle_deg": round(m.angle_deg, 1),
                "arc_deg":   round(arc, 2),
                "D_class":   m.D_class,
            })

        return sorted(results, key=lambda x: x["arc_deg"])[:k]

    def format_prompt_block(self, roots: list[str]) -> str:
        """
        Format clock annotations as a clearly labeled prompt block.
        Injected AFTER the TMQ walk section — never mixed with it.
        """
        annotations = self.annotate(roots)
        if not annotations:
            return ""
        lines = ["[CLOCK ORACLE — structural/modal coordinates, supporting signal only]"]
        for a in annotations:
            theta_str = f"  θ₀={a['theta0']}°" if "theta0" in a else ""
            lines.append(
                f"  {a['root']}: clock={a['angle_deg']}°  modal={a['D_class']}{theta_str}"
            )
        return "\n".join(lines)

    # ── Private ─────────────────────────────────────────────────────────────────

    def _compute(self, root: str) -> Optional[RootClockMeta]:
        if len(root) != 3:
            return None
        a_ch, b_ch, c_ch = root[0], root[1], root[2]

        positions = [ABJAD_POS.get(c) for c in (a_ch, b_ch, c_ch)]
        if None in positions:
            return None

        avg_pos   = sum(positions) / 3.0
        angle_deg = ((avg_pos - 1) / 28.0) * 360.0

        a_v = ABJAD_VAL.get(a_ch)
        b_v = ABJAD_VAL.get(b_ch)
        c_v = ABJAD_VAL.get(c_ch)

        D       = None
        D_class = "WAQF"
        theta0  = None

        if None not in (a_v, b_v, c_v):
            D = b_v * b_v - 4 * a_v * c_v
            if D > 0:
                D_class = "REAL"
            elif D < 0:
                D_class = "CMPLX"
                theta0  = math.degrees(math.atan2(math.sqrt(-D), abs(b_v)))
            else:
                D_class = "ZERO"

        return RootClockMeta(
            root=root, clock_pos=avg_pos, angle_deg=angle_deg,
            D=D, D_class=D_class, theta0=theta0,
        )
