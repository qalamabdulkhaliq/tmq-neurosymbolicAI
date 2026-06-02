#!/usr/bin/env python3
"""
Run orchestrator demos over a question list to accumulate trace JSONL.

Usage:
  python ikhtiyar/scripts/run_trace_batch.py
  python ikhtiyar/scripts/run_trace_batch.py --questions ikhtiyar/train/data/seed_questions.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_IKHTIYAR = _REPO / "ikhtiyar"
sys.path.insert(0, str(_IKHTIYAR))

def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="replace").decode("cp1252", errors="replace"))


DEFAULT_QUESTIONS = [
    "patience",
    "ما معنى الصبر؟",
    "What does the Quran command about prayer?",
    "guidance",
    "trust in Allah",
    "forgiveness",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, help="One question per line UTF-8")
    p.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    p.add_argument(
        "--reuse-session",
        action="store_true",
        help="Reuse a single orchestrator/session for all questions (default: new session per question)",
    )
    args = p.parse_args()

    if args.questions and args.questions.exists():
        lines = [
            ln.strip()
            for ln in args.questions.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        lines = DEFAULT_QUESTIONS

    if args.limit:
        lines = lines[: args.limit]

    from orchestrator.shahid import ShahidOrchestrator  # noqa: WPS433

    orch = ShahidOrchestrator() if args.reuse_session else None
    ok = 0
    for q in lines:
        if orch is None:
            orch = ShahidOrchestrator()
        out = orch.run_cycle(q)
        delivered = bool(out.get("delivered"))
        circuit = next((s for s in out.get("steps", []) if s[0] == "circuit"), None)
        _safe_print(
            f"[{'OK' if delivered else 'WAQF'}] {q[:50]!r} "
            f"session={out.get('session_id')} circuit={circuit}"
        )
        if delivered:
            ok += 1
        if not args.reuse_session:
            orch = None

    _safe_print(
        f"Done: {ok}/{len(lines)} delivered. Traces under ikhtiyar/sessions/traces/"
    )
    return 0 if ok == len(lines) else 1


if __name__ == "__main__":
    raise SystemExit(main())
