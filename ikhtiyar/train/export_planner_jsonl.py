#!/usr/bin/env python3
"""
Export Shahid Operator planner SFT examples from orchestrator trace JSONL.

Each example: state summary -> next ToolCall JSON (canonical plan or trace-derived).

Usage:
  python ikhtiyar/train/export_planner_jsonl.py
  python ikhtiyar/train/export_planner_jsonl.py --traces ikhtiyar/sessions/traces --out ikhtiyar/train/data/planner_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_TRACES = _REPO / "ikhtiyar" / "sessions" / "traces"
_DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "planner_sft.jsonl"

# Canonical happy path (labels for demos / rule teacher)
CANONICAL_PLAN = [
    {"organ": "ingress", "tool": "ingest_event", "args": {"source": "user"}},
    {"organ": "bilal", "tool": "extract_roots", "args": {}},
    {"organ": "tmq", "tool": "walk", "args": {"depth": 2}},
    {"organ": "circuit", "tool": "evaluate", "args": {}},
    {"organ": "constitution", "tool": "standing_orders", "args": {"limit": 500}},
    {"organ": "mushaf", "tool": "read_ayah", "args": {}},
    {"organ": "shahid", "tool": "deliver_message", "args": {}},
]

SYSTEM = (
    "You are the Shahid Operator planner for QUS-AI. "
    "Output exactly one JSON object: {\"organ\",\"tool\",\"args\"}. "
    "Never bypass Mizan. Scripture quotes require mushaf.read_ayah first. "
    "Contingent witness — not SOURCE."
)


def _load_traces(trace_dir: Path) -> list[Path]:
    if not trace_dir.exists():
        return []
    return sorted(trace_dir.glob("trace_*.jsonl"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _state_summary(events: list[dict], idx: int, user_text: str = "") -> str:
    prior = events[:idx]
    roots: list[str] = []
    tier = "UNKNOWN"
    last_organ = "none"
    blocks = 0
    for e in prior:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "mizan_block":
            blocks += 1
        if t == "organ_report" and p.get("ok"):
            last_organ = p.get("organ", last_organ)
            art = p.get("artifacts") or {}
            if art.get("roots_bw"):
                roots = art["roots_bw"]
            if art.get("tier"):
                tier = art["tier"]
    return (
        f"user_text={user_text!r}\n"
        f"roots_bw={roots}\n"
        f"tier={tier}\n"
        f"last_organ={last_organ}\n"
        f"mizan_blocks={blocks}\n"
        f"step={idx}"
    )


def examples_from_trace(path: Path) -> list[dict]:
    events = _read_jsonl(path)
    if any(e.get("type") == "mizan_block" for e in events):
        # Keep block->recovery pairs only if a later tool_call ALLOW implied
        pass  # still export; DPO script can filter

    user_text = ""
    for e in events:
        if e.get("type") == "tool_call":
            args = (e.get("payload") or {}).get("args") or {}
            if (e.get("payload") or {}).get("tool") == "ingest_event":
                user_text = args.get("text", user_text)

    out = []
    for i, e in enumerate(events):
        if e.get("type") != "tool_call":
            continue
        payload = e.get("payload") or {}
        call = {
            "organ": payload.get("organ"),
            "tool": payload.get("tool"),
            "args": payload.get("args") or {},
        }
        out.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": _state_summary(events, i, user_text),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(call, ensure_ascii=False),
                    },
                ],
                "source_trace": path.name,
                "session_id": events[0].get("session_id") if events else None,
            }
        )
    return out


def examples_from_canonical(user_text: str) -> list[dict]:
    out = []
    for i, step in enumerate(CANONICAL_PLAN):
        call = dict(step)
        args = dict(call.pop("args", {}))
        if call["tool"] == "ingest_event":
            args["text"] = user_text
            args.setdefault("source", "user")
        call["args"] = args
        out.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": _state_summary(
                            [{"type": "synthetic"}] * i, i, user_text
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(call, ensure_ascii=False),
                    },
                ],
                "source": "canonical_plan",
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=Path, default=_DEFAULT_TRACES)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--include-canonical",
        action="store_true",
        help="Add teacher examples for one demo question",
    )
    p.add_argument("--demo-text", default="patience")
    args = p.parse_args()

    examples: list[dict] = []
    for path in _load_traces(args.traces):
        examples.extend(examples_from_trace(path))

    if args.include_canonical or not examples:
        examples.extend(examples_from_canonical(args.demo_text))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
