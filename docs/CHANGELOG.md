# Changelog

All notable changes to QUS-AI (`ikhtiyar/` active build) are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Process for new entries: [`HOW_WE_LOG.md`](HOW_WE_LOG.md).

Each bullet includes **What**, **Why**, **Where**, and **How** (verify).

---

## [Unreleased]

### Fixed

- **What:** Constitution NAHY refinement: HVT laA-after filter, AMR-primary boundaries, Python active_command_set builder; Sbr removed from boundaries.
  - **Why:** Verse-level laA+JUS mis-tagged patience (3:120, 12:90) as prohibition; blocked Circuit and taught wrong Mizan negatives.
  - **Where:** `ikhtiyar/core/speech_act_refine.py`, `build_active_command_set.py`, `scripts/rebuild_constitution_pipeline.py`, `ikhtiyar/tools/enrich_speech_acts.py`, `full_quran_constitution.json`, `active_command_set.json`, `mizan_gate.py`
  - **How:** python ikhtiyar/scripts/rebuild_constitution_pipeline.py; python -m pytest ikhtiyar/tests/test_speech_act_refine.py -q; python ikhtiyar/run_orchestrator_demo.py patience



### Added — Tool-bus orchestrator (May 2026)

- **What:** Core orchestrator package: typed contracts (`ToolCall`, `OrganReport`, `Claim`, `MizanVerdict`, `SessionState`), trust tiers, Mizan gate (`admit` / `audit`), append-only trace memory, belief store with provenance chains, tool executor, and `ShahidOrchestrator.run_cycle` (no LLM).
  - **Why:** Replace monolithic `engine.chat()` reasoning with an explicit tool-bus where every step is logged, gated, and replayable before any planner/render model is trained.
  - **Where:** `ikhtiyar/orchestrator/schemas.py`, `trust.py`, `mizan_gate.py`, `memory_trace.py`, `belief_store.py`, `executor.py`, `shahid.py`, `ikhtiyar/orchestrator/README.md`
  - **How:** `python -m pytest ikhtiyar/tests/test_orchestrator.py -q`

- **What:** Deterministic organ implementations and machine-readable ACL registry (ingress, bilal, tmq, circuit, mushaf, constitution, hadith, web, render stub, voice, shahid).
  - **Why:** Scripture and graph tools must return catalog literals and trust-tiered claims—not free-generated quotes—so Mizan and belief promotion have stable inputs.
  - **Where:** `ikhtiyar/organs/*.py`, `ikhtiyar/organs/registry.yaml`, `ikhtiyar/organs/__init__.py` (`ORGAN_CLASSES`)
  - **How:** `python ikhtiyar/run_orchestrator_demo.py "patience"` (expect `delivered` or trace path; no GPU)

- **What:** CLI demo for the full agentic loop without loading an LLM.
  - **Why:** Lets collaborators smoke-test organs + Mizan + trace/belief writes in seconds on any machine.
  - **Where:** `ikhtiyar/run_orchestrator_demo.py`
  - **How:** `python ikhtiyar/run_orchestrator_demo.py` (exit 0 if message delivered)

- **What:** Orchestrator test suite (Mizan Asr deny/allow, ingress/mushaf/constitution organs, memory block logging, belief promotion, regeneration after aseity, engine env flag).
  - **Why:** Lock non-negotiable safety (Asr) and provenance rules into CI-local pytest without Ollama.
  - **Where:** `ikhtiyar/tests/test_orchestrator.py`
  - **How:** `python -m pytest ikhtiyar/tests/test_orchestrator.py -q`

### Added — Architecture & training docs (May 2026)

- **What:** Organ architecture spec v0.3 (trust tiers T0–T3, Mizan flow, memory vs belief, organ table).
  - **Why:** Single implementation authority so agents and humans agree on tool-bus invariants (no chat API; Mizan non-overridable).
  - **Where:** `docs/ORGAN_ARCHITECTURE.md`
  - **How:** Read sections “Principles” and “Mizan flow”; cross-check `ikhtiyar/organs/registry.yaml`

- **What:** Six-hour implementation plan for the core agentic loop (phases 0–5, acceptance criteria).
  - **Why:** Time-boxed surgical build order for the May 19 2026 session; separates “build tools” from “train models.”
  - **Where:** `docs/IMPLEMENTATION_PLAN.md`
  - **How:** Confirm Phase 0 checkboxes against files listed in this changelog

- **What:** Post-tools training corpus architecture (trace JSONL → SFT/DPO layers; explicit exclusions).
  - **Why:** Document how to train *after* traces exist—without mixing CLM soup or unverified web text into positive examples.
  - **Where:** `docs/TRAINING_CORPUS_ARCHITECTURE.md`
  - **How:** Inspect `ikhtiyar/sessions/traces/trace_*.jsonl` after a demo run; compare event types to doc §2

### Changed — Gap closure (May 2026)

- **What:** Engine defaults: QUS_ORCHESTRATOR_CYCLE=1; MoA/Shahid training regimen doc; trace batch + planner JSONL export scripts.
  - **Why:** Enable cycle grounding by default and document build/train path for Shahid Operator and MoA agents on corrected constitution.
  - **Where:** `ikhtiyar/engine.py`, `docs/MOA_SHAHID_TRAINING_REGIMEN.md`, `ikhtiyar/train/export_planner_jsonl.py`, `ikhtiyar/scripts/run_trace_batch.py`
  - **How:** python -m pytest ikhtiyar/tests/test_orchestrator.py::test_engine_orchestrator_flag -q; python ikhtiyar/train/export_planner_jsonl.py --include-canonical

- **What:** Foreground gap closure: Arabic lexicon (صبر→Sbr), Mushaf standing-order fallback (3:200), circuit ayah ref normalization, web fetch, optional engine cycle grounding.
  - **Why:** Close remaining orchestrator gaps so Arabic queries and patience map to correct BW roots and ayah without hardcoded 1:1; enable T2 fetch and optional full-engine grounding.
  - **Where:** `ikhtiyar/utils/concept_mapping.json`, `ikhtiyar/organs/lexicon_bridge.py`, `bilal_organ.py`, `mushaf_organ.py`, `circuit_organ.py`, `web_organ.py`, `engine.py`, `ikhtiyar/tests/test_orchestrator.py`
  - **How:** python -m pytest ikhtiyar/tests/test_orchestrator.py -q; python ikhtiyar/run_orchestrator_demo.py " ما معنى الصبر؟\

- **What:** `BilalOrgan` loads concept map + constitution paths; exposes `extract_roots` with `roots_bw` for TMQ/circuit.
  - **Why:** Orchestrator cycle needs deterministic roots from Arabic/English without re-embedding the old middleware stack.
  - **Where:** `ikhtiyar/organs/bilal_organ.py`
  - **How:** `python -m pytest ikhtiyar/tests/test_orchestrator.py::test_bilal_extract_roots_if_ready -q` (skips if data missing)

- **What:** `ShahidOrchestrator` wires TMQ walk + circuit evaluate when `TMQ_v12.json` / `TMQ_hvt.json` exist; mushaf target from circuit refs or constitution standing orders (not only 1:1).
  - **Why:** Close “graph + tape disconnected from demo” gap; deliver ayah-backed output when ground data is present.
  - **Where:** `ikhtiyar/orchestrator/shahid.py`
  - **How:** Run demo with `TMQ_PATH` and `HVT_PATH` set; check trace for `organ_report` rows from `tmq` / `circuit`

- **What:** Mizan regeneration path: `regenerate_deliver()` after Asr DENY; blocks recorded in trace.
  - **Why:** Blocked calls must be recoverable with a safe replacement message, not silent failure.
  - **Where:** `ikhtiyar/orchestrator/shahid.py`, `ikhtiyar/orchestrator/mizan_gate.py`
  - **How:** `python -m pytest ikhtiyar/tests/test_orchestrator.py::test_regeneration_on_aseity -q`

- **What:** `IkhtiyarEngine.chat()` routes through orchestrator by default (`QUS_USE_ORCHESTRATOR=1`); legacy path via `QUS_LEGACY_CHAT=1`.
  - **Why:** Wire the tool-bus into the existing FastAPI server without deleting the old deliberate path yet.
  - **Where:** `ikhtiyar/engine.py` (`_use_orchestrator_chat`, `_chat_orchestrator`)
  - **How:** `python -m pytest ikhtiyar/tests/test_orchestrator.py::test_engine_orchestrator_flag -q`

- **What:** `BeliefStore.promote_with_chain` requires T0/T1 trace pointers in the promotion chain.
  - **Why:** Web (T2) and blocked claims must not become beliefs without provenance to ground truth.
  - **Where:** `ikhtiyar/orchestrator/belief_store.py`
  - **How:** `python -m pytest ikhtiyar/tests/test_orchestrator.py::test_belief_promote_with_chain -q`

- **What:** `WebOrgan` returns T2 hypothesis claims; optional DuckDuckGo when `requests` is installed.
  - **Why:** External fetch is allowed only as untrusted observation, never auto-promoted to belief.
  - **Where:** `ikhtiyar/organs/web_organ.py`
  - **How:** Call `web.search` via executor in a REPL or extend orchestrator cycle; confirm `trust: T2` in report

### Not done (explicit)

- **What:** No new LLM pretrain, SFT, DPO, or LoRA runs in the May 2026 orchestrator session.
  - **Why:** Tool-bus, Mizan blocks, and trace format must exist before planner/render training (see `TRAINING_CORPUS_ARCHITECTURE.md`).
  - **Where:** N/A (training scripts unchanged: `ikhtiyar/train/*`, `ikhtiyar/lora_train.py`)
  - **How:** Do not run `pretrain.py` / `lora_train.py` as part of orchestrator verification; use pytest + demo only

---

## [2026-05-19] — Orchestrator session baseline

Dated snapshot of the same work as **[Unreleased]** above, for collaborators who prefer a fixed date anchor. Move bullets to a new dated section when releasing; keep **[Unreleased]** for in-flight work only.

---

## Template (copy for new entries)

```markdown
- **What:** <one-line change>
  - **Why:** <intent / problem solved>
  - **Where:** `path/to/file.py`, ...
  - **How:** `<verify command>`
```
