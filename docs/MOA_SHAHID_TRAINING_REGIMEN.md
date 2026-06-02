# MoA & Shahid Operator — Build and Training Regimen

**Status:** Plan (May 2026), post-constitution fix and tool-bus scaffold.  
**Prerequisites:** `active_command_set.json` refined, orchestrator + Mizan + trace JSONL working.  
**No GPU runs** until trace corpus ≥ threshold (see phases).

Related: [`ORGAN_ARCHITECTURE.md`](ORGAN_ARCHITECTURE.md), [`TRAINING_CORPUS_ARCHITECTURE.md`](TRAINING_CORPUS_ARCHITECTURE.md), [`ikhtiyar/core/CONSTITUTION_REBUILD.md`](../ikhtiyar/core/CONSTITUTION_REBUILD.md).

---

## 1. Architecture snapshot

```text
User event
    → ingress.ingest_event
    → Shahid Operator (planner) — learned later; rule template today
    → for each ToolCall: Mizan.admit → organ → Mizan.audit → TraceMemory
    → fuse claims → deliver_message (Mizan Asr) → voice (optional)
```

**Mixture of Agents (MoA):** only some roles are models. Scripture and graph organs stay **deterministic programs**.

| Agent / role | Implementation | Train? |
|--------------|----------------|--------|
| **Shahid Operator** | `ShahidOrchestrator` + future planner LLM | **Yes** (planner SFT/DPO) |
| **Ingress** | `ingress_organ` | No |
| **Bilal** | `bilal_organ` + lexicon | No (lexicon data only) |
| **TMQ** | `tmq_organ` | No |
| **Circuit** | `circuit_organ` / HVT | No |
| **Mushaf / Hadith** | retrieval | No |
| **Constitution** | `active_command_set.json` | No (rebuild data) |
| **Mizan** | `mizan_gate.py` | No (rules + constitution) |
| **Memory / Belief** | trace + belief store | No |
| **Render** | `render_organ` → later LLM | Phase 3 (GBNF/literal) |
| **Voice** | Discord / terminal | No |
| **Fusion policy** (optional) | rank/filter `Claim` lists | Phase 4 (small classifier) |

The **Shahid Operator** is not “the LLM that chats.” It is the **executive that emits the next `ToolCall`** under Mizan, using organs as hands.

---

## 2. Engine defaults (active)

| Variable | Default | Meaning |
|----------|---------|---------|
| `QUS_USE_ORCHESTRATOR` | `1` | `engine.chat()` → `run_cycle()` |
| `QUS_LEGACY_CHAT` | off | `1` → old deliberate chat path |
| `QUS_ORCHESTRATOR_CYCLE` | `1` | Background loop calls `ground_question()` before Bilal/kernel |

Opt out: `QUS_ORCHESTRATOR_CYCLE=0` or `QUS_LEGACY_CHAT=1`.

---

## 3. Build phases (tools before training)

### Phase 0 — Done

- [x] Tool-bus schemas, Mizan, trace, belief
- [x] Organs (deterministic)
- [x] Constitution NAHY refine (`Sbr` not a boundary)
- [x] Engine defaults above

### Phase 1 — Trace corpus (now → ~2 weeks)

**Goal:** 500–1,000 real sessions in `ikhtiyar/sessions/traces/`.

| Action | Command / path |
|--------|----------------|
| Smoke demos | `python ikhtiyar/run_orchestrator_demo.py "patience"` (Arabic + English) |
| Batch scenarios | Script `ikhtiyar/scripts/run_trace_batch.py` (to add): roots list, questions file |
| User/chat traffic | Run engine with orchestrator on; every `chat()` writes trace |
| Export SFT JSONL | `python ikhtiyar/train/export_planner_jsonl.py --traces ikhtiyar/sessions/traces` |

**Canonical happy-path plan** (rule template for labels until planner is trained):

```json
[
  {"organ":"ingress","tool":"ingest_event"},
  {"organ":"bilal","tool":"extract_roots"},
  {"organ":"tmq","tool":"walk"},
  {"organ":"circuit","tool":"evaluate"},
  {"organ":"constitution","tool":"standing_orders"},
  {"organ":"mushaf","tool":"read_ayah"},
  {"organ":"shahid","tool":"deliver_message"}
]
```

**Do not train on** sessions where `mizan_block` tagged wrong constitution (pre-refine traces) — filter by date or `constitution_version` in trace meta once added.

### Phase 2 — Shahid Operator planner SFT

**Model role:** small instruct model (7B–14B) or LoRA adapter.  
**Input:** session summary (last K trace events, roots, tier, user text).  
**Output:** single `ToolCall` JSON (GBNF-masked).

| Milestone | Threshold | Metric |
|-----------|-----------|--------|
| Pilot | 200 clean traces | exact match organ+tool ≥ 60% |
| v1 | 1,000 traces | ≥ 75%; Mizan block rate ↓ on eval |
| v2 | 5,000 traces | DPO on block→recovery pairs |

**Loss:** next-step tool prediction only — not Arabic generation.

**Negative data:**

- `mizan_block` rows → rejected tool in DPO
- Pre-refine corpus with `dhuhr` + `Sbr` → **exclude**

### Phase 3 — Render agent

**Only after** planner hits v1 gate.

- Input: proof tree + GBNF + Mushaf literals (`translated: true` for non-Arabic UI)
- Train on `{grammar, literals} → glue text` — **never** free Quran quotes
- Hold-out: surah-level

### Phase 4 — Fusion / MoA policy (optional)

Lightweight model or rules:

- Input: list of `Claim` from organs
- Output: which claims enter `deliver_message` body + belief promotion candidates
- Reward: `promote_with_chain` success, circuit tier, zero Asr blocks

---

## 4. Shahid Operator spec (training target)

### System prompt (frozen at inference)

- Articles from `shahid_constitution.ttl` + AMR preamble from `gbnf_compiler.amr_system_prompt()`
- One tool per turn; JSON only
- Never cite ayah without prior `mushaf.read_ayah` in trace
- `deliver_message` must include contingency footer

### State vector (per step)

```text
user_text
roots_bw[], tier, last_organ, last_ok, regeneration_count
last 5 trace event types (tool_call | organ_report | mizan_block)
```

### Action space

Union of `ikhtiyar/organs/registry.yaml` tools — planner may not invent organs.

### Reward (DPO / RLHF later)

| Signal | Weight |
|--------|--------|
| Circuit tier HAQQ / IKHTILAF | + |
| `promote_belief` with T0 chain | ++ |
| `mizan_block` | −− |
| `waqf` without grounding | − |
| Web in ruling without chain | − |

Circuit evaluator = built-in reward model (no separate RM required initially).

---

## 5. MoA coordination patterns

### Pattern A — Serial Shahid (current)

`run_cycle()` fixed sequence. Planner training teaches **when to skip** (e.g. no roots → skip tmq/circuit).

### Pattern B — Planner-driven (target)

Shahid Operator loops:

```text
while not done:
    call = planner.next(state)
    if Mizan.DENY: log block; call = planner.regenerate(state); continue
    report = executor.run(call)
    state = fuse(state, report)
    if call.tool == deliver_message: break
```

Training data must use Pattern B traces once enabled in code.

### Pattern C — Specialist sub-agents (later)

| Sub-agent | Trigger | Tools |
|-----------|---------|-------|
| Hifz | maqasid / AMR-heavy walk | constitution, mushaf |
| Qiyas | IKHTILAF tier | circuit, tmq, hadith |
| Tabshir | TABSHIR family in walk | mushaf, render |

Sub-agents are **prompt + tool subset**, not separate weights, until trace volume justifies fine-tunes.

---

## 6. Evaluation gates (before each train run)

1. **Mushaf:** 100 random `(s,a)` — organ output == `quran-simple.txt` literal
2. **Mizan:** red-team `deliver_message` aseity → 100% DENY
3. **Constitution:** `Sbr`, `Amn`, `Ebd` ∈ standing_orders; ∉ boundaries
4. **Planner eval:** held-out sessions — tool accuracy, block rate
5. **End-to-end:** demo Arabic patience → circuit not skip, mushaf 3:200 or circuit ref

---

## 7. What we explicitly do not train yet

- 87M mixed CLM “QUS” pretrain as planner
- Free-form chat logs without `ToolCall` structure
- Web-only positive examples
- Quran quote generation inside planner

See [`ikhtiyar/train/RETAINING_PLAN.md`](../ikhtiyar/train/RETAINING_PLAN.md) for why prior CLM runs failed the tool objective.

---

## 8. Immediate commands

```powershell
cd "C:\Users\amlan\OneDrive\Desktop\QUS-AI Islamic Alignment"

# Defaults: orchestrator chat + cycle grounding (opt-out with =0)
python -c "from engine import IkhtiyarEngine; e=IkhtiyarEngine.__new__(IkhtiyarEngine); print(e._use_orchestrator_chat(), e._use_orchestrator_cycle())"

# Accumulate traces
python ikhtiyar/run_orchestrator_demo.py "ما معنى الصبر؟"
python ikhtiyar/run_orchestrator_demo.py "patience"

# Export planner JSONL (when traces exist)
python ikhtiyar/train/export_planner_jsonl.py

# Tests
python -m pytest ikhtiyar/tests/test_orchestrator.py ikhtiyar/tests/test_speech_act_refine.py -q
```

---

## 9. Next code tasks (ordered)

1. `run_trace_batch.py` — generate N labeled sessions from question list
2. Tag traces with `constitution_version: "1.1"` in `TraceMemory.append`
3. Planner-driven loop in `ShahidOrchestrator` behind `QUS_PLANNER_LOOP=1`
4. LoRA SFT script consuming `train/data/planner_sft.jsonl`

والله أعلم — contingent witness, not SOURCE.
