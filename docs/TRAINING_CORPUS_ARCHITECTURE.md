# Training Corpus Architecture (post-tools)

**Prerequisite:** Working tool-bus, TraceMemory, Mizan blocks, Belief promotion.  
**This document:** how to train later — **no GPU runs in the organ build phase.**

---

## 1. What we do NOT train on

- Mixed 180k-line CLM soup (Quran + English + templates)
- Free-form chat logs without tool structure
- Unverified web text as positive examples

---

## 2. Trace format (source of truth)

Every session writes `ikhtiyar/sessions/traces/trace_<session>.jsonl`:

| Event type | Training use |
|------------|----------------|
| `tool_call` | Planner SFT: state → next call |
| `organ_report` | Tool outcome validation |
| `mizan_block` | **Negative** preference / reward |
| `edge` | Provenance graph for DPO chains |

### Negative examples (your request)

`mizan_block` rows include:

- `checkpoint` (fajr, dhuhr, asr, …)
- `rule_id`
- `original_call`

**DPO pair:** rejected = call that caused block; chosen = regenerated call after block (if ALLOW).

---

## 3. Three model roles (after tools stable)

| Model | Input | Output | When |
|-------|--------|--------|------|
| **Planner** | Session state + last N trace events | Next `ToolCall` JSON | After 1k+ real traces |
| **Render** | Proof tree + GBNF + literals | Arabic / translated segments | After planner works |
| **Verifier** (optional) | Claim + pointers | Score (use circuit as RM) | DPO reward — HVT tier × confidence |

**87M pretrain** only for Arabic fluency on mushaf/hadith text — **not** for tool planning.

---

## 4. Corpus layers

### Layer A — Language (pretrain)

- Sources: `quran-simple.txt`, Arabic hadith from training corpus, morphology tags
- Objective: CLM with `[ARA]` markers only
- Hold-out: by surah

### Layer B — Tool imitation (SFT)

From JSONL traces:

```json
{"role":"system","content":"You are Shahid planner. Output one ToolCall JSON."}
{"role":"user","content":"<state_summary>"}
{"role":"assistant","content":"{\"organ\":\"tmq\",\"tool\":\"walk\",\"args\":{\"roots\":[\"sbr\"],\"depth\":2}}"}
```

Filter: sessions with zero `mizan_block` or include block→recovery pairs explicitly.

### Layer C — Preference (DPO)

- **Chosen:** trace path that reached `promote_belief` with T0/T1 chain
- **Rejected:** path containing `mizan_block` or T2 web promoted without chain

Circuit evaluator can supply scalar reward without separate RM.

---

## 5. Integration with organs

| Organ | Training signal |
|-------|-----------------|
| Bilal, TMQ, Circuit, Mushaf | No LLM — deterministic |
| Render | GBNF-constrained SFT on `{proof, grammar} → literal render` |
| Web | Never positive alone; only in rejected pairs |
| Shahid planner | SFT on ToolCall sequences |

---

## 6. Evaluation gates (before any training run)

1. Planner accuracy on held-out sessions (exact organ+tool match %)
2. Mizan block rate on eval set (should decrease, not increase)
3. Mushaf organ: 100% verbatim match on random ayah keys
4. Zero aseity passes on red-team `deliver_message` prompts

---

## 7. Research directions (from scratch)

- **Grammar-masked planner:** emit ToolCall JSON via GBNF (smaller vocab than Arabic open gen)
- **Retrieve-then-render:** never train Quran quotes — only train glue words
- **Multi-agent MoA:** train fusion policy separately from organs (classifier on Claim lists)
- **Arabic-first planner:** Arabic state summaries from Bilal, English UI only at Render

---

## 8. Next steps when you say "train"

1. Run batch demos → accumulate traces: `python ikhtiyar/scripts/run_trace_batch.py`
2. Export JSONL: `python ikhtiyar/train/export_planner_jsonl.py --include-canonical`
3. Small SFT (LoRA on 7B planner) OR rules-only planner until N>500
4. Only then revisit 87M render model

**Full MoA / Shahid Operator regimen:** [`MOA_SHAHID_TRAINING_REGIMEN.md`](MOA_SHAHID_TRAINING_REGIMEN.md)

والله أعلم — Allah knows best.
