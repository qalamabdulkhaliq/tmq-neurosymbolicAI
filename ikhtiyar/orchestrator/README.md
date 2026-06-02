# Orchestrator — tool-bus core

## Gap closure status (May 2026)

| Gap | Status |
|-----|--------|
| Bilal load (concept_map + constitution) | Done — `organs/bilal_organ.py`, `roots_bw` for TMQ |
| TMQ/Circuit in `run_cycle` | Done — BW roots; graceful skip if files missing |
| Mushaf ayah from circuit/standing orders | Done — not hardcoded 1:1 only |
| Mizan regeneration loop | Done — `regenerate_deliver()`, blocks in trace |
| `engine.chat()` orchestrator hook | Done — default on; `QUS_LEGACY_CHAT=1` for old path |
| Web organ | Done — T2 hypothesis; optional DuckDuckGo if `requests` |
| Belief `promote_with_chain` | Done — requires T0/T1 trace in chain |
| Tests | `test_orchestrator.py` extended |
| Arabic **صبر** → **Sbr** | Done — `utils/concept_mapping.json`, `organs/lexicon_bridge.py` |
| Mushaf from standing orders (`Sbr` → 3:200) | Done — `s:v` refs + `_loc_from_standing_orders` |
| Web fetch (T2) | Done — `requests.get` in `web_organ.py` |
| Engine cycle grounding | Opt-in — `QUS_ORCHESTRATOR_CYCLE=1` in `engine._run_one_cycle` |

## Run demo (no LLM)

```powershell
cd "C:\Users\amlan\OneDrive\Desktop\QUS-AI Islamic Alignment"
python ikhtiyar/run_orchestrator_demo.py
python ikhtiyar/run_orchestrator_demo.py "ما معنى الصبر؟"
python ikhtiyar/run_orchestrator_demo.py "patience"
```

## Run tests

```powershell
python -m pytest ikhtiyar/tests/test_orchestrator.py -q
```

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `QUS_USE_ORCHESTRATOR` | `1` | `engine.chat()` uses tool-bus |
| `QUS_LEGACY_CHAT` | off | `1` → old `_chat_deliberate` |
| `QUS_ORCHESTRATOR_CYCLE` | **`1`** | `0` → skip `ground_question()` in `_run_one_cycle` |
| `TMQ_PATH` | `bismillah/TMQ_v12.json` | TMQ hypergraph |
| `HVT_PATH` | `ikhtiyar/TMQ_hvt.json` | Circuit tape |
| `QUS_DISCORD_WEBHOOK` | — | Voice organ |

## Flow

`ShahidOrchestrator.run_cycle` → each tool → `MizanGate.admit` → organ → `TraceMemory` JSONL.

Beliefs: `BeliefStore.promote_with_chain` with T0/T1 provenance.
