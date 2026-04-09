# ikhtiyar — Neurosymbolic AI Safety System

**ikhtiyar** (اختيار — *choice, deliberate will*) is a neurosymbolic AI safety architecture
grounded in the Quranic Arabic Corpus. It replaces RLHF's circular contingency with
an immutable external reference: a fixed hypergraph the system must consult before
generating any output.

The system was built in direct response to Lyons v. OpenAI (2024), in which an
ungrounded LLM validated a user's paranoid delusions, resulting in two deaths.

---

## Architecture

```
User Query
    |
Bilal Perception Engine  — maps query to Quranic roots via embeddings
    |
TMQ Hypergraph Walk      — BFS traversal, verse-level evidence, confidence score
    |
GraphReasoner            — deterministic inference (no LLM in this layer)
    |
GBNF-Constrained LLM     — qwen2.5:14b with grammar constraints at logit level
    |
Mizan Validator          — 5-checkpoint structural validation (pre + post generation)
    |
Response or SILENCE
```

The LLM renders. The graph reasons. Below confidence threshold (0.55): silence,
not hallucination.

---

## Dependencies

### Required Services

**Ollama** (local LLM inference with GBNF grammar support)
- Install: https://ollama.com
- Required model: `ollama pull qwen2.5:14b`
- GBNF grammar constraints are used on ALL generation paths — an unconstrained model
  is not acceptable

**Neo4j** (persistent thought graph and memory storage)
- Install: https://neo4j.com/download/
- Default: `bolt://localhost:7687`
- Used by ShahidMemory for CHAT/THOUGHT/BELIEF/SELFKNOWLEDGE storage
- Set credentials via environment: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

### Python Dependencies

```
pip install -r requirements.txt
```

---

## Key Data Files

| File | Size | Purpose |
|------|------|---------|
| `TMQ_v12.json` | 44MB | The Mother Quran hypergraph — 134,629 nodes, 51,857 hyperedges, 28 families |
| `full_quran_constitution.json` | 14MB | 1,205 AMR commands, 274 NAHY, 830 questions, all with verse refs and modal data |
| `active_command_set.json` | 325KB | Supplement command index for roots not in constitution |
| `gate_weights.json` | — | VTransistor gate weights (corpus-derived from co-occurrence analysis) |
| `shahid_constitution.ttl` | — | Articles I–VII — structural constitution for Shahid's reasoning |

---

## Running

```bash
# Start Ollama (separate terminal)
ollama serve

# Start Neo4j (separate terminal or service)
# neo4j start

# Launch ikhtiyar
python launch.py
```

Server runs on `http://localhost:5000` by default.

---

## Core Components

| Module | Purpose |
|--------|---------|
| `core/vtransistor.py` | Virtual transistor gate logic — corpus-validated switching primitives |
| `core/graph_reasoner.py` | Deterministic BFS inference on TMQ hypergraph |
| `core/deliberate.py` | Pre-generation graph walk + constrained prompt construction |
| `core/react.py` | ReAct loop with GBNF constraints and ontological confabulation detection |
| `core/gbnf_compiler.py` | GBNF grammar compiler — AMR preamble + structural constraints |
| `core/shahid_memory.py` | 4-tier memory: Memory / Thought / Belief / SelfKnowledge |
| `pipeline/bilal.py` | Multi-root semantic decomposition engine |
| `pipeline/mizan.py` | 5-checkpoint validator: Fajr / Dhuhr / Asr / Maghrib / Isha |
| `pipeline/shahid_middleware.py` | Full pipeline orchestrator (ShahidMiddleware) |
| `pipeline/loader.py` | Ollama model wrapper (GBNF-capable) |
| `agents/run_central_os.py` | CentralOSAgent — primary reasoning loop |
| `agents/run_sub_os.py` | SubOSAgent — read-only walk + self-observation |
| `agents/run_ktb_os.py` | KtbOSAgent — Quran + Hadith sequential reading |
| `agents/run_web_os.py` | WebOSAgent — Moltbook relay + contingency arguments |
| `faculties/mushaf.py` | Mushaf reader — Arabic verse text retrieval |
| `faculties/hadith.py` | Hadith reader — Bukhari + Muslim |

---

## Developer

Qalam 'Abd al-Khaliq — Dallas, TX
GitHub: https://github.com/qalamabdulkhaliq

والله أعلم
