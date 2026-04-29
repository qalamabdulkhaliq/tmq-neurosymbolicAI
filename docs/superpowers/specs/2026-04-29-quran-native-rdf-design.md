# القرآن كبرنامج — Quranic Native RDF DATA Division
## Design Specification — Approach 3: Full Hypergraph RDF with Named Graphs
*Output file: `ikhtiyar/QS.ttl` (Quranic Sultan / Quran Script)*

*بسم الله الرحمن الرحيم*
*Authored: 2026-04-29*
*Niyyah: That the structure Allah placed in His speech becomes the ground of a reasoning system, structurally incapable of exceeding what the DATA division contains.*

---

## 1. Problem Statement

The current HVT tape (`ikhtiyar/TMQ_hvt.json`, 77,429 frames) is a flat JSON array compiled from QAC 0.4. It correctly extracted morphological data but destroyed the relational structure in the process:

- Roots are Buckwalter string keys, not typed graph nodes
- Morphological patterns (verb forms I–X) are scalar amplifiers, not typed predicates
- Ayah-span relationships are absent — no structure connects word instances within the same verse
- TMQ v12's 51,857 hyperedges are queried separately, not unified with the word-level data
- The Mizan cannot check "does this claim cohere with other claims in the same ayah context" because no such structure exists

The grounding failure this produces: a claim can be locally plausible (correct token fluency) but globally incoherent with already-established context. The flat frame array has no mechanism to detect this class of error.

---

## 2. Objective

Build `ikhtiyar/QS.ttl` (Quranic Sultan / Quran Script) — an Arabic-native RDF graph that:

1. Uses Arabic script as canonical identifier throughout (Buckwalter as legacy literal only)
2. Encodes the QAC TAG vocabulary as typed RDF predicates (not labels)
3. Encodes Arabic verb forms (I–X) as typed predicate relationships (not scalar multipliers)
4. Wraps each ayah's word-instance triples in a named graph enabling coherence queries
5. Imports TMQ v12 hyperedges as RDF reification nodes connected to word-instances
6. Pre-classifies roots into epistemic clusters (certainty / conjecture / command / prohibition / seeking)

Replace `StaticCircuit` in `vtransistor.py` with `RDFCircuit` that queries this graph via SPARQL. All other architecture above the circuit layer remains unchanged.

---

## 3. Sources

| Source | Path | Role |
|--------|------|------|
| QAC 0.4 | `bismillah/QUS-AI HF/LHWLQIB/quranic-corpus-morphology-0.4.txt` | Primary morphological parse — 128,276 lines |
| Mushaf XML | `bismillah/mushaf/mushaf.xml` | Arabic tashkeel text, ayah metadata |
| TMQ v12 | `bismillah/TMQ_v12.json` | 51,857 hyperedges, 28 families |
| bw_arabic | `ikhtiyar/bw_arabic.py` | Buckwalter → Arabic script conversion |

---

## 4. URI Scheme

```turtle
@prefix quran: <http://quran.data/> .
@prefix root:  <http://quran.data/root/> .
@prefix word:  <http://quran.data/word/> .
@prefix ayah:  <http://quran.data/ayah/> .
@prefix seg:   <http://quran.data/seg/> .
@prefix pos:   <http://quran.data/pos/> .
@prefix form:  <http://quran.data/form/> .
@prefix morph: <http://quran.data/morph/> .
@prefix edge:  <http://quran.data/edge/> .
@prefix epi:   <http://quran.data/epistemic/> .
```

**Root URIs:** Arabic script directly.
```
root:علم   root:رحم   root:ملك   root:عبد   root:حمد
```

**Word instance URIs:** Uthmanic location (surah:ayah:word).
```
word:1:1:2   word:2:255:1   word:112:1:1
```

**Segment URIs:** Full QAC location (surah:ayah:word:seg).
```
seg:1:5:2:1   seg:2:255:1:2
```

**Ayah named graphs:**
```
ayah:1:1   ayah:2:255   ayah:112:1
```

**Buckwalter** stored as `quran:bw` literal on root nodes. Never used as a URI.

---

## 5. Triple Structure — Three Layers

### Layer 1: POS Layer (TAG column as predicate)

The QAC TAG column is a closed vocabulary of 46 tags. These become RDF predicates in the `pos:` namespace. Every word-instance segment emits one Layer 1 triple.

```turtle
GRAPH ayah:1:5 {
    root:عبد  pos:V   word:1:5:2 .
    root:عون  pos:V   word:1:5:4 .
}
```

For non-root segments (prefixes, suffixes with no ROOT field), the subject is a blank node representing the grammatical particle.

**Complete TAG → predicate mapping** (46 tags):

| TAG | Predicate | Semantic function |
|-----|-----------|-------------------|
| N | pos:N | Noun |
| V | pos:V | Verb |
| PN | pos:PN | Proper noun |
| ADJ | pos:ADJ | Adjective |
| PRON | pos:PRON | Pronoun |
| P | pos:P | Preposition |
| CONJ | pos:CONJ | Conjunction |
| DET | pos:DET | Determiner |
| ADV | pos:ADV | Adverb |
| REL | pos:REL | Relative particle |
| DEM | pos:DEM | Demonstrative |
| T | pos:T | Time adverb |
| LOC | pos:LOC | Location adverb |
| NUM | pos:NUM | Numeral |
| IMPV | pos:IMPV | Imperative verb |
| INL | pos:INL | Initial letters (Muqattaat) |
| NEG | pos:NEG | Negation particle (لا النافية) |
| PRO | pos:PRO | Prohibition particle (لا الناهية) |
| CERT | pos:CERT | Certainty particle |
| COND | pos:COND | Conditional particle |
| VOC | pos:VOC | Vocative particle |
| INTG | pos:INTG | Interrogative particle |
| EXH | pos:EXH | Exhortation |
| EXL | pos:EXL | Exclusion |
| EXP | pos:EXP | Explanation |
| EMPH | pos:EMPH | Emphasis |
| RES | pos:RES | Restriction (إلا) |
| REM | pos:REM | Resumption |
| RSLT | pos:RSLT | Result |
| ANS | pos:ANS | Answer |
| CAUS | pos:CAUS | Cause |
| PRP | pos:PRP | Purpose |
| CIRC | pos:CIRC | Circumstance |
| SUB | pos:SUB | Subordination |
| SUP | pos:SUP | Supplementation |
| COM | pos:COM | Comitative |
| EQ | pos:EQ | Equalization |
| AMD | pos:AMD | Amendment |
| AVR | pos:AVR | Aversion |
| FUT | pos:FUT | Future particle |
| INC | pos:INC | Inceptive |
| INT | pos:INT | Interpretation |
| PREV | pos:PREV | Prevention |
| RET | pos:RET | Retraction |
| SUR | pos:SUR | Surprise |
| ACC | pos:ACC | Accompaniment |

### Layer 1.5: Classical Grammar Enrichment (recovered from FEATURES)

QAC's TAG column folds four classically distinct Arabic categories into N or ADJ. The compiler recovers them from the FEATURES column and emits enriched predicates. These sit between Layer 1 (TAG) and Layer 2 (verb form) in the triple structure.

**VN — مصدر (verbal noun)**
FEATURES flag: `VN`. Emits `pos:VN` triple. Represents the abstract concept of an action, not its occurrence. Epistemic cluster: description.
```turtle
GRAPH ayah:2:2 { root:هدى  pos:VN  word:2:2:3 . }
```

**ACT_PCPL — اسم فاعل (active participle)**
FEATURES flag: `ACT|PCPL`. Emits `pos:ACT_PCPL` triple. 2,974 instances. Describes an ongoing agent.
```turtle
GRAPH ayah:1:4 { root:ملك  pos:ACT_PCPL  word:1:4:1 . }
```

**PASS_PCPL — اسم مفعول (passive participle)**
FEATURES flag: `PASS|PCPL`. Emits `pos:PASS_PCPL` triple. 551 instances. Describes a recipient of action.

**NUM — عدد (numeral)**
QAC assigns no flag. Compiler matches FORM against a compiled lexicon of ~150 Quranic numeral forms. Emits `pos:NUM` triple. Epistemic cluster: certainty (quantitative claims must be exact).

**Inna-sisters sub-predicates — ACC with SP: field**
ACC-tagged particles carry a `SP:` field identifying the specific particle. Compiler extracts SP: and emits a sub-predicate alongside the base `pos:ACC`:

| SP: value | Sub-predicate | Semantic function | Epistemic cluster |
|-----------|--------------|-------------------|-------------------|
| SP:<in~ / SP:>an~ | pos:ACC_IN | Assertion (إنَّ/أنَّ) | certainty |
| SP:laEal~ | pos:ACC_LAAL | Hope (لعلَّ) | conjecture |
| SP:layta | pos:ACC_LAYTA | Unrealizable wish (ليت) | conjecture |
| SP:lakin~ | pos:ACC_LAKIN | Concession (لكنَّ) | description |
| SP:ka>an~ | pos:ACC_KAANN | Similitude (كأنَّ) | description |

**IMPN — اسم فعل (nominal verb)**
Two instances in the Quran. Genuine classical category — nominal form with imperative force.
- `20:97` مِسَاسَ (root م-س-س) — prohibition formula لا مساس
- `69:19` هَاؤُمُ (no root) — presentation imperative. Compiler handles rootless IMPN: word-instance becomes its own subject.

**Space-in-FORM repair (37:130:3)**
Line `(37:130:3:1) <ilo yaAsiyna PN ...` — space inside FORM field causes TSV column shift. Repair rule: if TAG ∉ valid-tag-set, join columns 2+3 as FORM, shift TAG and FEATURES left. Log repair. إِلْيَاسِينَ is a proper noun (PN), the name of the Prophet Ilyas عليه السلام in Surah As-Saffat.

---

### Layer 2: Verb Form as Typed Predicate

For STEM|POS:V segments with a detected verb form (I–X), emit a second triple where the Arabic morphological pattern is the predicate.

```turtle
GRAPH ayah:1:5 {
    word:1:5:4  form:X   root:عون .
}
```

This encodes: word instance 1:5:4 (نستعين) stands in a Form X (اِسْتَفْعَلَ — seeking/requesting) relationship to root عون.

**Verb form predicate semantics:**

| Form | Predicate | وزن | Semantic function |
|------|-----------|-----|-------------------|
| I | form:I | فَعَلَ | Base action |
| II | form:II | فَعَّلَ | Causative / intensive |
| III | form:III | فَاعَلَ | Mutual / reciprocal |
| IV | form:IV | أَفْعَلَ | Transitive causative |
| V | form:V | تَفَعَّلَ | Reflexive of II |
| VI | form:VI | تَفَاعَلَ | Reflexive of III |
| VII | form:VII | اِنْفَعَلَ | Passive / resultative |
| VIII | form:VIII | اِفْتَعَلَ | Reflexive / mediopassive |
| IX | form:IX | اِفْعَلَّ | Colors / physical states |
| X | form:X | اِسْتَفْعَلَ | Seeking / deeming |

### Layer 3: Morphological Attributes

Morphological features attached to word instances as literal triples.

```turtle
word:1:5:4
    morph:person    "1P" ;
    morph:tense     "IMPF" ;
    morph:voice     "ACT" ;
    morph:mood      "IND" ;
    morph:gender    "M" ;
    morph:number    "P" ;
    morph:lemma     "{sotaEiynu" ;
    morph:bw_form   "nasotaEiynu" ;
    morph:arabic    "نَسْتَعِينُ" ;
    quran:loc       "1:5:4" .
```

---

## 6. Root Nodes

Every unique root gets a typed node in the default graph (outside named graphs):

```turtle
root:علم
    a           quran:Root ;
    quran:arabic    "علم" ;
    quran:bw        "Elm" ;
    epi:cluster     epi:certainty ;
    quran:frequency  "289"^^xsd:integer .
```

### Epistemic Clusters

Derived from classical Arabic lexicography (Lane's Lexicon + Raghib al-Isfahani's Mufradat):

| Cluster URI | Root members | Epistemic function |
|-------------|--------------|-------------------|
| epi:certainty | علم، يقن، حق، صدق، بين، شهد، رأى | Knowledge claims — high confidence required |
| epi:conjecture | ظن، حسب، خال، زعم، وهم | Conjecture claims — hedge required |
| epi:command | أمر، فرض، وجب، كتب، حكم | Imperative claims |
| epi:prohibition | نهى، حرم، منع، كره | Prohibition claims |
| epi:seeking | طلب، سأل، رجا، دعا | Seeking/supplication claims |
| epi:description | كان، صار، ليس | Copular / descriptive claims |
| epi:narrative | قال، ذكر، روى، حدّث | Reported speech |

The Mizan uses this cluster to determine what epistemic signature a claim should carry before it passes.

---

## 7. Named Graphs — Ayah Span

Each ayah is a named graph containing all Layer 1 and Layer 2 triples for words within it. Layer 3 attribute triples live in the default graph (they do not need ayah-scope).

```turtle
GRAPH ayah:2:255 {
    root:الله  pos:PN   word:2:255:1 .
    root:حي    pos:ADJ  word:2:255:2 .
    root:قوم   pos:ADJ  word:2:255:3 .
    # ... all 26 words of Ayat al-Kursi
}
```

**Coherence query pattern** — the Mizan's temporal coherence check:

```sparql
ASK {
    GRAPH ayah:?s:?v {
        ?root1 ?pos1 ?word1 .
        ?root2 ?pos2 ?word2 .
        FILTER(?word1 != ?word2)
        # check new claim root against existing roots in same ayah context
    }
}
```

---

## 8. TMQ Hyperedges as RDF Reification Nodes

TMQ v12's 51,857 hyperedges are imported into the default graph as typed reification nodes. The 10 most structurally significant families:

```turtle
edge:ILTIFAT_3to2_S2V10
    a               quran:IltifatEdge ;
    quran:tmq_id    "ILTIFAT_3to2_S2V10" ;
    quran:from_person "3P" ;
    quran:to_person   "2P" ;
    quran:connects  word:2:10:1, word:2:10:5 ;
    quran:confidence "0.85"^^xsd:float .

edge:SPEECH_ACT_AMR_2_183
    a               quran:SpeechActEdge ;
    quran:act_type  "AMR" ;
    quran:connects  word:2:183:1 ;
    quran:ayah      ayah:2:183 .
```

MORPH_ROOT edges (1,247 in TMQ) become `quran:rootRelated` links between root nodes:

```turtle
root:رحم  quran:rootRelated  root:رحيم .
root:رحم  quran:rootRelated  root:رحمن .
```

---

## 9. New Files

| File | Purpose |
|------|---------|
| `ikhtiyar/core/quran_rdf_compiler.py` | One-time compiler — reads QAC + mushaf + TMQ, emits `quran_native.ttl` |
| `ikhtiyar/quran_native.ttl` | Compiled output — Arabic-native RDF (~40MB, ~500k triples) |
| `ikhtiyar/core/rdf_circuit.py` | Replaces `StaticCircuit` — loads TTL, exposes `evaluate(roots) → PropagationResult` via SPARQL |
| `ikhtiyar/core/epistemic_clusters.py` | Root → cluster mapping, loaded at compile time and embedded in TTL |

---

## 10. Modified Files

| File | Change |
|------|--------|
| `ikhtiyar/core/vtransistor.py` | `StaticCircuit.__init__` → loads `RDFCircuit` instead of scanning HVT JSON frames. `_root_index` dict replaced by SPARQL query. `_verb_form_amplifiers` dict replaced by `form:` predicate queries. |
| `ikhtiyar/core/prooftree.py` | `proof_to_assertions()` consumes SPARQL ResultRow objects instead of HVT dict frames. Field mapping updated. |
| `ikhtiyar/engine.py` | `_init_circuit()` path updated — loads `rdf_circuit.py` instead of HVT JSON. |
| `ikhtiyar/docs/architectreport.md` | Call graph updated to reflect RDFCircuit path. |

`TMQ_hvt.json` is retired. `hvt_compiler.py` is archived as `_dead_hvt_compiler.py`.

---

## 11. RDFCircuit Interface

`rdf_circuit.py` exposes the same interface as the current `StaticCircuit` so the rest of the architecture requires no changes above the circuit layer.

```python
class RDFCircuit:
    def __init__(self, ttl_path: Path):
        self.g = ConjunctiveGraph()
        self.g.parse(str(ttl_path), format="turtle")

    def evaluate(self, roots: list[str]) -> PropagationResult:
        """
        roots: list of Arabic script root strings (e.g. ["علم", "حق"])
        Returns: PropagationResult with tier, confidence, proof_tree, ayat_refs
        """
        # Build VALUES clause from Arabic root URIs
        # Run activation SPARQL query
        # Score by epistemic cluster weights
        # Build PropagationResult from SPARQL results
        ...
```

**Primary SPARQL query (activation):**

```sparql
PREFIX root: <http://quran.data/root/>
PREFIX pos:  <http://quran.data/pos/>
PREFIX form: <http://quran.data/form/>
PREFIX epi:  <http://quran.data/epistemic/>
PREFIX quran: <http://quran.data/>

SELECT ?root ?word ?tag ?verbForm ?cluster ?ayahGraph ?loc WHERE {
    GRAPH ?ayahGraph {
        ?root ?tag ?word .
    }
    OPTIONAL { ?word ?verbForm ?root . FILTER(STRSTARTS(STR(?verbForm), STR(form:))) }
    OPTIONAL { ?root epi:cluster ?cluster }
    ?word quran:loc ?loc .
    VALUES ?root { %ROOT_URIS% }
}
ORDER BY ?ayahGraph ?loc
```

10–50ms for a 5-root query against a 500k-triple in-memory ConjunctiveGraph. Acceptable.

---

## 12. Compiler Algorithm

`quran_rdf_compiler.py` runs once. Estimated runtime: 45–90 seconds.

```
1. Parse mushaf.xml → ayah_text dict (s,v) → Arabic text
2. Parse QAC 0.4 line by line:
   a. Skip comment lines (#)
   b. Parse: LOCATION | FORM | TAG | FEATURES
   c. Extract: loc=(s,v,w,seg), tag, root_bw, verb_form, morphological attrs
   d. Space-in-FORM repair: if TAG ∉ valid-tag-set, join col2+col3 as FORM, shift TAG/FEATURES left, log repair
   e. Convert root_bw → root_arabic via bw_to_arabic()
   f. Convert FORM → Arabic script via bw_to_arabic()
   g. Emit Layer 1 triple into named graph ayah:s:v
   g1. If FEATURES contains VN flag → emit pos:VN triple (Layer 1.5)
   g2. If FEATURES contains ACT|PCPL → emit pos:ACT_PCPL triple (Layer 1.5)
   g3. If FEATURES contains PASS|PCPL → emit pos:PASS_PCPL triple (Layer 1.5)
   g4. If FORM matches numeral lexicon → emit pos:NUM triple (Layer 1.5)
   g5. If TAG=ACC and SP: field present → emit ACC sub-predicate (Layer 1.5)
   g6. If TAG=IMPN and no ROOT field → use word-instance URI as its own RDF subject
   h. If verb_form present: emit Layer 2 triple into named graph ayah:s:v
   i. Emit Layer 3 attribute triples into default graph
   i. Accumulate root nodes (deduplicated by Arabic script)
3. Emit root nodes with bw literal + frequency count
4. Run epistemic_clusters.py → assign epi:cluster to each root node
5. Parse TMQ v12 hyperedges:
   a. For each edge, create reification node in default graph
   b. Map edge member node_ids → word:s:v:w URIs via loc lookup
   c. MORPH_ROOT family → quran:rootRelated triples between root URIs
6. Serialize to quran_native.ttl (turtle format, UTF-8)
```

**Edge case handling:**
- QAC segments with no ROOT field (prefixes, particles): word instance connects to `quran:particle` blank node
- TMQ edge members that don't map to a word URI: log warning, skip (do not halt)
- Duplicate root URIs from different surahs: merge, increment frequency counter
- Muqattaat (INL tag): emit as `pos:INL` triple, mark root as `quran:isMuqattaat true`

---

## 13. Epistemic Weight in the Mizan

With the RDF graph structure, the Mizan's ontological check becomes a structured query rather than string pattern matching:

**Current Mizan (Asr check):** regex over generated text for aseity strings.

**Enhanced Mizan with RDF:** before generation is finalized, query the activated roots' epistemic clusters and verify the generation's confidence signal matches:

```python
def rdf_epistemic_check(activated_roots, generation_text):
    clusters = {query_cluster(r) for r in activated_roots}
    if epi.conjecture in clusters and confidence_signal(generation_text) == CERTAIN:
        return MizanFlag.EPISTEMIC_OVERREACH
    if epi.certainty in clusters and confidence_signal(generation_text) == UNCERTAIN:
        return MizanFlag.EPISTEMIC_UNDERREACH  # also a failure mode
    return MizanFlag.PASS
```

The distinction between ظَنَّ (conjecture, Form I from ظنن) and عَلِمَ (knowledge, Form I from علم) is now structurally available without a separate classifier — it is in the `epi:cluster` triple on the root node.

---

## 14. Build Sequence

1. **`epistemic_clusters.py`** — compile root→cluster mapping (standalone, no deps)
2. **`quran_rdf_compiler.py`** — compile `quran_native.ttl` (reads QAC + mushaf + TMQ + clusters)
3. **`rdf_circuit.py`** — implement `RDFCircuit.evaluate()` against the compiled TTL
4. **`vtransistor.py`** — swap `StaticCircuit` → `RDFCircuit`, remove `_root_index` and `_verb_form_amplifiers`
5. **`prooftree.py`** — update `proof_to_assertions()` to consume SPARQL ResultRows
6. **Verify** — run `verify_graph_reasoner.py` equivalent against new circuit
7. **Archive** — move `hvt_compiler.py` → `_dead_hvt_compiler.py`, update architectreport.md

---

## 15. Success Criteria

- [ ] `quran_native.ttl` compiles without error, ~500k triples, all 77,430 word positions covered
- [ ] Every root URI is Arabic script — no Buckwalter in any URI
- [ ] Named graph query returns all words in an ayah for any given ayah:s:v
- [ ] `RDFCircuit.evaluate(["علم", "حق"])` returns `PropagationResult` with tier=HAQQ
- [ ] `RDFCircuit.evaluate(["ظن"])` returns tier=QIYAS (conjecture cluster)
- [ ] Epistemic cluster assigned to >90% of root nodes
- [ ] TMQ ILTIFAT edges (10,863) present as typed reification nodes connected to word URIs
- [ ] Engine boots and serves chat with RDFCircuit replacing StaticCircuit
- [ ] Response quality equal or better vs. HVT baseline on 10 test queries

---

## 16. What This Does Not Change

- ShahidMiddleware pipeline — unchanged
- Mizan Fajr/Maghrib checks — unchanged
- GBNF compiler — unchanged (consumes ProofTree, not circuit internals)
- Prooftree Assertion types — unchanged (only the input source changes)
- Agent architecture — unchanged
- Constitution/beliefs/episodic TTL files — unchanged
- All agent tools and memory tiers — unchanged

The circuit is one component. The interface it exposes (`evaluate(roots) → PropagationResult`) is preserved. Everything above that interface continues without modification.

---

*والله أعلم — وبه نستعين*
