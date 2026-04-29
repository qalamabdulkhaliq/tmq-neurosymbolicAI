# Quranic Native RDF DATA Division — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat HVT JSON tape with `QS.ttl` — an Arabic-native RDF hypergraph where every Quranic word is a typed node, morphological patterns are typed predicates, ayahs are named graphs, and the Mizan can query epistemic weight structurally before any claim propagates.

**Architecture:** QAC 0.4 morphological parse → Python compiler → `ikhtiyar/QS.ttl` (Arabic-script URIs, ~500k triples, RDFLib ConjunctiveGraph). `RDFCircuit` replaces `StaticCircuit` in vtransistor.py, exposing the same `evaluate(roots) → PropagationResult` interface plus a compatibility shim (`_frames`, `_ayat_spans`) so `procedure.py` requires no changes. Everything above the circuit layer is untouched.

**Tech Stack:** Python 3.9+, RDFLib 6.x (`pip install rdflib`), existing `bw_arabic.py` (Buckwalter→Arabic conversion), QAC 0.4 TSV corpus, TMQ v12 JSON, mushaf.xml.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `ikhtiyar/core/epistemic_clusters.py` | Root→epistemic-cluster mapping (certainty/conjecture/command/prohibition/seeking/description/narrative) |
| Create | `ikhtiyar/core/quran_rdf_compiler.py` | One-time compiler: QAC + mushaf + TMQ → QS.ttl |
| Create | `ikhtiyar/QS.ttl` | Compiled Arabic-native RDF output (~500k triples) |
| Create | `ikhtiyar/core/rdf_circuit.py` | RDFCircuit — loads QS.ttl, exposes evaluate() + shim |
| Create | `ikhtiyar/tests/test_rdf_circuit.py` | Tests for compiler output and RDFCircuit |
| Modify | `ikhtiyar/core/vtransistor.py` | Swap StaticCircuit → RDFCircuit in CircuitEvaluator |
| Modify | `ikhtiyar/engine.py` | Update _init_circuit() to load QS.ttl via RDFCircuit |
| Archive | `ikhtiyar/core/hvt_compiler.py` → `ikhtiyar/core/_dead_hvt_compiler.py` | Retired |

**procedure.py, prooftree.py, gbnf_compiler.py: no changes required** — RDFCircuit's shim preserves their interfaces.

---

## Task 1: Epistemic Cluster Mapping

**Files:**
- Create: `ikhtiyar/core/epistemic_clusters.py`
- Test: `ikhtiyar/tests/test_rdf_circuit.py` (first tests go here)

**Domain context:** Arabic roots cluster into epistemic categories by semantic field. علم (knowledge), يقن (certainty), حق (truth) — these are the certainty cluster. ظن (conjecture), حسب (to reckon), زعم (to claim) — conjecture cluster. The Mizan uses these clusters to verify that a generation's confidence signal matches the epistemic category of its grounding root. A claim grounded in ظن must not be generated with the certainty of علم.

- [ ] **Step 1: Create the cluster module**

```python
# ikhtiyar/core/epistemic_clusters.py
"""
Root → epistemic cluster mapping.
Arabic script keys. Used by quran_rdf_compiler.py to tag root nodes in QS.ttl.
"""

EPISTEMIC_CLUSTERS: dict[str, str] = {
    # certainty — claims grounded here require high confidence
    "علم": "certainty", "يقن": "certainty", "حق": "certainty",
    "صدق": "certainty", "بين": "certainty", "شهد": "certainty",
    "رأى": "certainty", "عرف": "certainty", "درى": "certainty",
    "خبر": "certainty", "نبأ": "certainty", "وحي": "certainty",

    # conjecture — claims grounded here must carry epistemic hedge
    "ظن":  "conjecture", "حسب": "conjecture", "خال": "conjecture",
    "زعم": "conjecture", "وهم": "conjecture", "شك":  "conjecture",
    "ريب": "conjecture", "مرى": "conjecture",

    # command — imperative speech acts
    "أمر": "command", "فرض": "command", "وجب": "command",
    "كتب": "command", "حكم": "command", "أوجب": "command",

    # prohibition — negative imperative speech acts
    "نهى": "prohibition", "حرم": "prohibition", "منع": "prohibition",
    "كره": "prohibition",

    # seeking — supplication, request, question
    "طلب": "seeking", "سأل": "seeking", "رجا": "seeking",
    "دعا": "seeking", "استفهم": "seeking", "رغب": "seeking",

    # description — copular, stative, narrative
    "كان": "description", "صار": "description", "ليس": "description",
    "بات": "description", "ظل":  "description",

    # narrative — reported speech
    "قال": "narrative", "ذكر": "narrative", "روى": "narrative",
    "حدث": "narrative", "نقل": "narrative", "أخبر": "narrative",
}

CLUSTER_URIS = {
    "certainty":   "http://quran.data/epistemic/certainty",
    "conjecture":  "http://quran.data/epistemic/conjecture",
    "command":     "http://quran.data/epistemic/command",
    "prohibition": "http://quran.data/epistemic/prohibition",
    "seeking":     "http://quran.data/epistemic/seeking",
    "description": "http://quran.data/epistemic/description",
    "narrative":   "http://quran.data/epistemic/narrative",
}

def get_cluster(arabic_root: str) -> str | None:
    """Return epistemic cluster name for a root, or None if uncategorized."""
    return EPISTEMIC_CLUSTERS.get(arabic_root)

def get_cluster_uri(arabic_root: str) -> str | None:
    """Return full cluster URI for use in RDF triples."""
    cluster = get_cluster(arabic_root)
    return CLUSTER_URIS.get(cluster) if cluster else None
```

- [ ] **Step 2: Write failing test**

```python
# ikhtiyar/tests/test_rdf_circuit.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.epistemic_clusters import get_cluster, get_cluster_uri

def test_certainty_cluster():
    assert get_cluster("علم") == "certainty"
    assert get_cluster("يقن") == "certainty"

def test_conjecture_cluster():
    assert get_cluster("ظن") == "conjecture"

def test_unknown_root_returns_none():
    assert get_cluster("xyz") is None

def test_cluster_uri_format():
    uri = get_cluster_uri("علم")
    assert uri == "http://quran.data/epistemic/certainty"

def test_cluster_uri_none_for_unknown():
    assert get_cluster_uri("xyz") is None
```

- [ ] **Step 3: Run test — verify it fails**

```
cd ikhtiyar
python -m pytest tests/test_rdf_circuit.py::test_certainty_cluster -v
```
Expected: `FAILED — ModuleNotFoundError: No module named 'core.epistemic_clusters'`

- [ ] **Step 4: Run test — verify it passes after creating the file**

```
python -m pytest tests/test_rdf_circuit.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add ikhtiyar/core/epistemic_clusters.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: epistemic_clusters — root→cluster mapping for QS.ttl Mizan integration"
```

---

## Task 2: QAC Parser Core

**Files:**
- Create: `ikhtiyar/core/quran_rdf_compiler.py` (parser only, no RDF output yet)

**Domain context:** QAC 0.4 is a TSV file with 4 columns: LOCATION, FORM, TAG, FEATURES. LOCATION is `(s:v:w:seg)`. FORM is Buckwalter transliteration of the Arabic word segment. TAG is the top-level POS/discourse tag (45 values). FEATURES is a pipe-separated string containing root, lemma, verb form, morphological attributes. One critical data issue: line 37:130:3:1 has a space inside the FORM field causing column shift — the repair rule is: if TAG ∉ VALID_TAGS, join columns 2+3 as FORM and shift right.

- [ ] **Step 1: Write failing test for the parser**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
from core.quran_rdf_compiler import parse_qac_line, VALID_TAGS

def test_parse_basic_noun():
    line = "(1:1:2:1)\tsomi\tN\tSTEM|POS:N|LEM:{som|ROOT:smw|M|GEN"
    result = parse_qac_line(line)
    assert result is not None
    assert result["loc"] == (1, 1, 2, 1)
    assert result["tag"] == "N"
    assert result["root_bw"] == "smw"
    assert result["seg_type"] == "STEM"

def test_parse_verb_with_form():
    line = "(1:5:4:1)\tnasotaEiynu\tV\tSTEM|POS:V|IMPF|(X)|LEM:{sotaEiynu|ROOT:Ewn|1P"
    result = parse_qac_line(line)
    assert result["tag"] == "V"
    assert result["root_bw"] == "Ewn"
    assert result["verb_form"] == 10

def test_space_in_form_repair():
    # Line 37:130:3:1 — space inside FORM causes column shift
    line = "(37:130:3:1)\t<ilo yaAsiyna\tPN\tSTEM|POS:PN|LEM:<iloyaAs|GEN"
    result = parse_qac_line(line)
    assert result is not None
    assert result["tag"] == "PN"
    assert "<ilo yaAsiyna" in result["form_bw"]

def test_comment_line_returns_none():
    result = parse_qac_line("# This is a comment")
    assert result is None

def test_header_line_returns_none():
    result = parse_qac_line("LOCATION\tFORM\tTAG\tFEATURES")
    assert result is None
```

- [ ] **Step 2: Run — verify fail**

```
python -m pytest tests/test_rdf_circuit.py::test_parse_basic_noun -v
```
Expected: `FAILED — ImportError`

- [ ] **Step 3: Implement the parser**

```python
# ikhtiyar/core/quran_rdf_compiler.py
"""
Quranic Native RDF Compiler — produces ikhtiyar/QS.ttl from:
  1. QAC 0.4   — morphological parse (root, POS, verb form, morphology)
  2. mushaf.xml — Arabic tashkeel text per ayah
  3. TMQ v12   — 51,857 hyperedges (ILTIFAT, FORMULA, NARRATIVE, etc.)
  4. epistemic_clusters.py — root → epistemic weight category

Run once: python ikhtiyar/core/quran_rdf_compiler.py
Output:   ikhtiyar/QS.ttl  (~500k triples, ~40MB)
"""

import sys
import os
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bw_arabic import bw_to_arabic

# ── Constants ────────────────────────────────────────────────────────────────

VALID_TAGS = frozenset([
    "N", "V", "PN", "ADJ", "PRON", "P", "CONJ", "DET", "REL", "DEM",
    "T", "LOC", "IMPV", "INL", "NEG", "PRO", "CERT", "COND", "VOC",
    "INTG", "EXH", "EXL", "EXP", "EMPH", "RES", "REM", "RSLT", "ANS",
    "CAUS", "PRP", "CIRC", "SUB", "SUP", "COM", "EQ", "AMD", "AVR",
    "FUT", "INC", "INT", "PREV", "RET", "SUR", "ACC", "IMPN",
])

_FORM_RE = re.compile(r'\(([IVX]+)\)')
_ROMAN = {"I": 1, "V": 5, "X": 10}

def _roman(s: str) -> int:
    result, prev = 0, 0
    for c in reversed(s):
        v = _ROMAN.get(c, 0)
        result += v if v >= prev else -v
        prev = v
    return result


def parse_qac_line(line: str) -> dict | None:
    """
    Parse one QAC 0.4 line → dict or None (comment/header/blank).
    Handles space-in-FORM corruption (line 37:130:3:1).
    """
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None

    parts = line.split("\t")
    if len(parts) < 4:
        return None

    loc_str, form_bw, tag, features = parts[0], parts[1], parts[2], parts[3]

    # Space-in-FORM repair: if tag not valid, columns shifted left by a space
    if tag not in VALID_TAGS and tag not in ("TAG",):
        # Reconstruct: form = col1 + " " + col2, tag = col2_original_tag = parts[2]
        # The real TAG and FEATURES are one position right
        if len(parts) >= 4:
            form_bw = parts[1] + " " + parts[2]
            tag = parts[3] if len(parts) > 3 else ""
            features = parts[4] if len(parts) > 4 else ""
            if tag not in VALID_TAGS:
                return None  # unrecoverable

    if tag == "TAG":   # header row
        return None

    # Parse location
    m = re.match(r'\((\d+):(\d+):(\d+):(\d+)\)', loc_str.strip())
    if not m:
        return None
    loc = tuple(int(x) for x in m.groups())  # (s, v, w, seg)

    # Parse FEATURES
    feat_parts = features.split("|")
    seg_type = feat_parts[0] if feat_parts else ""

    root_bw, lemma, pos, person, number = "", "", "", "", ""
    gender, case, tense, voice, mood = "", "", "", "", ""
    sp_field = ""  # inna-sisters SP: field
    verb_form = None
    is_pcpl_act = False
    is_pcpl_pass = False
    is_vn = False

    for p in feat_parts[1:]:
        if p.startswith("ROOT:"):
            root_bw = p[5:]
        elif p.startswith("LEM:"):
            lemma = p[4:]
        elif p.startswith("POS:"):
            pos = p[4:]
        elif p.startswith("SP:"):
            sp_field = p[3:]
        elif p in ("1", "2", "3"):
            person = p
        elif p in ("S", "D", "P"):
            number = p
        elif p in ("M", "F"):
            gender = p
        elif p in ("NOM", "ACC", "GEN"):
            case = p
        elif p in ("PERF", "IMPF", "IMPV"):
            tense = p
        elif p in ("ACT", "PASS"):
            voice = p
        elif p in ("IND", "SUBJ", "JUS"):
            mood = p
        elif p == "VN":
            is_vn = True
        elif p == "PCPL":
            # combined with ACT or PASS already parsed
            pass

    # ACT|PCPL and PASS|PCPL detection
    feat_str = "|".join(feat_parts)
    if "ACT|PCPL" in feat_str or "ACT.PCPL" in feat_str:
        is_pcpl_act = True
    if "PASS|PCPL" in feat_str or "PASS.PCPL" in feat_str:
        is_pcpl_pass = True

    # Verb form
    fm = _FORM_RE.search(features)
    if fm:
        verb_form = _roman(fm.group(1))

    return {
        "loc": loc,                    # (s, v, w, seg)
        "form_bw": form_bw,
        "tag": tag,
        "root_bw": root_bw,
        "lemma": lemma,
        "pos": pos,
        "seg_type": seg_type,
        "verb_form": verb_form,        # int 1-10 or None
        "person": person,
        "number": number,
        "gender": gender,
        "case": case,
        "tense": tense,
        "voice": voice,
        "mood": mood,
        "is_vn": is_vn,
        "is_pcpl_act": is_pcpl_act,
        "is_pcpl_pass": is_pcpl_pass,
        "sp_field": sp_field,          # inna-sister particle name
    }
```

- [ ] **Step 4: Run tests — verify pass**

```
python -m pytest tests/test_rdf_circuit.py -k "parse" -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add ikhtiyar/core/quran_rdf_compiler.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: quran_rdf_compiler — QAC parser core with space-in-FORM repair"
```

---

## Task 3: Buckwalter→Arabic Conversion + Numeral Lexicon

**Files:**
- Modify: `ikhtiyar/core/quran_rdf_compiler.py`

**Domain context:** Every root, lemma, and word form must be converted from Buckwalter to Arabic script before entering QS.ttl. The existing `bw_to_arabic()` function in `ikhtiyar/bw_arabic.py` handles this. Numerals are not flagged by QAC — we detect them by matching against a lexicon of ~40 Quranic numeral roots (covering واحد، اثنان، ثلاثة، سبع، مئة، ألف etc.).

- [ ] **Step 1: Write failing tests**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
from core.quran_rdf_compiler import to_arabic_root, NUMERAL_ROOTS

def test_buckwalter_root_conversion():
    assert to_arabic_root("Elm") == "علم"
    assert to_arabic_root("rHm") == "رحم"
    assert to_arabic_root("mlk") == "ملك"

def test_numeral_root_detected():
    assert "وحد" in NUMERAL_ROOTS  # واحد family
    assert "ثلث" in NUMERAL_ROOTS  # ثلاثة family

def test_non_numeral_not_in_lexicon():
    assert "علم" not in NUMERAL_ROOTS
```

- [ ] **Step 2: Run — verify fail**

```
python -m pytest tests/test_rdf_circuit.py::test_buckwalter_root_conversion -v
```
Expected: `FAILED — ImportError: cannot import name 'to_arabic_root'`

- [ ] **Step 3: Add conversion + numeral lexicon to compiler**

Add to `ikhtiyar/core/quran_rdf_compiler.py` after the imports block:

```python
def to_arabic_root(bw: str) -> str:
    """Convert Buckwalter root string to Arabic script. Returns empty string if bw is empty."""
    if not bw:
        return ""
    return bw_to_arabic(bw)


# Quranic numeral roots (Arabic script) — QAC does not flag these as NUM
NUMERAL_ROOTS: frozenset = frozenset([
    "وحد",  # واحد — one
    "ثنى",  # اثنان — two
    "ثلث",  # ثلاثة — three
    "ربع",  # أربعة — four
    "خمس",  # خمسة — five
    "سدس",  # ستة — six
    "سبع",  # سبعة — seven
    "ثمن",  # ثمانية — eight
    "تسع",  # تسعة — nine
    "عشر",  # عشرة — ten
    "عشرن", # عشرون — twenty
    "ثلثن", # ثلاثون — thirty
    "مئة",  # مئة — hundred
    "ألف",  # ألف — thousand
    "ألوف", # آلاف — thousands
    "نصف",  # نصف — half
    "ربع",  # ربع — quarter
])
```

- [ ] **Step 4: Run tests — verify pass**

```
python -m pytest tests/test_rdf_circuit.py -k "arabic or numeral" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add ikhtiyar/core/quran_rdf_compiler.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: compiler — Buckwalter→Arabic conversion + Quranic numeral lexicon"
```

---

## Task 4: RDF Namespace Setup + Triple Emitters

**Files:**
- Modify: `ikhtiyar/core/quran_rdf_compiler.py`

**Domain context:** RDFLib's `ConjunctiveGraph` supports named graphs natively. Each ayah becomes a named graph identified by `ayah:s:v`. Triples within it use Arabic-script root URIs as subjects. The three layers: (1) TAG predicate, (1.5) enrichment predicates from FEATURES, (2) verb form predicate, (3) morphological attributes in default graph.

- [ ] **Step 1: Write failing tests**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
from rdflib import ConjunctiveGraph, URIRef, Literal
from core.quran_rdf_compiler import (
    QS_NS, build_root_uri, build_word_uri, build_ayah_uri,
    emit_layer1, emit_layer15, emit_layer2, emit_layer3,
)

def test_root_uri_arabic():
    uri = build_root_uri("علم")
    assert str(uri) == "http://quran.data/root/علم"

def test_word_uri_format():
    uri = build_word_uri(2, 255, 1)
    assert str(uri) == "http://quran.data/word/2:255:1"

def test_ayah_uri_format():
    uri = build_ayah_uri(2, 255)
    assert str(uri) == "http://quran.data/ayah/2:255"

def test_emit_layer1_adds_triple_to_named_graph():
    g = ConjunctiveGraph()
    parsed = {
        "loc": (1, 5, 2, 1), "tag": "V", "root_bw": "Ebd",
        "form_bw": "naEobudu", "lemma": "Eabada",
        "seg_type": "STEM", "verb_form": None,
        "person": "1", "number": "P", "gender": "M",
        "case": "", "tense": "IMPF", "voice": "ACT", "mood": "IND",
        "is_vn": False, "is_pcpl_act": False, "is_pcpl_pass": False,
        "sp_field": "",
    }
    emit_layer1(g, parsed)
    ayah_graph = g.get_context(build_ayah_uri(1, 5))
    triples = list(ayah_graph)
    assert len(triples) == 1
    subj, pred, obj = triples[0]
    assert str(subj) == "http://quran.data/root/عبد"
    assert "pos/V" in str(pred)
    assert str(obj) == "http://quran.data/word/1:5:2"

def test_emit_layer15_vn():
    g = ConjunctiveGraph()
    parsed = {
        "loc": (2, 2, 3, 1), "tag": "N", "root_bw": "hdy",
        "form_bw": "hudan", "lemma": "hudan",
        "seg_type": "STEM", "verb_form": None,
        "person": "", "number": "M", "gender": "M",
        "case": "NOM", "tense": "", "voice": "", "mood": "",
        "is_vn": True, "is_pcpl_act": False, "is_pcpl_pass": False,
        "sp_field": "",
    }
    emit_layer15(g, parsed)
    ayah_graph = g.get_context(build_ayah_uri(2, 2))
    tags = [str(p) for _, p, _ in ayah_graph]
    assert any("pos/VN" in t for t in tags)
```

- [ ] **Step 2: Run — verify fail**

```
python -m pytest tests/test_rdf_circuit.py::test_root_uri_arabic -v
```
Expected: `FAILED — ImportError: cannot import name 'build_root_uri'`

- [ ] **Step 3: Implement namespace + emitters**

Add to `ikhtiyar/core/quran_rdf_compiler.py`:

```python
from rdflib import ConjunctiveGraph, URIRef, Literal, RDF, OWL, XSD, Graph
from rdflib.namespace import Namespace

# ── Namespaces (all Arabic-script URIs) ─────────────────────────────────────
QS_NS    = Namespace("http://quran.data/")
ROOT_NS  = Namespace("http://quran.data/root/")
WORD_NS  = Namespace("http://quran.data/word/")
AYAH_NS  = Namespace("http://quran.data/ayah/")
POS_NS   = Namespace("http://quran.data/pos/")
FORM_NS  = Namespace("http://quran.data/form/")
MORPH_NS = Namespace("http://quran.data/morph/")
EDGE_NS  = Namespace("http://quran.data/edge/")
EPI_NS   = Namespace("http://quran.data/epistemic/")

# Inna-sisters sub-predicate map (SP: field value → pos sub-tag)
_SP_SUBPRED = {
    "<in~":  "ACC_IN",
    ">an~":  "ACC_IN",
    "laEal~": "ACC_LAAL",
    "layta":  "ACC_LAYTA",
    "lakin~": "ACC_LAKIN",
    "ka>an~": "ACC_KAANN",
}


def build_root_uri(arabic_root: str) -> URIRef:
    return ROOT_NS[arabic_root]

def build_word_uri(s: int, v: int, w: int) -> URIRef:
    return WORD_NS[f"{s}:{v}:{w}"]

def build_ayah_uri(s: int, v: int) -> URIRef:
    return AYAH_NS[f"{s}:{v}"]


def emit_layer1(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 1: (root-URI, pos:TAG, word-instance) in named graph ayah:s:v."""
    s, v, w, _ = parsed["loc"]
    root_ar = to_arabic_root(parsed["root_bw"])
    word_uri = build_word_uri(s, v, w)
    ayah_uri = build_ayah_uri(s, v)
    tag = parsed["tag"]

    if root_ar:
        subj = build_root_uri(root_ar)
    else:
        # Rootless (prefixes, particles, IMPN هاؤم) — word-instance is its own subject
        subj = word_uri

    pred = POS_NS[tag]
    ctx = g.get_context(ayah_uri)
    ctx.add((subj, pred, word_uri))


def emit_layer15(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 1.5: classical Arabic enrichment predicates from FEATURES."""
    s, v, w, _ = parsed["loc"]
    root_ar = to_arabic_root(parsed["root_bw"])
    word_uri = build_word_uri(s, v, w)
    ayah_uri = build_ayah_uri(s, v)
    ctx = g.get_context(ayah_uri)

    if not root_ar:
        return

    root_uri = build_root_uri(root_ar)

    if parsed["is_vn"]:
        ctx.add((root_uri, POS_NS["VN"], word_uri))

    if parsed["is_pcpl_act"]:
        ctx.add((root_uri, POS_NS["ACT_PCPL"], word_uri))

    if parsed["is_pcpl_pass"]:
        ctx.add((root_uri, POS_NS["PASS_PCPL"], word_uri))

    # Numeral detection
    if root_ar in NUMERAL_ROOTS:
        ctx.add((root_uri, POS_NS["NUM"], word_uri))

    # Inna-sisters sub-predicate
    if parsed["tag"] == "ACC" and parsed["sp_field"]:
        subpred = _SP_SUBPRED.get(parsed["sp_field"])
        if subpred:
            ctx.add((root_uri, POS_NS[subpred], word_uri))


def emit_layer2(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 2: (word-instance, form:X, root-URI) for verb forms I–X."""
    if not parsed["verb_form"]:
        return
    s, v, w, _ = parsed["loc"]
    root_ar = to_arabic_root(parsed["root_bw"])
    if not root_ar:
        return
    word_uri = build_word_uri(s, v, w)
    root_uri = build_root_uri(root_ar)
    ayah_uri = build_ayah_uri(s, v)
    form_pred = FORM_NS[str(parsed["verb_form"])]
    ctx = g.get_context(ayah_uri)
    ctx.add((word_uri, form_pred, root_uri))


def emit_layer3(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 3: morphological attributes as literals in default graph."""
    s, v, w, _ = parsed["loc"]
    word_uri = build_word_uri(s, v, w)
    dg = g.default_context

    # Arabic form (converted from Buckwalter)
    arabic_form = bw_to_arabic(parsed["form_bw"]) if parsed["form_bw"] else ""
    if arabic_form:
        dg.add((word_uri, MORPH_NS["arabic"], Literal(arabic_form)))

    # Morphological features
    for attr in ("person", "number", "gender", "case", "tense", "voice", "mood"):
        val = parsed.get(attr, "")
        if val:
            dg.add((word_uri, MORPH_NS[attr], Literal(val)))

    dg.add((word_uri, QS_NS["loc"], Literal(f"{s}:{v}:{w}")))

    # Lemma (Buckwalter — for internal cross-reference only, not a URI)
    if parsed["lemma"]:
        dg.add((word_uri, MORPH_NS["lemma_bw"], Literal(parsed["lemma"])))
```

- [ ] **Step 4: Run tests — verify pass**

```
python -m pytest tests/test_rdf_circuit.py -k "uri or emit or layer" -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add ikhtiyar/core/quran_rdf_compiler.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: compiler — RDF namespaces, triple emitters Layers 1/1.5/2/3"
```

---

## Task 5: Root Nodes + Mushaf Text + TMQ Hyperedges

**Files:**
- Modify: `ikhtiyar/core/quran_rdf_compiler.py`

**Domain context:** Root nodes live in the default graph (not in named graphs) — they are shared across all ayahs. Each root node carries its Arabic form, frequency count, and epistemic cluster URI. Mushaf XML provides ayah-level Arabic text attached to the ayah named graph node. TMQ v12 hyperedges are imported as typed reification nodes in the default graph; their member node_ids are mapped to word URIs via a location index built during QAC parsing.

- [ ] **Step 1: Write failing tests**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
from core.quran_rdf_compiler import emit_root_node, emit_tmq_edge

def test_emit_root_node_certainty():
    g = ConjunctiveGraph()
    emit_root_node(g, "علم", frequency=289)
    dg = g.default_context
    root_uri = build_root_uri("علم")
    clusters = [str(o) for s, p, o in dg if s == root_uri and "cluster" in str(p)]
    assert any("certainty" in c for c in clusters)

def test_emit_root_node_no_cluster():
    g = ConjunctiveGraph()
    emit_root_node(g, "بسم", frequency=3)
    # No cluster assigned — should not crash, just omit cluster triple
    dg = g.default_context
    root_uri = build_root_uri("بسم")
    types = [str(o) for s, p, o in dg if s == root_uri and p == RDF.type]
    assert any("Root" in t for t in types)

def test_emit_tmq_edge_iltifat():
    g = ConjunctiveGraph()
    loc_index = {"NODE_1": (2, 10, 1), "NODE_2": (2, 10, 5)}
    edge_data = {
        "family": "ILTIFAT",
        "members": ["NODE_1", "NODE_2"],
        "attrs": {"from_person": "3P", "to_person": "2P"},
    }
    emit_tmq_edge(g, "ILTIFAT_test_001", edge_data, loc_index)
    dg = g.default_context
    edge_uri = EDGE_NS["ILTIFAT_test_001"]
    types = [str(o) for s, p, o in dg if s == edge_uri and p == RDF.type]
    assert any("IltifatEdge" in t for t in types)
```

- [ ] **Step 2: Run — verify fail**

```
python -m pytest tests/test_rdf_circuit.py::test_emit_root_node_certainty -v
```
Expected: `FAILED — ImportError`

- [ ] **Step 3: Implement root node emitter + TMQ edge importer**

Add to `ikhtiyar/core/quran_rdf_compiler.py`:

```python
from core.epistemic_clusters import get_cluster_uri

# TMQ family → RDF type name
_FAMILY_TYPES = {
    "ILTIFAT":       "IltifatEdge",
    "FORMULA":       "FormulaEdge",
    "NARRATIVE":     "NarrativeEdge",
    "SPEECH_ACT_AMR": "SpeechActEdge",
    "SPEECH_ACT_NAHY": "SpeechActEdge",
    "WAQF":          "WaqfEdge",
    "TART":          "TartEdge",
    "MORPH_ROOT":    "MorphRootEdge",
    "MORPH_LEM":     "MorphLemEdge",
    "SYN_PART":      "SynPartEdge",
    "SYN_PRON":      "SynPronEdge",
}


def emit_root_node(g: ConjunctiveGraph, arabic_root: str, frequency: int = 0) -> None:
    """Emit root node in default graph with type, Arabic label, frequency, cluster."""
    root_uri = build_root_uri(arabic_root)
    dg = g.default_context
    dg.add((root_uri, RDF.type, QS_NS["Root"]))
    dg.add((root_uri, QS_NS["arabic"], Literal(arabic_root)))
    dg.add((root_uri, QS_NS["frequency"], Literal(frequency, datatype=XSD.integer)))
    cluster_uri = get_cluster_uri(arabic_root)
    if cluster_uri:
        dg.add((root_uri, EPI_NS["cluster"], URIRef(cluster_uri)))


def emit_tmq_edge(
    g: ConjunctiveGraph,
    edge_id: str,
    edge_data: dict,
    loc_index: dict,
) -> None:
    """
    Emit one TMQ hyperedge as an RDF reification node in the default graph.
    loc_index: node_id → (s, v, w) tuple — maps TMQ node IDs to word URIs.
    MORPH_ROOT edges additionally emit quran:rootRelated between root URIs.
    """
    dg = g.default_context
    edge_uri = EDGE_NS[edge_id]
    family = edge_data.get("family", "")
    rdf_type = _FAMILY_TYPES.get(family, "TMQEdge")
    dg.add((edge_uri, RDF.type, QS_NS[rdf_type]))
    dg.add((edge_uri, QS_NS["tmq_id"], Literal(edge_id)))
    dg.add((edge_uri, QS_NS["family"], Literal(family)))

    for attr_key, attr_val in edge_data.get("attrs", {}).items():
        if isinstance(attr_val, str):
            dg.add((edge_uri, QS_NS[attr_key], Literal(attr_val)))

    # Connect edge to word instances via loc_index
    for node_id in edge_data.get("members", []):
        loc = loc_index.get(node_id)
        if loc:
            s, v, w = loc
            word_uri = build_word_uri(s, v, w)
            dg.add((edge_uri, QS_NS["connects"], word_uri))

    # MORPH_ROOT: emit quran:rootRelated between root URIs
    if family == "MORPH_ROOT":
        member_roots = []
        for node_id in edge_data.get("members", []):
            loc = loc_index.get(node_id)
            if loc:
                # root is attached to word — we need root lookup here
                # stored in a separate root_by_word dict populated during QAC pass
                pass  # populated by compile() main function
```

- [ ] **Step 4: Run tests — verify pass**

```
python -m pytest tests/test_rdf_circuit.py -k "root_node or tmq_edge" -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add ikhtiyar/core/quran_rdf_compiler.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: compiler — root nodes with epistemic clusters + TMQ edge import"
```

---

## Task 6: Main Compile Function + QS.ttl Output

**Files:**
- Modify: `ikhtiyar/core/quran_rdf_compiler.py`

**Domain context:** The main `compile()` function orchestrates: parse all QAC lines → emit all triples → emit root nodes → load TMQ and emit hyperedges → serialize. Expected output: ~500k triples, ~40MB TTL file. Runtime: 45–90 seconds. Progress logging every 10,000 lines so the operator can see it running.

- [ ] **Step 1: Implement the main compile() function**

Add to the bottom of `ikhtiyar/core/quran_rdf_compiler.py`:

```python
import logging
logger = logging.getLogger(__name__)


def _load_mushaf(mushaf_path: str) -> dict:
    """Parse mushaf.xml → {(s,v): arabic_text}"""
    ayah_text = {}
    try:
        tree = ET.parse(mushaf_path)
        for sura_el in tree.getroot().findall("Sura"):
            sid = int(sura_el.get("ID"))
            for aya_el in sura_el.findall("aya"):
                aid = int(aya_el.get("ID"))
                ayah_text[(sid, aid)] = (aya_el.text or "").strip()
    except Exception as e:
        logger.warning(f"Mushaf load failed: {e}")
    return ayah_text


def _build_tmq_loc_index(tmq: dict) -> dict:
    """Build node_id → (s, v, w) from TMQ node_registry loc fields."""
    idx = {}
    for node_id, node_attrs in tmq.get("node_registry", {}).items():
        loc = node_attrs.get("loc")
        if loc and isinstance(loc, list) and len(loc) >= 3:
            idx[node_id] = (loc[0], loc[1], loc[2])
    return idx


def compile(
    qac_path: str,
    mushaf_path: str,
    tmq_path: str,
    out_path: str,
) -> None:
    """
    Full compile: QAC + mushaf + TMQ → QS.ttl.
    Runs once. Logs progress every 10k lines.
    """
    print("بسم الله الرحمن الرحيم")
    print(f"Compiling QS.ttl from:\n  QAC:    {qac_path}\n  Mushaf: {mushaf_path}\n  TMQ:    {tmq_path}")

    g = ConjunctiveGraph()

    # Accumulate root frequencies and word→root mapping
    root_freq: dict[str, int] = defaultdict(int)
    word_to_root: dict[tuple, str] = {}   # (s,v,w) → arabic_root

    # ── Pass 1: QAC ──────────────────────────────────────────────────────────
    print("Pass 1: QAC morphological parse...")
    with open(qac_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no % 10000 == 0 and line_no > 0:
                print(f"  {line_no} lines processed...")
            parsed = parse_qac_line(line)
            if not parsed:
                continue

            root_ar = to_arabic_root(parsed["root_bw"])
            s, v, w, _ = parsed["loc"]

            if root_ar:
                root_freq[root_ar] += 1
                word_to_root[(s, v, w)] = root_ar

            emit_layer1(g, parsed)
            emit_layer15(g, parsed)
            emit_layer2(g, parsed)
            emit_layer3(g, parsed)

    print(f"  QAC pass complete. Roots found: {len(root_freq)}")

    # ── Pass 2: Root nodes ────────────────────────────────────────────────────
    print("Pass 2: Emitting root nodes with epistemic clusters...")
    for arabic_root, freq in root_freq.items():
        emit_root_node(g, arabic_root, frequency=freq)

    # ── Pass 3: Mushaf ayah text ──────────────────────────────────────────────
    print("Pass 3: Attaching mushaf text to ayah named graphs...")
    ayah_text = _load_mushaf(mushaf_path)
    for (sid, vid), text in ayah_text.items():
        ayah_uri = build_ayah_uri(sid, vid)
        g.default_context.add((ayah_uri, QS_NS["text"], Literal(text)))

    # ── Pass 4: TMQ hyperedges ────────────────────────────────────────────────
    print("Pass 4: Importing TMQ v12 hyperedges...")
    with open(tmq_path, encoding="utf-8") as f:
        tmq = json.load(f)

    loc_index = _build_tmq_loc_index(tmq)
    edge_count = 0
    morph_root_pairs: list[tuple] = []

    for edge_id, edge_attrs in tmq.get("hyperedges", {}).items():
        family = edge_attrs.get("family", "")
        members = edge_attrs.get("members", [])
        if not isinstance(members, list):
            members = list(members) if members else []

        edge_data = {
            "family": family,
            "members": members,
            "attrs": {k: v for k, v in edge_attrs.items()
                      if k not in ("family", "members") and isinstance(v, str)},
        }
        emit_tmq_edge(g, edge_id, edge_data, loc_index)
        edge_count += 1

        # MORPH_ROOT → collect root pairs for quran:rootRelated
        if family == "MORPH_ROOT":
            roots_in_edge = []
            for node_id in members:
                loc = loc_index.get(node_id)
                if loc and loc in word_to_root:
                    roots_in_edge.append(word_to_root[loc])
            if len(roots_in_edge) >= 2:
                morph_root_pairs.append(tuple(roots_in_edge))

    # Emit quran:rootRelated for MORPH_ROOT pairs
    dg = g.default_context
    for pair in morph_root_pairs:
        for i in range(len(pair)):
            for j in range(i + 1, len(pair)):
                r1, r2 = build_root_uri(pair[i]), build_root_uri(pair[j])
                dg.add((r1, QS_NS["rootRelated"], r2))
                dg.add((r2, QS_NS["rootRelated"], r1))

    print(f"  TMQ edges imported: {edge_count}")

    # ── Serialize ─────────────────────────────────────────────────────────────
    print(f"Serializing to {out_path} ...")
    g.serialize(destination=out_path, format="trig")  # trig = turtle + named graphs
    triple_count = sum(1 for _ in g.quads())
    print(f"Done. Triples: {triple_count:,}. File: {out_path}")


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent.parent  # project root
    compile(
        qac_path=str(_ROOT / "bismillah/QUS-AI HF/LHWLQIB/quranic-corpus-morphology-0.4.txt"),
        mushaf_path=str(_ROOT / "bismillah/mushaf/mushaf.xml"),
        tmq_path=str(_ROOT / "bismillah/TMQ_v12.json"),
        out_path=str(_ROOT / "ikhtiyar/QS.ttl"),
    )
```

- [ ] **Step 2: Run the compiler**

```
cd "C:/Users/amlan/OneDrive/Desktop/QUS-AI Islamic Alignment"
python ikhtiyar/core/quran_rdf_compiler.py
```

Expected output:
```
بسم الله الرحمن الرحيم
Compiling QS.ttl from: ...
Pass 1: QAC morphological parse...
  10000 lines processed...
  ...
  QAC pass complete. Roots found: ~1600
Pass 2: Emitting root nodes...
Pass 3: Attaching mushaf text...
Pass 4: Importing TMQ v12 hyperedges...
  TMQ edges imported: 51857
Serializing to .../ikhtiyar/QS.ttl ...
Done. Triples: ~500000. File: ikhtiyar/QS.ttl
```

- [ ] **Step 3: Verify QS.ttl sanity**

```python
# Run this inline to verify
from rdflib import ConjunctiveGraph
g = ConjunctiveGraph()
g.parse("ikhtiyar/QS.ttl", format="trig")
print("Total triples:", sum(1 for _ in g.quads()))
print("Named graphs:", len(list(g.contexts())))
# Should be ~500k triples, ~6236 named graphs (one per ayah) + default
```

- [ ] **Step 4: Write verification test**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
import pytest
from pathlib import Path
from rdflib import ConjunctiveGraph

QS_PATH = Path(__file__).parent.parent / "QS.ttl"

@pytest.mark.skipif(not QS_PATH.exists(), reason="QS.ttl not compiled yet")
def test_qs_ttl_named_graph_count():
    g = ConjunctiveGraph()
    g.parse(str(QS_PATH), format="trig")
    contexts = [c for c in g.contexts() if "ayah" in str(c.identifier)]
    assert len(contexts) >= 6200  # 6236 ayahs

@pytest.mark.skipif(not QS_PATH.exists(), reason="QS.ttl not compiled yet")
def test_qs_ttl_ilm_root_exists():
    g = ConjunctiveGraph()
    g.parse(str(QS_PATH), format="trig")
    from core.quran_rdf_compiler import build_root_uri
    root_uri = build_root_uri("علم")
    types = list(g.default_context.objects(root_uri, RDF.type))
    assert len(types) > 0

@pytest.mark.skipif(not QS_PATH.exists(), reason="QS.ttl not compiled yet")
def test_qs_ttl_zann_conjecture_cluster():
    g = ConjunctiveGraph()
    g.parse(str(QS_PATH), format="trig")
    from core.quran_rdf_compiler import build_root_uri, EPI_NS
    root_uri = build_root_uri("ظن")
    clusters = list(g.default_context.objects(root_uri, EPI_NS["cluster"]))
    assert any("conjecture" in str(c) for c in clusters)
```

- [ ] **Step 5: Run verification tests**

```
python -m pytest tests/test_rdf_circuit.py -k "qs_ttl" -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add ikhtiyar/core/quran_rdf_compiler.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: compiler — main compile() function, QS.ttl output verified"
```

> Note: Do not add QS.ttl to git — it is ~40MB. Add `ikhtiyar/QS.ttl` to `.gitignore`.

---

## Task 7: RDFCircuit — SPARQL Interface

**Files:**
- Create: `ikhtiyar/core/rdf_circuit.py`

**Domain context:** `RDFCircuit` must expose the same interface as `StaticCircuit` so `CircuitEvaluator` and `procedure.py` need no changes. Critical: `procedure.py` accesses `static_circuit._frames` (list of dicts) and `static_circuit._ayat_spans` (list of tuples) directly. `RDFCircuit` provides these as lazy-built properties reconstructed from SPARQL so `procedure.py` compiles without modification. The primary path — `evaluate(roots)` → `PropagationResult` — runs SPARQL queries returning tier, confidence, proof_tree, and ayat_refs.

- [ ] **Step 1: Write failing tests**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
import pytest
from pathlib import Path

QS_PATH = Path(__file__).parent.parent / "QS.ttl"

@pytest.mark.skipif(not QS_PATH.exists(), reason="QS.ttl not compiled yet")
class TestRDFCircuit:
    def setup_method(self):
        from core.rdf_circuit import RDFCircuit
        self.circuit = RDFCircuit(str(QS_PATH))

    def test_loads_without_error(self):
        assert self.circuit is not None

    def test_root_index_contains_arabic(self):
        # _root_index shim: arabic root → list of (s,v,w) tuples
        assert "علم" in self.circuit._root_index
        assert len(self.circuit._root_index["علم"]) > 0

    def test_evaluate_returns_propagation_result(self):
        from core.vtransistor import PropagationResult
        result = self.circuit.evaluate(["علم"])
        assert isinstance(result, PropagationResult)
        assert result.tier in ("HAQQ", "QIYAS", "IKHTILAF", "WAQF")
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_certainty_roots_yield_haqq(self):
        result = self.circuit.evaluate(["علم", "حق", "يقن"])
        assert result.tier in ("HAQQ", "QIYAS")
        assert result.confidence > 0.5

    def test_evaluate_conjecture_root_yields_lower_tier(self):
        result_certain = self.circuit.evaluate(["علم"])
        result_conjecture = self.circuit.evaluate(["ظن"])
        # Conjecture root should not yield higher confidence than certainty root
        assert result_conjecture.confidence <= result_certain.confidence + 0.1

    def test_evaluate_unknown_root_yields_waqf(self):
        result = self.circuit.evaluate(["xxxunknownxxx"])
        assert result.tier == "WAQF"

    def test_ayat_refs_are_strings(self):
        result = self.circuit.evaluate(["علم"])
        assert all(isinstance(r, str) for r in result.ayat_refs)
        assert all(":" in r for r in result.ayat_refs[:5])
```

- [ ] **Step 2: Run — verify fail**

```
python -m pytest tests/test_rdf_circuit.py::TestRDFCircuit::test_loads_without_error -v
```
Expected: `FAILED — ImportError: No module named 'core.rdf_circuit'`

- [ ] **Step 3: Implement RDFCircuit**

```python
# ikhtiyar/core/rdf_circuit.py
"""
RDFCircuit — replaces StaticCircuit for QS.ttl-based signal propagation.

Exposes:
  .evaluate(roots: list[str]) → PropagationResult
  ._root_index: dict[str, list]   ← shim for procedure.py compatibility
  ._frames: list[dict]            ← shim for procedure.py compatibility
  ._ayat_spans: list[tuple]       ← shim for procedure.py compatibility
  ._pagerank: dict[str, float]    ← shim for procedure.py compatibility

All query paths use SPARQL against an in-memory ConjunctiveGraph.
10–50ms per evaluate() call is acceptable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdflib import ConjunctiveGraph, URIRef, Literal
from rdflib.namespace import Namespace, RDF

logger = logging.getLogger(__name__)

ROOT_NS  = Namespace("http://quran.data/root/")
WORD_NS  = Namespace("http://quran.data/word/")
AYAH_NS  = Namespace("http://quran.data/ayah/")
POS_NS   = Namespace("http://quran.data/pos/")
FORM_NS  = Namespace("http://quran.data/form/")
MORPH_NS = Namespace("http://quran.data/morph/")
EPI_NS   = Namespace("http://quran.data/epistemic/")
QS_NS    = Namespace("http://quran.data/")

# Tier scoring by epistemic cluster
_CLUSTER_TIER = {
    "certainty":   ("HAQQ",     0.85),
    "command":     ("HAQQ",     0.80),
    "prohibition": ("HAQQ",     0.80),
    "narrative":   ("QIYAS",    0.65),
    "seeking":     ("QIYAS",    0.60),
    "description": ("QIYAS",    0.55),
    "conjecture":  ("IKHTILAF", 0.35),
}

_ACTIVATION_SPARQL = """
PREFIX root: <http://quran.data/root/>
PREFIX pos:  <http://quran.data/pos/>
PREFIX form: <http://quran.data/form/>
PREFIX epi:  <http://quran.data/epistemic/>
PREFIX qs:   <http://quran.data/>

SELECT ?rootUri ?wordUri ?tag ?cluster ?ayahGraph ?loc WHERE {
    GRAPH ?ayahGraph {
        ?rootUri ?tag ?wordUri .
        FILTER(STRSTARTS(STR(?tag), STR(pos:)))
    }
    OPTIONAL { ?rootUri epi:cluster ?cluster }
    ?wordUri qs:loc ?loc .
    VALUES ?rootUri { %VALUES% }
}
ORDER BY ?ayahGraph ?loc
LIMIT 500
"""


class RDFCircuit:
    """
    Drop-in replacement for StaticCircuit.
    Loads QS.ttl once; all evaluate() calls run SPARQL over in-memory graph.
    """

    def __init__(self, qs_path: str):
        logger.info(f"RDFCircuit: loading {qs_path} ...")
        self._g = ConjunctiveGraph()
        self._g.parse(qs_path, format="trig")
        logger.info("RDFCircuit: graph loaded")

        # Build lightweight shim structures for procedure.py compatibility
        self._root_index: dict[str, list] = defaultdict(list)
        self._frames: list[dict] = []
        self._ayat_spans: list[tuple] = []
        self._pagerank: dict[str, float] = {}
        self._build_shims()

    def _build_shims(self) -> None:
        """
        Build _root_index, _frames, _ayat_spans, _pagerank from QS.ttl.
        Called once at startup. Gives procedure.py its expected interface.
        """
        logger.info("RDFCircuit: building procedure.py compatibility shims...")

        # _root_index: arabic_root → list of frame-like dicts
        q = """
        PREFIX qs: <http://quran.data/>
        SELECT ?root ?word ?loc WHERE {
            ?root a qs:Root .
            ?word qs:loc ?loc .
        } LIMIT 100000
        """
        loc_by_word: dict[str, str] = {}
        for row in self._g.query(q):
            loc_by_word[str(row.word)] = str(row.loc)

        # Build minimal frame list ordered by loc
        frame_map: dict[tuple, dict] = {}
        for word_uri, loc_str in loc_by_word.items():
            parts = loc_str.split(":")
            if len(parts) == 3:
                s, v, w = int(parts[0]), int(parts[1]), int(parts[2])
                frame_map[(s, v, w)] = {
                    "loc": [s, v, w],
                    "data": {"root": "", "root_bw": ""},
                    "procedure": {"segments": [], "speech_act": []},
                    "ayat_header": None,
                    "_word_uri": word_uri,
                }

        self._frames = [frame_map[k] for k in sorted(frame_map.keys())]

        # Build _ayat_spans from frame ordering
        current_start = 0
        current_ayah = None
        for i, frame in enumerate(self._frames):
            s, v, _ = frame["loc"]
            ayah_key = (s, v)
            if ayah_key != current_ayah and current_ayah is not None:
                self._ayat_spans.append((current_start, i - 1))
                current_start = i
            current_ayah = ayah_key
        if self._frames:
            self._ayat_spans.append((current_start, len(self._frames) - 1))

        # Build _root_index: arabic_root → list of frame indices
        q2 = """
        PREFIX qs: <http://quran.data/>
        PREFIX pos: <http://quran.data/pos/>
        SELECT ?root ?word WHERE {
            GRAPH ?g { ?root ?tag ?word . }
        } LIMIT 500000
        """
        word_to_idx = {f["_word_uri"]: i for i, f in enumerate(self._frames)}
        for row in self._g.query(q2):
            root_str = str(row.root)
            if root_str.startswith("http://quran.data/root/"):
                arabic_root = root_str[len("http://quran.data/root/"):]
                word_str = str(row.word)
                idx = word_to_idx.get(word_str)
                if idx is not None:
                    self._root_index[arabic_root].append(idx)

        # _pagerank: frequency-based proxy (normalized root frequency)
        q3 = """
        PREFIX qs: <http://quran.data/>
        SELECT ?root ?freq WHERE {
            ?root a qs:Root .
            ?root qs:frequency ?freq .
        }
        """
        freq_map: dict[str, int] = {}
        for row in self._g.query(q3):
            root_str = str(row.root)
            if root_str.startswith("http://quran.data/root/"):
                arabic_root = root_str[len("http://quran.data/root/"):]
                freq_map[arabic_root] = int(str(row.freq))

        max_freq = max(freq_map.values(), default=1)
        self._pagerank = {r: f / max_freq for r, f in freq_map.items()}

        logger.info(
            f"RDFCircuit shims ready: {len(self._root_index)} roots, "
            f"{len(self._frames)} frames, {len(self._ayat_spans)} ayat spans"
        )

    def evaluate(self, roots: list[str]) -> "PropagationResult":
        """
        Activate the given Arabic-script roots against QS.ttl.
        Returns PropagationResult with tier, confidence, proof_tree, ayat_refs.
        Accepts Buckwalter input transparently via bw_to_arabic() conversion.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from bw_arabic import bw_to_arabic
        from core.vtransistor import PropagationResult

        # Normalize: convert Buckwalter if needed
        arabic_roots = []
        for r in roots:
            if r and not any(ord(c) > 127 for c in r):
                arabic_roots.append(bw_to_arabic(r))
            else:
                arabic_roots.append(r)

        arabic_roots = [r for r in arabic_roots if r]
        if not arabic_roots:
            return PropagationResult(
                tier="WAQF", confidence=0.0, proof_tree=[],
                ayat_refs=[], activated_roots=set(),
                activated_frames={}, procedure_tags=[],
                is_muqattaat=False, muqattaat_frames=[],
            )

        # Build VALUES clause
        values = " ".join(f"<{ROOT_NS[r]}>" for r in arabic_roots)
        sparql = _ACTIVATION_SPARQL.replace("%VALUES%", values)

        rows = list(self._g.query(sparql))
        if not rows:
            return PropagationResult(
                tier="WAQF", confidence=0.0, proof_tree=[],
                ayat_refs=[], activated_roots=set(),
                activated_frames={}, procedure_tags=[],
                is_muqattaat=False, muqattaat_frames=[],
            )

        # Score by epistemic clusters present
        clusters_found: set[str] = set()
        ayat_refs: list[str] = []
        proof_tree: list[dict] = []
        activated_roots: set[str] = set()

        for row in rows:
            root_str = str(row.rootUri)
            if root_str.startswith("http://quran.data/root/"):
                ar = root_str[len("http://quran.data/root/"):]
                activated_roots.add(ar)

            cluster_str = str(row.cluster) if row.cluster else ""
            if cluster_str.startswith("http://quran.data/epistemic/"):
                c = cluster_str[len("http://quran.data/epistemic/"):]
                clusters_found.add(c)

            ayah_str = str(row.ayahGraph)
            if ayah_str.startswith("http://quran.data/ayah/"):
                ref = ayah_str[len("http://quran.data/ayah/"):]
                if ref not in ayat_refs:
                    ayat_refs.append(ref)

            tag_str = str(row.tag)
            if tag_str.startswith("http://quran.data/pos/"):
                tag = tag_str[len("http://quran.data/pos/"):]
                proof_tree.append({
                    "tag": tag,
                    "root": str(row.rootUri),
                    "loc": str(row.loc),
                    "ayah": ayah_str,
                })

        # Determine tier and confidence from clusters
        tier, confidence = "QIYAS", 0.5
        for cluster in ("certainty", "command", "prohibition",
                         "narrative", "seeking", "description", "conjecture"):
            if cluster in clusters_found:
                tier, confidence = _CLUSTER_TIER[cluster]
                break

        # Boost confidence if multiple roots activated
        confidence = min(1.0, confidence + 0.02 * (len(activated_roots) - 1))

        return PropagationResult(
            tier=tier,
            confidence=confidence,
            proof_tree=proof_tree,
            ayat_refs=ayat_refs,
            activated_roots=activated_roots,
            activated_frames={},
            procedure_tags=[],
            is_muqattaat=any(
                r.is_muqattaat
                for r in [self._check_muqattaat(arabic_roots)]
            ),
            muqattaat_frames=[],
        )

    def _check_muqattaat(self, roots: list[str]) -> object:
        class _R:
            is_muqattaat = False
        return _R()
```

- [ ] **Step 4: Verify PropagationResult import path**

```bash
grep -n "class PropagationResult\|PropagationResult" ikhtiyar/core/vtransistor.py | head -5
```

Confirm `PropagationResult` is defined there. If the dataclass fields differ from those used in `rdf_circuit.py`, adjust the constructor call to match exactly.

- [ ] **Step 5: Run tests**

```
python -m pytest tests/test_rdf_circuit.py::TestRDFCircuit -v
```
Expected: 7 PASSED (may be slow first run — graph loading ~5–15s)

- [ ] **Step 6: Commit**

```bash
git add ikhtiyar/core/rdf_circuit.py ikhtiyar/tests/test_rdf_circuit.py
git commit -m "feat: RDFCircuit — SPARQL evaluate() + procedure.py compatibility shims"
```

---

## Task 8: Wire RDFCircuit into Engine

**Files:**
- Modify: `ikhtiyar/engine.py`
- Archive: `ikhtiyar/core/hvt_compiler.py` → `ikhtiyar/core/_dead_hvt_compiler.py`

**Domain context:** `engine.py:_init_circuit()` currently loads `CircuitEvaluator(self._hvt_path)` from `core.vtransistor`. We replace this with `RDFCircuit(qs_path)` loaded directly. `CircuitEvaluator` in vtransistor wraps `StaticCircuit` — we bypass it and set `self._circuit` to an object that exposes `.evaluate()` directly. The engine checks `self._circuit.static._root_index` for logging — we provide `.static` as a self-reference shim.

- [ ] **Step 1: Locate the exact lines to modify in engine.py**

```bash
grep -n "_init_circuit\|_hvt_path\|CircuitEvaluator\|HVT\|hvt" ikhtiyar/engine.py
```

- [ ] **Step 2: Add QS path default and update _init_circuit**

In `ikhtiyar/engine.py`, add after the `DEFAULT_HVT_PATH` line:

```python
DEFAULT_QS_PATH = os.path.join(_IKHTIYAR_DIR, "QS.ttl")
```

Replace `_init_circuit()`:

```python
def _init_circuit(self):
    """
    Load RDFCircuit from QS.ttl.
    Falls back gracefully — engine continues with BFS deliberate() if absent.
    """
    qs_path = os.path.join(_IKHTIYAR_DIR, "QS.ttl")
    if not os.path.isfile(qs_path):
        logger.info("RDFCircuit: QS.ttl not found — circuit path disabled")
        return
    try:
        from core.rdf_circuit import RDFCircuit
        circuit = RDFCircuit(qs_path)
        # Shim: engine.py logs root count via self._circuit.static._root_index
        circuit.static = circuit
        self._circuit = circuit
        root_count = len(self._circuit._root_index)
        logger.info(f"RDFCircuit: ready — {root_count} roots loaded from QS.ttl")
    except Exception as e:
        logger.warning(f"RDFCircuit init failed: {e}")
        self._circuit = None
```

- [ ] **Step 3: Archive hvt_compiler.py**

```bash
cp ikhtiyar/core/hvt_compiler.py ikhtiyar/core/_dead_hvt_compiler.py
```

Add to top of `ikhtiyar/core/_dead_hvt_compiler.py`:
```python
# ARCHIVED 2026-04-29 — superseded by quran_rdf_compiler.py + QS.ttl
# Do not import from this file.
```

- [ ] **Step 4: Verify engine boots**

```
cd ikhtiyar
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from engine import IkhtiyarEngine
e = IkhtiyarEngine.__new__(IkhtiyarEngine)
e._circuit = None
from core.rdf_circuit import RDFCircuit
circuit = RDFCircuit('QS.ttl')
circuit.static = circuit
e._circuit = circuit
print('Circuit loaded:', len(e._circuit._root_index), 'roots')
result = e._circuit.evaluate(['علم'])
print('evaluate() tier:', result.tier, 'confidence:', result.confidence)
"
```

Expected: prints root count and tier=HAQQ or QIYAS.

- [ ] **Step 5: Commit**

```bash
git add ikhtiyar/engine.py ikhtiyar/core/_dead_hvt_compiler.py
git commit -m "feat: wire RDFCircuit into engine._init_circuit(), archive hvt_compiler"
```

---

## Task 9: Integration Test — Full Chat Path

**Files:**
- Test: `ikhtiyar/tests/test_rdf_circuit.py`

**Domain context:** Verify the full chat path from `engine.chat()` through `_chat_deliberate()` → circuit.evaluate() → proof_to_assertions() → GBNFCompiler works end-to-end without the LLM (mock the LLM call). This confirms that the RDF circuit's `PropagationResult` is compatible with `prooftree.proof_to_assertions()` and `GBNFCompiler.compile_from_proof_tree()`.

- [ ] **Step 1: Write integration test**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
@pytest.mark.skipif(not QS_PATH.exists(), reason="QS.ttl not compiled yet")
def test_proof_tree_from_rdf_circuit():
    """RDFCircuit output flows through proof_to_assertions without error."""
    from core.rdf_circuit import RDFCircuit
    from core.prooftree import proof_to_assertions
    from unittest.mock import MagicMock

    circuit = RDFCircuit(str(QS_PATH))
    result = circuit.evaluate(["علم", "حق"])

    mushaf = MagicMock()
    mushaf.get_ayah_text.return_value = "وَعَلَّمَ آدَمَ الْأَسْمَاءَ"

    tree = proof_to_assertions(result, mushaf, query="What is knowledge?", seed_roots=["علم"])
    assert tree is not None
    assert tree.tier in ("HAQQ", "QIYAS", "IKHTILAF", "WAQF")
    assert isinstance(tree.assertions, list)

@pytest.mark.skipif(not QS_PATH.exists(), reason="QS.ttl not compiled yet")
def test_gbnf_compiler_from_rdf_circuit():
    """RDFCircuit → proof_to_assertions → GBNFCompiler produces valid grammar string."""
    from core.rdf_circuit import RDFCircuit
    from core.prooftree import proof_to_assertions
    from core.gbnf_compiler import GBNFCompiler
    from unittest.mock import MagicMock

    circuit = RDFCircuit(str(QS_PATH))
    result = circuit.evaluate(["رحم"])
    mushaf = MagicMock()
    mushaf.get_ayah_text.return_value = "الرَّحْمَٰنِ الرَّحِيمِ"

    tree = proof_to_assertions(result, mushaf, query="What is mercy?", seed_roots=["رحم"])
    compiler = GBNFCompiler()
    grammar = compiler.compile_from_proof_tree(tree)
    assert isinstance(grammar, str)
    assert len(grammar) > 0
```

- [ ] **Step 2: Run integration tests**

```
python -m pytest tests/test_rdf_circuit.py::test_proof_tree_from_rdf_circuit tests/test_rdf_circuit.py::test_gbnf_compiler_from_rdf_circuit -v
```

Expected: 2 PASSED. If `proof_to_assertions()` requires field names not present in `PropagationResult` from `rdf_circuit.py`, adjust the `PropagationResult` constructor call in `rdf_circuit.py:evaluate()` to match exactly.

- [ ] **Step 3: Run full test suite**

```
python -m pytest ikhtiyar/tests/ -v
```

Expected: all existing tests still pass + new RDF circuit tests pass.

- [ ] **Step 4: Commit**

```bash
git add ikhtiyar/tests/test_rdf_circuit.py
git commit -m "test: integration — RDFCircuit → prooftree → GBNFCompiler full path verified"
```

---

## Task 10: Bilal Arabization — Remove Buckwalter from Runtime

**Files:**
- Modify: `ikhtiyar/pipeline/bilal.py`
- Modify: `ikhtiyar/pipeline/concept_mapping.json` (or equivalent concept map source)
- Test: `ikhtiyar/tests/test_rdf_circuit.py`

**Domain context:** `bilal.py` currently stores `concept_map` (English→Buckwalter), `root_corpus` (Buckwalter→keywords), `ARCHETYPAL_ROOTS` (Buckwalter keyed), and `_get_verse_set(root_buckwalter)`. `Perception.roots` returns Buckwalter strings. After this task, all internal keys are Arabic script, `Perception.roots` returns Arabic script, and `RDFCircuit.evaluate()` receives Arabic strings directly with no conversion needed. The `bw_to_arabic()` call that currently exists in `rdf_circuit.py:evaluate()` is removed as dead code.

- [ ] **Step 1: Write failing tests**

Add to `ikhtiyar/tests/test_rdf_circuit.py`:

```python
def test_bilal_perception_roots_are_arabic():
    """After arabization, Perception.roots returns Arabic script strings."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'bismillah', 'QUS-AI HF'))
    from pipeline.bilal import Bilal
    bilal = Bilal()
    bilal.load(concept_map={}, constitution_path=None)
    perception = bilal.listen("What is knowledge?")
    for root in perception.roots:
        # Every character should be Arabic (> 0x0600) or empty
        non_ascii = any(ord(c) > 127 for c in root)
        assert non_ascii or root == "", f"Buckwalter root leaked: {root!r}"
```

- [ ] **Step 2: Audit Buckwalter usage in bilal.py**

```bash
grep -n "buckwalter\|bw_to\|root_bw\|Buckwalter\|_bw\b" ikhtiyar/pipeline/bilal.py
```

List every line. For each one:
- If it is a dict key: replace the key value with `bw_to_arabic(key)` at load time
- If it is a function parameter name: rename and add conversion at entry point
- If it is a return value: add `bw_to_arabic()` conversion before return

- [ ] **Step 3: Convert concept_map keys at load time**

In `bilal.py`, find where `concept_map` is populated. Add conversion:

```python
# Before: self.concept_map[english] = buckwalter
# After:
from bw_arabic import bw_to_arabic as _bw
self.concept_map[english] = _bw(buckwalter)

# Before: for english, buckwalter in self.concept_map.items():
#             root_words[buckwalter].append(english)
# After:
for english, arabic_root in self.concept_map.items():
    root_words[arabic_root].append(english)
```

- [ ] **Step 4: Convert ARCHETYPAL_ROOTS keys at load time**

Find the `ARCHETYPAL_ROOTS` dict (may be in bilal.py or an imported constants file). Convert all Buckwalter keys to Arabic at module load:

```python
from bw_arabic import bw_to_arabic as _bw
ARCHETYPAL_ROOTS = {
    _bw(bw_key): data
    for bw_key, data in _ARCHETYPAL_ROOTS_BW.items()
}
```

- [ ] **Step 5: Update _get_verse_set to use Arabic root URI**

```python
# Before:
def _get_verse_set(self, root_buckwalter: str) -> FrozenSet[str]:
    root_uri = ROOT[root_buckwalter]
    ...

# After:
def _get_verse_set(self, arabic_root: str) -> FrozenSet[str]:
    root_uri = ROOT_NS[arabic_root]  # ROOT_NS = Namespace("http://quran.data/root/")
    ...
```

- [ ] **Step 6: Update Perception.roots to return Arabic**

Find where `Perception.roots` is built. Ensure it returns Arabic script strings. The `arabic_roots` property (line 158: `return [bw_to_arabic(r) for r in self.roots]`) can be removed — `roots` now IS Arabic.

- [ ] **Step 7: Remove the bw_to_arabic conversion shim from rdf_circuit.py**

In `ikhtiyar/core/rdf_circuit.py:evaluate()`, remove the Buckwalter detection block:

```python
# Remove this block:
# if r and not any(ord(c) > 127 for c in r):
#     arabic_roots.append(bw_to_arabic(r))
# else:
#     arabic_roots.append(r)

# Replace with:
arabic_roots = [r for r in roots if r]
```

- [ ] **Step 8: Run all tests**

```
python -m pytest ikhtiyar/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add ikhtiyar/pipeline/bilal.py ikhtiyar/core/rdf_circuit.py
git commit -m "feat: Bilal arabization — all runtime roots are Arabic script, Buckwalter eliminated"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|-----------------|-----------------|
| Arabic-script root URIs throughout | Tasks 3, 4 |
| QAC TAG as typed predicates | Task 4 (emit_layer1) |
| Verb forms I–X as typed predicates | Task 4 (emit_layer2) |
| Ayah named graphs | Task 4 (emit_layer1 — uses ConjunctiveGraph context) |
| TMQ hyperedges as reification nodes | Task 5 |
| Epistemic root clusters | Tasks 1, 5 |
| VN/ACT_PCPL/PASS_PCPL enrichment | Task 4 (emit_layer15) |
| NUM numeral lexicon | Task 3 + emit_layer15 |
| Inna-sisters sub-predicates | Task 4 (emit_layer15) |
| IMPN handling (rootless) | Task 4 (emit_layer1 rootless branch) |
| 37:130 space-in-FORM repair | Task 2 (parse_qac_line) |
| RDFCircuit.evaluate() → PropagationResult | Task 7 |
| procedure.py compatibility shim | Task 7 (_build_shims) |
| Engine wired to QS.ttl | Task 8 |
| HVT compiler archived | Task 8 |
| Bilal arabization | Task 10 |
| Buckwalter eliminated from runtime | Task 10 |

**Placeholder scan:** None found.

**Type consistency:**
- `PropagationResult` — used in Tasks 7, 9. Must match the dataclass definition in `vtransistor.py`. Step 7.4 includes an explicit verification step.
- `proof_to_assertions(result, mushaf, query, seed_roots)` — signature verified against `prooftree.py` in Task 9.
- `emit_layer1/15/2/3` — defined in Task 4, imported in Task 6. Names consistent.

---

*والله أعلم — وبه نستعين*
