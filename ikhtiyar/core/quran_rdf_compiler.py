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
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from urllib.parse import quote as _urlencode

from rdflib import ConjunctiveGraph, URIRef, Literal, RDF, XSD
from rdflib.namespace import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bw_arabic import bw_to_arabic

logger = logging.getLogger(__name__)

# ── Namespaces (all Arabic-script URIs) ──────────────────────────────────────
QS_NS    = Namespace("http://quran.data/")
ROOT_NS  = Namespace("http://quran.data/root/")
WORD_NS  = Namespace("http://quran.data/word/")
AYAH_NS  = Namespace("http://quran.data/ayah/")
POS_NS   = Namespace("http://quran.data/pos/")
FORM_NS  = Namespace("http://quran.data/form/")
MORPH_NS = Namespace("http://quran.data/morph/")
EDGE_NS  = Namespace("http://quran.data/edge/")
EPI_NS   = Namespace("http://quran.data/epistemic/")

_SP_SUBPRED = {
    "<in~":   "ACC_IN",
    ">an~":   "ACC_IN",
    "laEal~": "ACC_LAAL",
    "layta":  "ACC_LAYTA",
    "lakin~": "ACC_LAKIN",
    "ka>an~": "ACC_KAANN",
}

_FAMILY_TYPES = {
    "ILTIFAT":        "IltifatEdge",
    "FORMULA":        "FormulaEdge",
    "NARRATIVE":      "NarrativeEdge",
    "SPEECH_ACT_AMR": "SpeechActEdge",
    "SPEECH_ACT_NAHY":"SpeechActEdge",
    "WAQF":           "WaqfEdge",
    "TART":           "TartEdge",
    "MORPH_ROOT":     "MorphRootEdge",
    "MORPH_LEM":      "MorphLemEdge",
    "SYN_PART":       "SynPartEdge",
    "SYN_PRON":       "SynPronEdge",
}

# ── Constants ────────────────────────────────────────────────────────────────


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
])

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
    """Convert Roman numeral string to integer."""
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

    QAC 0.4 format:
      LOCATION	FORM	TAG	FEATURES
      (s:v:w:seg)	Buckwalter	POS	pipe-separated

    Space-in-FORM repair:
      When FORM contains a space, the tab split distributes columns incorrectly.
      If TAG is not in VALID_TAGS, rejoin columns 1+2 as FORM and shift indices right.

    Returns:
      dict with keys: loc, form_bw, tag, root_bw, lemma, pos, seg_type, verb_form,
                      person, number, gender, case, tense, voice, mood, is_vn,
                      is_pcpl_act, is_pcpl_pass, sp_field
      None for comments, headers, blank lines
    """
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None

    parts = line.split("\t")
    if len(parts) < 4:
        return None

    loc_str, form_bw, tag, features = parts[0], parts[1], parts[2], parts[3]

    # Space-in-FORM repair: if tag not valid, columns shifted left by a space
    # When FORM has space, split by tab gives too many parts; we rejoin parts[1] + " " + parts[2]
    if tag not in VALID_TAGS and tag != "TAG":
        # Columns shifted: form_bw is actually part of what should be FORM
        # Rejoin: form_bw (parts[1]) + " " + tag (parts[2]) → new form_bw
        # New tag, features come from parts[3], parts[4]
        form_bw = parts[1] + " " + parts[2]
        tag = parts[3] if len(parts) > 3 else ""
        features = parts[4] if len(parts) > 4 else ""
        if tag not in VALID_TAGS:
            return None  # unrecoverable

    if tag == "TAG":   # header row
        return None

    # Parse location (s:v:w:seg)
    m = re.match(r'\((\d+):(\d+):(\d+):(\d+)\)', loc_str.strip())
    if not m:
        return None
    loc = tuple(int(x) for x in m.groups())  # (s, v, w, seg)

    # Parse FEATURES: pipe-separated string
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
            pass
        elif len(p) == 2 and p[0] in "123" and p[1] in "SDP":
            # Combined person+number like "3S", "2P", "1D"
            person = p[0]
            number = p[1]
        elif len(p) == 3 and p in ("1MS", "1FS", "2MS", "2FS", "3MS", "3FS", "1MP", "1FP", "2MP", "2FP", "3MP", "3FP"):
            # Extended person+gender+number like "1MS", "2FS"
            person = p[0]
            gender = p[1]
            number = p[2]

    # ACT|PCPL and PASS|PCPL detection
    feat_str = "|".join(feat_parts)
    if "ACT|PCPL" in feat_str or "ACT.PCPL" in feat_str:
        is_pcpl_act = True
    if "PASS|PCPL" in feat_str or "PASS.PCPL" in feat_str:
        is_pcpl_pass = True

    # Verb form: extract Roman numeral from (I), (II), ..., (X)
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


# ── URI builders ─────────────────────────────────────────────────────────────

def build_root_uri(arabic_root: str) -> URIRef:
    return ROOT_NS[arabic_root]

def build_word_uri(s: int, v: int, w: int) -> URIRef:
    return WORD_NS[f"{s}:{v}:{w}"]

def build_ayah_uri(s: int, v: int) -> URIRef:
    return AYAH_NS[f"{s}:{v}"]


# ── Triple emitters ───────────────────────────────────────────────────────────

def emit_layer1(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 1: (root-URI, pos:TAG, word-instance) in named graph ayah:s:v."""
    s, v, w, _ = parsed["loc"]
    root_ar = to_arabic_root(parsed["root_bw"])
    word_uri = build_word_uri(s, v, w)
    ayah_uri = build_ayah_uri(s, v)
    tag = parsed["tag"]
    subj = build_root_uri(root_ar) if root_ar else word_uri
    ctx = g.get_context(ayah_uri)
    ctx.add((subj, POS_NS[tag], word_uri))


def emit_layer15(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 1.5: classical Arabic enrichment predicates from FEATURES."""
    s, v, w, _ = parsed["loc"]
    root_ar = to_arabic_root(parsed["root_bw"])
    if not root_ar:
        return
    word_uri = build_word_uri(s, v, w)
    ayah_uri = build_ayah_uri(s, v)
    ctx = g.get_context(ayah_uri)
    root_uri = build_root_uri(root_ar)
    if parsed["is_vn"]:
        ctx.add((root_uri, POS_NS["VN"], word_uri))
    if parsed["is_pcpl_act"]:
        ctx.add((root_uri, POS_NS["ACT_PCPL"], word_uri))
    if parsed["is_pcpl_pass"]:
        ctx.add((root_uri, POS_NS["PASS_PCPL"], word_uri))
    if root_ar in NUMERAL_ROOTS:
        ctx.add((root_uri, POS_NS["NUM"], word_uri))
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
    ctx = g.get_context(ayah_uri)
    ctx.add((word_uri, FORM_NS[str(parsed["verb_form"])], root_uri))


def emit_layer3(g: ConjunctiveGraph, parsed: dict) -> None:
    """Layer 3: morphological attributes as literals in default graph."""
    s, v, w, _ = parsed["loc"]
    word_uri = build_word_uri(s, v, w)
    dg = g.default_context
    arabic_form = bw_to_arabic(parsed["form_bw"]) if parsed["form_bw"] else ""
    if arabic_form:
        dg.add((word_uri, MORPH_NS["arabic"], Literal(arabic_form)))
    for attr in ("person", "number", "gender", "case", "tense", "voice", "mood"):
        val = parsed.get(attr, "")
        if val:
            dg.add((word_uri, MORPH_NS[attr], Literal(val)))
    dg.add((word_uri, QS_NS["loc"], Literal(f"{s}:{v}:{w}")))
    if parsed["lemma"]:
        dg.add((word_uri, MORPH_NS["lemma_bw"], Literal(parsed["lemma"])))


def emit_root_node(g: ConjunctiveGraph, arabic_root: str, frequency: int = 0) -> None:
    """Emit root node in default graph with type, Arabic label, frequency, cluster."""
    from core.epistemic_clusters import get_cluster_uri
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
    """Emit one TMQ hyperedge as an RDF reification node in the default graph."""
    dg = g.default_context
    safe_id = _urlencode(edge_id, safe="-_.")
    edge_uri = EDGE_NS[safe_id]
    family = edge_data.get("family", "")
    rdf_type = _FAMILY_TYPES.get(family, "TMQEdge")
    dg.add((edge_uri, RDF.type, QS_NS[rdf_type]))
    dg.add((edge_uri, QS_NS["tmq_id"], Literal(edge_id)))
    dg.add((edge_uri, QS_NS["family"], Literal(family)))
    for attr_key, attr_val in edge_data.get("attrs", {}).items():
        if isinstance(attr_val, str):
            dg.add((edge_uri, QS_NS[attr_key], Literal(attr_val)))
    for node_id in edge_data.get("members", []):
        loc = loc_index.get(node_id)
        if loc:
            s, v, w = loc
            dg.add((edge_uri, QS_NS["connects"], build_word_uri(s, v, w)))


# ── Main compile ──────────────────────────────────────────────────────────────

def _load_mushaf(mushaf_path: str) -> dict:
    """Parse mushaf.xml → {(s,v): arabic_text}"""
    ayah_text = {}
    try:
        tree = ET.parse(mushaf_path)
        for i, sura_el in enumerate(tree.getroot().findall("Sura"), start=1):
            sid = int(sura_el.get("ID", i))
            for j, aya_el in enumerate(sura_el.findall("aya"), start=1):
                aid = int(aya_el.get("ID", j))
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
    Runs once (~45-90s). Logs progress every 10k lines.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    print("Bismillah Al-Rahman Al-Raheem")
    print(f"Compiling QS.ttl\n  QAC:    {qac_path}\n  TMQ:    {tmq_path}")

    g = ConjunctiveGraph()
    root_freq: dict[str, int] = defaultdict(int)
    word_to_root: dict[tuple, str] = {}

    # ── Pass 1: QAC ──────────────────────────────────────────────────────────
    print("Pass 1: QAC morphological parse...")
    with open(qac_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no % 10000 == 0 and line_no > 0:
                print(f"  {line_no:,} lines...")
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
    print(f"  QAC done. Roots: {len(root_freq)}")

    # ── Pass 2: Root nodes ────────────────────────────────────────────────────
    print("Pass 2: Root nodes + epistemic clusters...")
    for arabic_root, freq in root_freq.items():
        emit_root_node(g, arabic_root, frequency=freq)

    # ── Pass 3: Mushaf text ───────────────────────────────────────────────────
    print("Pass 3: Mushaf text...")
    ayah_text = _load_mushaf(mushaf_path)
    if ayah_text:
        for (sid, vid), text in ayah_text.items():
            ayah_uri = build_ayah_uri(sid, vid)
            g.default_context.add((ayah_uri, QS_NS["text"], Literal(text)))
        print(f"  Mushaf: {len(ayah_text)} ayahs")
    else:
        print("  Mushaf: skipped (not found)")

    # ── Pass 4: TMQ hyperedges ────────────────────────────────────────────────
    print("Pass 4: TMQ v12 hyperedges...")
    with open(tmq_path, encoding="utf-8") as f:
        tmq = json.load(f)
    loc_index = _build_tmq_loc_index(tmq)
    morph_root_pairs: list = []
    edge_count = 0
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
        if family == "MORPH_ROOT":
            roots_in_edge = [
                word_to_root[loc_index[nid]]
                for nid in members
                if nid in loc_index and loc_index[nid] in word_to_root
            ]
            if len(roots_in_edge) >= 2:
                morph_root_pairs.append(tuple(roots_in_edge))
    dg = g.default_context
    for pair in morph_root_pairs:
        for i in range(len(pair)):
            for j in range(i + 1, len(pair)):
                r1, r2 = build_root_uri(pair[i]), build_root_uri(pair[j])
                dg.add((r1, QS_NS["rootRelated"], r2))
                dg.add((r2, QS_NS["rootRelated"], r1))
    print(f"  TMQ edges: {edge_count}")

    # ── Serialize ─────────────────────────────────────────────────────────────
    print(f"Serializing → {out_path} ...")
    g.serialize(destination=out_path, format="trig")
    triple_count = sum(1 for _ in g.quads())
    print(f"Done. Triples: {triple_count:,}. File: {out_path}")


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parent.parent.parent
    compile(
        qac_path=str(_ROOT / "bismillah/QUS-AI HF/LHWLQIB/quranic-corpus-morphology-0.4.txt"),
        mushaf_path=str(_ROOT / "bismillah/mushaf/mushaf.xml"),
        tmq_path=str(_ROOT / "bismillah/TMQ_v12.json"),
        out_path=str(_ROOT / "ikhtiyar/QS.ttl"),
    )
