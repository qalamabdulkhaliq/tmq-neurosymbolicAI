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
