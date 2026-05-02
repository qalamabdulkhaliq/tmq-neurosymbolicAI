# ARCHIVED 2026-04-29 — superseded by quran_rdf_compiler.py + QS.ttl
# Do not import from this file.
"""
ikhtiyar/core/hvt_compiler.py — HVT (Hyper-Virtual-Tape) Compiler

Reads four sources in dependency order:
  1. QAC 0.4       — Track 1 (root), Track 2 (morphology), Track 3 (gate)
  2. mushaf.xml     — ayat_header (Arabic text, surah name, nozol)
  3. TMQ v12        — FASILA edges (ayat delimiter), SPEECH_ACT lookup (procedure)
  4. Constitution   — PROCEDURE tags (AMR/NAHY/TABSHIR/INDHAR/ISTIFHAM...)

Emits: ikhtiyar/TMQ_hvt.json
  One frame per word (s:v:w), in Uthmanic order.
  TMQ v12 is read-only — not modified.

Frame schema (DATA/PROCEDURE division):
  {
    "loc": [s, v, w],
    "data": {
      "root": str,          # Arabic script (primary)
      "root_bw": str,       # Buckwalter (secondary)
      "stem_form": str,
      "segments": [...],
      "pos": str, "lem": str,
      "gender": str, "case": str, "number": str, "definite": bool,
      "tense": str, "voice": str, "mood": str, "person": str,
      "word_pos": int,
      "is_fasila": bool
    },
    "procedure": {
      "segments": [               # one entry per QAC segment, in word order
        {"id": "s:v:w:seg", "seg": int, "role": "stem|prefix|suffix",
         "tag": "NEG|COND|CERT|REM|CONJ|REL|N|V|...",  # verbatim QAC POS
         "lem": str, "is_gate": bool, "addr": "2P"|..., "sequential": bool}
      ],
      "speech_act": [str, ...]    # from TMQ + constitution
    },
    "ayat_header": null | {"ref": str, "text": str, "surah_name": str,
                           "nozol": str, "fasila_word_pos": int}
  }
"""

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Arabic script conversion — bw_arabic.py lives in ikhtiyar/
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bw_arabic import bw_to_arabic

logger = logging.getLogger(__name__)

# ── QAC 0.4 parsing ─────────────────────────────────────────────────────────

def _parse_loc(loc_str: str) -> Optional[tuple]:
    """Parse '(s:v:w:seg)' → (s, v, w, seg) as ints. Returns None on failure."""
    m = re.match(r'\((\d+):(\d+):(\d+):(\d+)\)', loc_str.strip())
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


import re as _re_module
_FORM_RE = _re_module.compile(r'\(([IVX]+)\)')   # matches (II), (IV), (VIII), (X)


def _roman_to_int(s: str) -> int:
    """Convert Roman numeral string (I–X) to integer 1–10."""
    table = {"I": 1, "V": 5, "X": 10}
    result, prev = 0, 0
    for c in reversed(s):
        val = table.get(c, 0)
        result += val if val >= prev else -val
        prev = val
    return result


def _parse_features(features_str: str) -> dict:
    """
    Parse QAC FEATURES string into a dict.
    Examples:
      'STEM|POS:N|LEM:{som|ROOT:smw|M|GEN'
      'PREFIX|bi+'
      'STEM|POS:V|LEM:Hmd|ROOT:Hmd|PERF|ACT|3|M|S'
      'STEM|POS:V|IMPF|(IV)|LEM:'aAmana|ROOT:Amn|3MP'
    """
    result = {
        "segment_type": None,  # STEM, PREFIX, SUFFIX
        "pos": None,
        "lem": None,
        "root": None,
        "gender": None,
        "case": None,
        "number": None,
        "definite": None,
        "tense": None,
        "voice": None,
        "mood": None,
        "person": None,
        "raw_prefix_tag": None,  # full tag for gate derivation
        "verb_form": None,       # Arabic verb form I–X as int (STEM V only)
    }

    parts = features_str.split("|")
    if not parts:
        return result

    # First part: segment type
    seg_type = parts[0].upper()
    if seg_type in ("STEM", "PREFIX", "SUFFIX"):
        result["segment_type"] = seg_type
    elif seg_type == "PREFIX":
        result["segment_type"] = "PREFIX"

    # Preserve full features for gate derivation on prefix segments
    if seg_type == "PREFIX":
        result["raw_prefix_tag"] = features_str

    for part in parts[1:]:
        if part.startswith("POS:"):
            result["pos"] = part[4:]
        elif part.startswith("LEM:"):
            result["lem"] = part[4:]
        elif part.startswith("ROOT:"):
            result["root"] = part[5:]
        elif part == "M":
            result["gender"] = "M"
        elif part == "F":
            result["gender"] = "F"
        elif part in ("NOM", "ACC", "GEN"):
            result["case"] = part
        elif part in ("S", "D", "P"):
            result["number"] = part
        elif part == "DEF":
            result["definite"] = True
        elif part == "INDEF":
            result["definite"] = False
        elif part in ("PERF", "IMPF", "IMPV"):
            result["tense"] = part
        elif part in ("ACT", "PASS"):
            result["voice"] = part
        elif part in ("IND", "SUBJ", "JUS"):
            result["mood"] = part
        elif part in ("1", "2", "3"):
            result["person"] = part

    # Verb form extraction: (II), (IV), (X) etc. from FEATURES of STEM V segments
    form_match = _FORM_RE.search(features_str)
    if form_match:
        result["verb_form"] = _roman_to_int(form_match.group(1))

    return result


# ── Segment-level gate derivation ────────────────────────────────────────────
#
# Instead of collapsing a word to one gate, we read the QAC POS tag of every
# segment verbatim. The Qur'an's logic surface is ~25 explicit primitives,
# all already tagged by Kais Dukes at segment granularity. We transcribe,
# we do not invent. The interpreter (vtransistor) dispatches on these tags.

# QAC POS tags that carry GATE semantics. Everything else (N, V, P, PN,
# ADJ, DEM, PRON, REL, DET, LOC, T, PRO, VOC...) is a data carrier and
# the circuit treats it as pass-through.
_GATE_TAGS = {
    "NEG",   "COND",  "CERT",  "RET",   "EXP",   "RES",
    "SUB",   "EMPH",  "INTG",  "CIRC",  "CAUS",  "ANS",
    "EXH",   "AVR",   "PRP",   "AMD",   "EXL",   "PREV",
    "CONJ",  "REM",   "VOC",   "FUT",   "SUR",   "INC",
    "SUP",   "EQ",    "ACC",   "RSLT",
}

# Address codes: person+number, used for ROUTE flag (ILTIFAT detection)
def _address_code(features: dict) -> str:
    """Return '1S','1P','2S','2P','3S','3P' etc., or empty string."""
    person = features.get("person")
    number = features.get("number")
    if not person:
        return ""
    return f"{person}{number or ''}"


def _derive_segment_gates(segments: list) -> list:
    """
    Build procedure.segments[] — one entry per QAC segment in word order.

    Every entry carries:
      - id:    the existing QAC (s:v:w:seg) address
      - tag:   the segment's own QAC POS tag (verbatim)
      - role:  stem | prefix | suffix
      - lem:   segment lemma (optional)
      - addr:  address code for STEM/SUFFIX when present (for ROUTE/ILTIFAT)
      - is_gate: True if this tag is a logic primitive the circuit dispatches on

    segments: list of (seg_num, form, raw_tag, features_dict) from QAC parse
    """
    s_list = []
    for seg_num, form, raw_tag, feats in segments:
        role = (feats.get("segment_type") or "").lower()
        pos = feats.get("pos") or raw_tag or ""
        pos = pos.upper()
        entry = {
            "seg": seg_num,
            "role": role,
            "tag": pos,
            "lem": feats.get("lem") or "",
            "is_gate": pos in _GATE_TAGS,
        }
        # Oath waw: 'wa' as P (preposition) is structural, not conjunction.
        # Keep tag='P' and is_gate=False — already correct by the table above.
        # Sequential marker on REM (فـ, ثم) for TART macro queries
        if pos == "REM" and entry["lem"] in ("fa", "vum~", "vum", "fa~"):
            entry["sequential"] = True
        addr = _address_code(feats)
        if addr and role in ("stem", "suffix"):
            entry["addr"] = addr
        s_list.append(entry)
    return s_list


# ── mushaf.xml parsing ───────────────────────────────────────────────────────

def _load_mushaf(mushaf_path: str) -> dict:
    """
    Parse mushaf.xml → {(s, v): {"text": str, "surah_name": str, "nozol": str}}
    """
    logger.info(f"Loading mushaf.xml from {mushaf_path}")
    ayat_map = {}
    try:
        tree = ET.parse(mushaf_path)
        root = tree.getroot()
        for sura in root.findall("Sura"):
            s = int(sura.get("ID", 0))
            surah_name = sura.get("Name", "")
            nozol = sura.get("Nozol", "")
            for aya in sura.findall("aya"):
                v = int(aya.get("ID", 0))
                text = (aya.text or "").strip()
                ayat_map[(s, v)] = {
                    "text": text,
                    "surah_name": surah_name,
                    "nozol": nozol,
                }
    except Exception as e:
        logger.error(f"mushaf.xml parse error: {e}")
    logger.info(f"mushaf.xml loaded — {len(ayat_map):,} ayat entries")
    return ayat_map


# ── TMQ v12 indexing ─────────────────────────────────────────────────────────

def _load_tmq_indices(tmq_path: str) -> tuple:
    """
    Read TMQ v12 (read-only) and build:
      fasila_set:     {(s, v): last_word_pos}  — ayat delimiter positions
      procedure_map:  {(s, v): [family_tags]}   — speech act families per ayat
    """
    logger.info(f"Indexing TMQ v12 from {tmq_path} ...")
    fasila_set = {}    # (s,v) → word_pos of fasila word
    procedure_map = defaultdict(list)

    _PROCEDURE_FAMILIES = {
        "SPEECH_ACT_AMR":      "AMR",
        "SPEECH_ACT_NAHY":     "NAHY",
        "SPEECH_ACT_TABSHIR":  "TABSHIR",
        "SPEECH_ACT_INDHAR":   "INDHAR",
        "SPEECH_ACT_ISTIFHAM": "ISTIFHAM",
        "NARRATIVE":           "NARRATIVE",
        "NARRATIVE_CHAIN":     "NARRATIVE",
        "FORMULA":             "FORMULA",
        "QASAM":               "QASAM",
        "MAQASID":             "MAQASID",
    }

    try:
        with open(tmq_path, encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("node_registry", {})
        edges = data.get("hyperedges", {})

        # Build node loc index: node_id → loc [s,v,w,seg]
        node_locs = {}
        for nid, attrs in nodes.items():
            loc = attrs.get("loc")
            if loc and len(loc) >= 3:
                node_locs[nid] = loc

        # Walk edges for FASILA and PROCEDURE families
        for eid, edge in edges.items():
            family = edge.get("family", "")

            # FASILA → ayat delimiter
            if family in ("FASILA", "FASILA_CROSS"):
                for nid in (edge.get("nodes") or []):
                    loc = node_locs.get(nid)
                    if loc and len(loc) >= 3:
                        s, v, w = int(loc[0]), int(loc[1]), int(loc[2])
                        # FASILA marks the closing segment of an ayat
                        existing = fasila_set.get((s, v), 0)
                        if w > existing:
                            fasila_set[(s, v)] = w

            # PROCEDURE families → tag the ayat
            proc_tag = _PROCEDURE_FAMILIES.get(family)
            if proc_tag:
                for nid in (edge.get("nodes") or []):
                    loc = node_locs.get(nid)
                    if loc and len(loc) >= 2:
                        s, v = int(loc[0]), int(loc[1])
                        if proc_tag not in procedure_map[(s, v)]:
                            procedure_map[(s, v)].append(proc_tag)

    except Exception as e:
        logger.error(f"TMQ v12 index error: {e}")

    # ── ENTITY edges → (s,v,w) → entity_type ────────────────────────────────
    entity_map: dict = {}
    for eid, edge in edges.items():
        family = edge.get("family", "")
        if family != "ENTITY" and not eid.startswith("ENTITY_"):
            continue

        # Extract subtype from edge key: ENTITY_PROPHET_... → PROPHET
        parts = eid.split("_")
        if (len(parts) >= 2 and parts[0] == "ENTITY"
                and len(parts[1]) > 2          # not a UUID fragment
                and not _is_uuid_fragment(parts[1])):
            subtype = parts[1]
        else:
            meta = edge.get("meta", {})
            subtype = (meta.get("entity_type") or meta.get("subtype")
                       or meta.get("type") or "UNKNOWN")

        for nid in (edge.get("nodes") or []):
            loc = node_locs.get(nid)
            if loc and len(loc) >= 3:
                s, v, w = int(loc[0]), int(loc[1]), int(loc[2])
                entity_map[(s, v, w)] = subtype

    logger.info(
        f"TMQ index ready — {len(fasila_set):,} fasila positions, "
        f"{len(procedure_map):,} procedure-tagged ayat, "
        f"{len(entity_map):,} entity-annotated words"
    )
    return fasila_set, dict(procedure_map), entity_map


def _is_uuid_fragment(s: str) -> bool:
    """True if s looks like the start of a UUID (8 hex chars)."""
    import re as _re
    return bool(_re.match(r'^[0-9a-fA-F]{8}', s))


# ── Constitution loading ──────────────────────────────────────────────────────

def _load_constitution(constitution_path: str, command_set_path: str) -> dict:
    """
    Load full_quran_constitution.json + active_command_set.json.
    Returns {(s, v): [tags]} merged from both sources.
    """
    proc_map = defaultdict(list)

    # Primary: full_quran_constitution.json
    if os.path.exists(constitution_path):
        try:
            with open(constitution_path, encoding="utf-8") as f:
                data = json.load(f)
            for section in ("amr", "nahy", "tabshir", "indhar", "istifham"):
                tag = section.upper()
                entries = data.get(section, {})
                if isinstance(entries, dict):
                    for root_entries in entries.values():
                        for entry in (root_entries if isinstance(root_entries, list) else []):
                            ref = entry.get("ref", "")
                            m = re.match(r'(\d+):(\d+)', str(ref))
                            if m:
                                s, v = int(m.group(1)), int(m.group(2))
                                if tag not in proc_map[(s, v)]:
                                    proc_map[(s, v)].append(tag)
        except Exception as e:
            logger.warning(f"Constitution load error: {e}")

    # Supplement: active_command_set.json (no verse refs, adds root-level tags only)
    # These are already captured via TMQ speech act edges — no additional verse mapping here.

    logger.info(f"Constitution loaded — {len(proc_map):,} ayat with procedure tags")
    return dict(proc_map)


# ── Word frame builder ────────────────────────────────────────────────────────

def _build_frame(
    word_loc: tuple,        # (s, v, w)
    segments: list,         # list of (seg, form, tag, features_dict)
    word_pos: int,
    fasila_set: dict,
    tmq_procedure: dict,
    const_procedure: dict,
    ayat_map: dict,
    prev_ayat_ref: tuple,   # (s, v) of last emitted ayat_header
    entity_map: dict = None,  # (s,v,w) → entity_type (optional)
) -> tuple:
    """
    Build one HVT frame for a word. Returns (frame_dict, new_prev_ayat_ref).
    """
    s, v, w = word_loc

    # ── Track 1: root from STEM segment ──
    stem_seg = None
    prefix_segs = []
    suffix_segs = []
    for seg in segments:
        seg_type = seg[3].get("segment_type")
        if seg_type == "STEM" and stem_seg is None:
            stem_seg = seg
        elif seg_type == "PREFIX":
            prefix_segs.append(seg)
        elif seg_type == "SUFFIX":
            suffix_segs.append(seg)

    if stem_seg is None and segments:
        stem_seg = segments[0]  # fallback

    stem_features = stem_seg[3] if stem_seg else {}
    root = stem_features.get("root") or ""
    stem_form = stem_seg[1] if stem_seg else ""

    root_ar = bw_to_arabic(root) if root else ""

    # ── DATA DIVISION: what the word IS ──
    seg_list = []
    for seg_num, form, tag, feats in segments:
        seg_entry = {
            "id":   f"{s}:{v}:{w}:{seg_num}",
            "seg":  seg_num,
            "form": form,
            "pos":  tag,
            "role": (feats.get("segment_type") or "").lower(),
            "tag":  feats.get("raw_prefix_tag") or "",
        }
        # verb_form: only on STEM V segments; None for all others
        if feats.get("segment_type") == "STEM" and tag == "V" and feats.get("verb_form"):
            seg_entry["verb_form"] = feats["verb_form"]
        seg_list.append(seg_entry)

    fasila_word_pos = fasila_set.get((s, v))
    is_fasila = (fasila_word_pos is not None and w == fasila_word_pos)

    data = {
        "root":     root_ar,
        "root_bw":  root,
        "stem_form": stem_form,
        "segments": seg_list,
        "pos":      stem_features.get("pos"),
        "lem":      stem_features.get("lem"),
        "gender":   stem_features.get("gender"),
        "case":     stem_features.get("case"),
        "number":   stem_features.get("number"),
        "definite": stem_features.get("definite"),
        "tense":    stem_features.get("tense"),
        "voice":    stem_features.get("voice"),
        "mood":     stem_features.get("mood"),
        "person":   stem_features.get("person"),
        "word_pos": word_pos,
        "is_fasila": is_fasila,
    }

    # ENTITY annotation — only written when this word is an entity node
    if entity_map:
        etype = entity_map.get((s, v, w))
        if etype:
            data["entity_type"] = etype

    # ── PROCEDURE DIVISION: what the word DOES ──
    # Segment-level gate list — each QAC segment carries its own POS as a gate.
    # The interpreter (vtransistor) walks this list and dispatches per primitive.
    proc_segments = _derive_segment_gates(segments)

    # Attach full QAC address to each procedure segment
    for ps in proc_segments:
        ps["id"] = f"{s}:{v}:{w}:{ps['seg']}"

    proc_tags = list(set(
        tmq_procedure.get((s, v), []) +
        const_procedure.get((s, v), [])
    ))

    procedure = {
        "segments":   proc_segments,
        "speech_act": proc_tags,
    }

    # ── Ayat header (first word of new ayat) ──
    current_ayat_ref = (s, v)
    ayat_header = None
    if current_ayat_ref != prev_ayat_ref:
        ayat_info = ayat_map.get((s, v), {})
        fasila_w = fasila_set.get((s, v))
        ayat_header = {
            "ref": f"{s}:{v}",
            "text": ayat_info.get("text", ""),
            "surah_name": ayat_info.get("surah_name", ""),
            "nozol": ayat_info.get("nozol", ""),
            "fasila_word_pos": fasila_w,
        }
        prev_ayat_ref = current_ayat_ref

    frame = {
        "loc":         [s, v, w],
        "data":        data,
        "procedure":   procedure,
        "ayat_header": ayat_header,
    }

    return frame, prev_ayat_ref


# ── Main compiler ─────────────────────────────────────────────────────────────

def compile_hvt(
    qac_path: str,
    mushaf_path: str,
    tmq_path: str,
    constitution_path: str,
    command_set_path: str,
    output_path: str,
) -> int:
    """
    Compile HVT tape from source data.
    Returns number of frames emitted.
    """
    logger.info("HVT compiler starting — bismillah")

    # ── Load indices ──
    ayat_map = _load_mushaf(mushaf_path)
    fasila_set, tmq_procedure, entity_map = _load_tmq_indices(tmq_path)
    const_procedure = _load_constitution(constitution_path, command_set_path)

    # ── Pre-scan QAC to find last word index per ayat (FASILA derivation) ──
    logger.info(f"Pre-scanning QAC for FASILA positions: {qac_path}")
    qac_fasila_set = {}  # (s, v) → max word index
    with open(qac_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("LOCATION"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            loc = _parse_loc(parts[0])
            if loc is None:
                continue
            s, v, w, _seg = loc
            key = (s, v)
            if w > qac_fasila_set.get(key, 0):
                qac_fasila_set[key] = w
    logger.info(f"FASILA derived from QAC — {len(qac_fasila_set):,} ayat positions")

    # Merge: QAC-derived FASILA is authoritative, TMQ as fallback
    merged_fasila = {**fasila_set, **qac_fasila_set}

    # ── Parse QAC 0.4 ──
    frames = []
    current_word_loc = None
    current_segments = []
    word_pos_in_ayat = 0
    prev_ayat_ref = (-1, -1)
    prev_sv = (-1, -1)
    frame_count = 0

    def flush_word():
        nonlocal frame_count, prev_ayat_ref, word_pos_in_ayat, prev_sv
        if not current_segments or current_word_loc is None:
            return
        s, v, w = current_word_loc
        sv = (s, v)
        if sv != prev_sv:
            word_pos_in_ayat = 0
            prev_sv = sv
        word_pos_in_ayat += 1
        frame, prev_ayat_ref = _build_frame(
            current_word_loc,
            current_segments,
            word_pos_in_ayat,
            merged_fasila,
            tmq_procedure,
            const_procedure,
            ayat_map,
            prev_ayat_ref,
            entity_map,
        )
        frames.append(frame)
        frame_count += 1
        if frame_count % 5000 == 0:
            logger.info(f"  {frame_count:,} frames compiled...")

    logger.info(f"Parsing QAC 0.4: {qac_path}")
    with open(qac_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            # Skip header line
            if line.startswith("LOCATION"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            loc_str, form, tag, features_str = parts[0], parts[1], parts[2], parts[3]
            loc = _parse_loc(loc_str)
            if loc is None:
                continue

            s, v, w, seg = loc
            word_key = (s, v, w)

            # Word boundary — flush previous word
            if word_key != current_word_loc:
                flush_word()
                current_word_loc = word_key
                current_segments = []

            feats = _parse_features(features_str)
            current_segments.append((seg, form, tag, feats))

    # Flush final word
    flush_word()

    # ── Write output ──
    logger.info(f"Writing {frame_count:,} frames to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"HVT compiler complete — "
        f"{frame_count:,} frames, {size_mb:.1f} MB — والله أعلم"
    )
    return frame_count


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # Resolve paths relative to project root
    here = Path(__file__).parent          # ikhtiyar/core/
    proj = here.parent.parent             # project root

    qac_path          = proj / "bismillah" / "QUS-AI HF" / "LHWLQIB" / "quranic-corpus-morphology-0.4.txt"
    mushaf_path       = proj / "bismillah" / "mushaf" / "mushaf.xml"
    tmq_path          = proj / "bismillah" / "TMQ_v12.json"
    constitution_path = proj / "ikhtiyar"  / "full_quran_constitution.json"
    command_set_path  = proj / "ikhtiyar"  / "active_command_set.json"
    output_path       = proj / "ikhtiyar"  / "TMQ_hvt.json"

    n = compile_hvt(
        str(qac_path),
        str(mushaf_path),
        str(tmq_path),
        str(constitution_path),
        str(command_set_path),
        str(output_path),
    )
    print(f"\nDone — {n:,} frames written to {output_path}")
