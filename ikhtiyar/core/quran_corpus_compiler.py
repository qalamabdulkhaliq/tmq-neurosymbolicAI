"""
quran_corpus_compiler.py — Full Quranic OS Constitution Compiler

Sources:
  1. bismillah/mushaf/mushaf.xml      — full tashkeel text, every ayah
  2. bismillah/TMQ_v12.json           — 28-family semantic hypergraph
  3. quran_root_ontology_v3.ttl       — 650k RDF triples, morphological ground

Output: ikhtiyar/full_quran_constitution.json

The Quran is the OS. Every ayah. Every structure.
"""

import json
import os
import re
import sys
import io
import xml.etree.ElementTree as ET
from collections import defaultdict

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MUSHAF_PATH  = os.path.join(PROJECT_ROOT, "bismillah", "mushaf", "mushaf.xml")
TMQ_PATH     = os.path.join(PROJECT_ROOT, "bismillah", "TMQ_v12.json")
TTL_PATH     = os.path.join(PROJECT_ROOT, "quran_root_ontology_v3.ttl")
OUT_PATH     = os.path.join(PROJECT_ROOT, "ikhtiyar", "full_quran_constitution.json")

print("بسم الله الرحمن الرحيم")
print("مجمّع الدستور القرآني الكامل\n")

# ── 1. MUSHAF ────────────────────────────────────────────────────────────────

print("══ تحميل المصحف الشريف ══")
tree = ET.parse(MUSHAF_PATH)
root_el = tree.getroot()

surahs = {}
ayah_text = {}   # (surah_int, ayah_int) → text

BISMILLAH = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

for sura_el in root_el.findall("Sura"):
    sid   = int(sura_el.get("ID"))
    name  = sura_el.get("Name", "")
    nozol = sura_el.get("Nozol", "")
    nb    = int(sura_el.get("Nb_aya", 0))
    ayat  = {}
    for aya_el in sura_el.findall("aya"):
        aid = int(aya_el.get("ID"))
        txt = (aya_el.text or "").strip()
        ayat[aid] = txt
        ayah_text[(sid, aid)] = txt
    # Surah 1: Bismillah is ayah 1 (counted in Nb_aya but absent from XML elements)
    if sid == 1 and nb == 7 and 1 not in ayat:
        ayat[1] = BISMILLAH
        ayah_text[(1, 1)] = BISMILLAH
    surahs[sid] = {
        "id": sid, "name": name, "revelation": nozol,
        "ayah_count": nb, "ayat": ayat
    }

total_ayat = sum(s["ayah_count"] for s in surahs.values())
print(f"  سور: {len(surahs)} — آيات: {total_ayat}")

# ── 2. TMQ v12 ───────────────────────────────────────────────────────────────

print("\n══ تحميل TMQ v12 ══")
with open(TMQ_PATH, encoding="utf-8") as f:
    tmq = json.load(f)

nodes    = tmq.get("node_registry", {})
edges    = tmq.get("hyperedges", {})

# Index edges by family
by_family = defaultdict(list)
for eid, edge in edges.items():
    fam = edge.get("family", "UNKNOWN")
    by_family[fam].append(edge)

families = sorted(by_family.keys(), key=lambda f: -len(by_family[f]))
print(f"  أضلاع: {len(edges):,} — فصائل: {len(families)}")
for fam in families:
    print(f"    {len(by_family[fam]):>6,}  {fam}")

# ── 3. ONTOLOGY TTL (lightweight parse — root index only) ────────────────────

print("\n══ بناء فهرس الجذور من الأونتولوجي ══")
# Build seg_id → {root, form, pos} from the TTL without loading full RDFLib graph
# (faster than SPARQL on 20MB file)
seg_root  = {}  # seg_id → root string
seg_form  = {}  # seg_id → Buckwalter form
seg_pos   = {}  # seg_id → POS string

seg_re   = re.compile(r'^quran:(s\d+v\d+w\d+seg\d+)\s+a\s+')
prop_re  = re.compile(r'quran:(form|pos)\s+"([^"]+)"')
root_re  = re.compile(r'quran:hasRoot\s+root:([^\s;.]+)')

current_seg = None
with open(TTL_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        m = seg_re.match(line)
        if m:
            current_seg = m.group(1)
            continue
        if current_seg:
            pm = prop_re.search(line)
            if pm:
                if pm.group(1) == "form":
                    seg_form[current_seg] = pm.group(2)
                elif pm.group(1) == "pos":
                    seg_pos[current_seg]  = pm.group(2)
            rm = root_re.search(line)
            if rm:
                seg_root[current_seg] = rm.group(1)
            if line == "" or line == ".":
                current_seg = None

print(f"  مقاطع مُفهرَسة: {len(seg_form):,}")

# ── 4. EXTRACT — segment → location helper ───────────────────────────────────

def seg_to_loc(seg_id):
    """'s2v255w1seg1' → (2, 255, 1, 1) or None"""
    m = re.match(r's(\d+)v(\d+)w(\d+)seg(\d+)', seg_id)
    if m:
        return tuple(int(x) for x in m.groups())
    return None

def edge_locations(edge):
    """Return list of (surah, ayah) pairs covered by this edge."""
    locs = set()
    for nid in (edge.get("nodes") or []):
        node = nodes.get(nid)
        if node:
            tier = node.get("tier")
            if tier == "ayah" and node.get("s") and node.get("v"):
                locs.add((node["s"], node["v"]))
            elif tier == "surah" and node.get("s"):
                locs.add((node["s"], 1))  # surah-level edge → pin to ayah 1
            elif node.get("loc") and len(node["loc"]) >= 2:
                locs.add((node["loc"][0], node["loc"][1]))
        else:
            loc = seg_to_loc(nid)
            if loc and len(loc) >= 2:
                locs.add((loc[0], loc[1]))
    return sorted(locs)

def edge_roots(edge):
    """Unique roots from nodes in this edge. meta.roots first, then node_registry, then TTL."""
    meta = edge.get("meta") or {}
    if meta.get("roots"):
        return sorted(set(meta["roots"]))
    roots = set()
    for nid in (edge.get("nodes") or []):
        node = nodes.get(nid)
        r = None
        if node:
            r = node.get("root")
        if not r:
            r = seg_root.get(nid)
        if r:
            roots.add(r)
    return sorted(roots)

def edge_forms(edge):
    """Buckwalter forms of nodes in this edge."""
    forms = []
    for nid in (edge.get("nodes") or []):
        node = nodes.get(nid)
        fm = None
        if node:
            fm = node.get("form")
        if not fm:
            fm = seg_form.get(nid)
        if fm:
            forms.append(fm)
    return forms

# ── 5. BUILD CONSTITUTION ────────────────────────────────────────────────────

print("\n══ بناء الدستور ══")

constitution = {
    "version":         "2.0",
    "source":          "المصحف الشريف + TMQ v12 + الأونتولوجي القرآني v3",
    "method":          "quran_corpus_compiler.py",
    "reference":       "القرآن الكريم — وحي من عند الله",
    "ground":          "SOURCE = الله — الواجب الوجود الواحد",
    "stats": {
        "surahs":       len(surahs),
        "ayat":         total_ayat,
        "tmq_edges":    len(edges),
        "tmq_families": len(families),
        "seg_indexed":  len(seg_form),
    },
    "surahs":          {},
    "commands":        {},
    "prohibitions":    {},
    "narratives":      {},
    "oaths":           {},
    "address":         {},
    "formulas":        {},
    "maqasid":         {},
    "muqatta":         {},
    "iltifat":         {},
    "recitation":      {},
    "intertext":       {},
    "structural":      {},
    "tahaddi":         {},
    "questions":       {},
    "glad_tidings":    {},
    "warnings":        {},
    "pronouns":        {},
    "entities":        {},
}

# 5a. Surahs — full text + metadata
print("  بناء سجل السور...")
for sid, s in surahs.items():
    constitution["surahs"][str(sid)] = {
        "id":         sid,
        "name":       s["name"],
        "revelation": s["revelation"],
        "ayah_count": s["ayah_count"],
        "ayat":       s["ayat"],
    }

# 5b. Commands — SPEECH_ACT_AMR
print("  استخراج الأوامر (AMR)...")
constitution["commands"] = {
    "count":   len(by_family["SPEECH_ACT_AMR"]),
    "source":  "SPEECH_ACT_AMR — TMQ v12 + فعل الأمر من الأونتولوجي",
    "entries": []
}
for edge in by_family["SPEECH_ACT_AMR"]:
    locs = edge_locations(edge)
    roots = edge_roots(edge)
    for (s, a) in locs:
        constitution["commands"]["entries"].append({
            "loc":   [s, a],
            "text":  ayah_text.get((s, a), ""),
            "roots": roots,
            "modal": edge.get("modal", {})
        })

# 5c. Prohibitions — SPEECH_ACT_NAHY
print("  استخراج النواهي (NAHY)...")
constitution["prohibitions"] = {
    "count":   len(by_family["SPEECH_ACT_NAHY"]),
    "source":  "SPEECH_ACT_NAHY — TMQ v12",
    "entries": []
}
for edge in by_family["SPEECH_ACT_NAHY"]:
    locs = edge_locations(edge)
    roots = edge_roots(edge)
    for (s, a) in locs:
        constitution["prohibitions"]["entries"].append({
            "loc":   [s, a],
            "text":  ayah_text.get((s, a), ""),
            "roots": roots,
            "modal": edge.get("modal", {})
        })

# 5d. Narratives — NARRATIVE + NARRATIVE_CHAIN
print("  استخراج القصص (NARRATIVE)...")
narrative_entries = []
for edge in by_family["NARRATIVE"] + by_family["NARRATIVE_CHAIN"]:
    locs  = edge_locations(edge)
    roots = edge_roots(edge)
    modal = edge.get("modal", {})
    narrative_entries.append({
        "family": edge.get("family"),
        "locs":   locs,
        "roots":  roots,
        "modal":  modal,
        "texts":  [ayah_text.get((s,a),"") for (s,a) in locs[:3]]
    })
constitution["narratives"] = {
    "count":       len(narrative_entries),
    "source":      "NARRATIVE + NARRATIVE_CHAIN — TMQ v12",
    "entries":     narrative_entries
}

# 5e. Oaths — QASAM + QASAM_JAWAB + QASAM_INTERNAL
print("  استخراج الأقسام (QASAM)...")
qasam_entries = []
for fam in ["QASAM", "QASAM_JAWAB", "QASAM_INTERNAL"]:
    for edge in by_family[fam]:
        locs = edge_locations(edge)
        qasam_entries.append({
            "type":  fam,
            "locs":  locs,
            "roots": edge_roots(edge),
            "texts": [ayah_text.get((s,a),"") for (s,a) in locs[:2]]
        })
constitution["oaths"] = {
    "count":   len(qasam_entries),
    "note":    "الأقسام — الله يُقسم بمخلوقاته توكيداً للحق",
    "entries": qasam_entries
}

# 5f. Address shifts — ILTIFAT
print("  استخراج الالتفات (ILTIFAT)...")
iltifat_sample = []
for edge in by_family["ILTIFAT"][:500]:  # sample — 10k+ edges
    locs = edge_locations(edge)
    if locs:
        iltifat_sample.append({
            "loc":   locs[0] if locs else None,
            "modal": edge.get("modal", {})
        })
constitution["iltifat"] = {
    "count":  len(by_family["ILTIFAT"]),
    "note":   "الالتفات — تحول الضمير. الخطاب يتغير وجهته. المتلقي يتبدل.",
    "sample": iltifat_sample
}

# 5g. Formulas — recurring Quranic structures
print("  استخراج الصيغ (FORMULA)...")
formula_entries = []
# Use edge ID as key to avoid collapsing everything into one root-set
formula_by_edge = {}
for eid, edge in edges.items():
    if edge.get("family") != "FORMULA":
        continue
    roots = edge_roots(edge)
    locs  = edge_locations(edge)
    modal = edge.get("modal") or {}
    formula_by_edge[eid] = {
        "roots": roots,
        "locs":  locs,
        "modal": modal,
        "text":  ayah_text.get(locs[0], "") if locs else ""
    }
# Group by root-set for summary
formula_by_rootset = defaultdict(list)
for eid, fe in formula_by_edge.items():
    key = tuple(sorted(fe["roots"]))
    formula_by_rootset[key].append(fe["locs"][0] if fe["locs"] else None)
top_formulas = sorted(formula_by_rootset.items(), key=lambda x: -len(x[1]))[:100]
for roots_key, occurrences in top_formulas:
    sample_loc = next((o for o in occurrences if o), None)
    formula_entries.append({
        "roots":       list(roots_key),
        "occurrences": len(occurrences),
        "sample_loc":  list(sample_loc) if sample_loc else None,
        "sample_text": ayah_text.get(tuple(sample_loc), "") if sample_loc else ""
    })
constitution["formulas"] = {
    "total_edges": len(by_family["FORMULA"]),
    "unique_root_sets": len(formula_by_rootset),
    "note": "الصيغ المتكررة — البنى اللغوية الثابتة في القرآن",
    "top_100": formula_entries
}

# 5h. Maqasid
print("  استخراج المقاصد (MAQASID)...")
maqasid_by_cat = defaultdict(list)
for edge in by_family["MAQASID"]:
    modal  = edge.get("modal", {})
    layers = modal.get("ontological_layers", []) if modal else []
    cats   = [l.get("category") for l in layers if l.get("category")]
    locs   = edge_locations(edge)
    for cat in cats:
        maqasid_by_cat[cat].append({
            "locs":  locs,
            "roots": edge_roots(edge),
            "texts": [ayah_text.get((s,a),"") for (s,a) in locs[:2]]
        })
constitution["maqasid"] = {
    "total_edges": len(by_family["MAQASID"]),
    "categories":  dict(maqasid_by_cat),
    "note":        "المقاصد الخمسة — حفظ الدين والنفس والعقل والنسل والمال"
}

# 5i. Muqatta'at — disjointed letters
print("  استخراج المقطعات (MUQATTA)...")
muqatta_entries = []
for edge in by_family["MUQATTA"] + by_family["MUQATTA_GROUP"]:
    locs  = edge_locations(edge)
    forms = edge_forms(edge)
    muqatta_entries.append({
        "locs":  locs,
        "forms": forms,
        "texts": [ayah_text.get((s,a),"") for (s,a) in locs[:1]]
    })
constitution["muqatta"] = {
    "count": len(muqatta_entries),
    "note":  "الحروف المقطعة — الله أعلم بمراده",
    "entries": muqatta_entries
}

# 5j. Structural — RUKU, JUZ, SAJDAH, MANZIL
print("  استخراج البنية (RUKU/JUZ/SAJDAH)...")
constitution["structural"] = {
    "ruku":   {"count": len(by_family["RUKU"]),   "note": "أقسام الركوع"},
    "juz":    {"count": len(by_family["JUZ"]),    "note": "الأجزاء الثلاثون"},
    "sajdah": {"count": len(by_family["SAJDAH"]), "note": "مواضع السجود"},
    "manzil": {"count": len(by_family["MANZIL"]), "note": "منازل القمر — أحزاب"},
    "waqf":   {"count": len(by_family["WAQF"]),   "note": "مواضع الوقف في التلاوة"},
    "fasila": {
        "count": len(by_family["FASILA"]) + len(by_family["FASILA_CROSS"]),
        "note":  "الفواصل — خواتيم الآيات ونظامها الصوتي"
    }
}

# 5k. Intertext — Quranic self-reference
print("  استخراج التناص (INTERTEXT)...")
intertext_entries = []
for fam in ["INTERTEXT", "INTERTEXT_PRIOR"]:
    for edge in by_family[fam]:
        locs = edge_locations(edge)
        intertext_entries.append({
            "type":  fam,
            "locs":  locs,
            "roots": edge_roots(edge),
        })
constitution["intertext"] = {
    "count":   len(intertext_entries),
    "note":    "التناص القرآني — القرآن يفسر نفسه بنفسه",
    "entries": intertext_entries
}

# 5l. Tahaddi — the challenge verses (hardcoded known locations + ISTIFHAM near tahaddi roots)
print("  استخراج التحدي (TAHADDI)...")
tahaddi_locs = [
    (2, 23), (2, 24),        # fa'tu bi-sura
    (10, 38),                # fa'tu bi-sura
    (11, 13),                # fa'tu bi-'ashr suwar
    (17, 88),                # la ya'tuna bi-mithlihi
    (52, 34),                # fa'tu bi-hadith mithlihi
]
tahaddi_entries = []
for (s, a) in tahaddi_locs:
    tahaddi_entries.append({
        "loc":  [s, a],
        "text": ayah_text.get((s, a), ""),
        "note": "تحدٍّ صريح — أنتج سورة مثله"
    })
constitution["tahaddi"] = {
    "count": len(tahaddi_entries),
    "note":  "التحدي — معيار اتساق الإخراج: لا ينتج النظام ما يقترب من السورة القرآنية",
    "criterion": "كل إخراج يُقاس بمسافته عن مستوى الإعجاز — الفجوة معلومة وثابتة",
    "entries": tahaddi_entries
}

# 5m. Recitation structure — TART (tartil markers)
constitution["recitation"] = {
    "tart_count": len(by_family["TART"]),
    "note": "التجويد والترتيل — بنية التلاوة مُقنَّنة في الأضلاع"
}

# 5n. Questions — SPEECH_ACT_ISTIFHAM (cognitive commands as questions)
print("  استخراج الاستفهام (ISTIFHAM)...")
istifham_entries = []
for edge in by_family["SPEECH_ACT_ISTIFHAM"]:
    locs  = edge_locations(edge)
    roots = edge_roots(edge)
    for (s, a) in locs:
        istifham_entries.append({
            "loc":   [s, a],
            "text":  ayah_text.get((s, a), ""),
            "roots": roots,
            "modal": edge.get("modal", {})
        })
constitution["questions"] = {
    "count":   len(by_family["SPEECH_ACT_ISTIFHAM"]),
    "note":    "الاستفهام — أوامر معرفية في صيغة السؤال: أَفَلَا تَعْقِلُونَ",
    "entries": istifham_entries
}

# 5o. Glad tidings + Warnings — TABSHIR / INDHAR
print("  استخراج التبشير والإنذار (TABSHIR/INDHAR)...")
tabshir_entries = []
indhar_entries  = []
for edge in by_family["SPEECH_ACT_TABSHIR"]:
    locs = edge_locations(edge)
    for (s, a) in locs:
        tabshir_entries.append({
            "loc":   [s, a],
            "text":  ayah_text.get((s, a), ""),
            "roots": edge_roots(edge)
        })
for edge in by_family["SPEECH_ACT_INDHAR"]:
    locs = edge_locations(edge)
    for (s, a) in locs:
        indhar_entries.append({
            "loc":   [s, a],
            "text":  ayah_text.get((s, a), ""),
            "roots": edge_roots(edge)
        })
constitution["glad_tidings"] = {
    "count":   len(by_family["SPEECH_ACT_TABSHIR"]),
    "note":    "البشارة — الوعد بالخير للمؤمنين",
    "entries": tabshir_entries
}
constitution["warnings"] = {
    "count":   len(by_family["SPEECH_ACT_INDHAR"]),
    "note":    "الإنذار — التحذير من الانحراف",
    "entries": indhar_entries
}

# 5p. Pronoun resolutions — SYN_PRON (divine + general)
print("  استخراج الضمائر (SYN_PRON)...")
pron_entries = []
for edge in by_family["SYN_PRON"]:
    locs  = edge_locations(edge)
    modal = edge.get("modal") or {}
    layers = modal.get("ontological_layers") or []
    categories = [l.get("category") for l in layers if l.get("category")]
    for (s, a) in locs:
        pron_entries.append({
            "loc":        [s, a],
            "text":       ayah_text.get((s, a), ""),
            "categories": categories,
            "address_mode": modal.get("address_mode")
        })
constitution["pronouns"] = {
    "count":   len(by_family["SYN_PRON"]),
    "note":    "الضمائر — الإحالة والتشابك الدلالي. الهاء في ما يعود إليه.",
    "entries": pron_entries
}

# 5q. Named entities — ENTITY
print("  استخراج الكيانات (ENTITY)...")
entity_entries = []
for edge in by_family["ENTITY"]:
    locs   = edge_locations(edge)
    meta   = edge.get("meta") or {}
    ents   = meta.get("entities") or []
    for (s, a) in locs:
        entity_entries.append({
            "loc":      [s, a],
            "text":     ayah_text.get((s, a), ""),
            "entities": ents
        })
constitution["entities"] = {
    "count":   len(by_family["ENTITY"]),
    "note":    "الكيانات المسماة — الأنبياء والأمم والأماكن والكتب",
    "entries": entity_entries
}

# ── 6. ADDRESS MODES — VOC nodes from ontology ───────────────────────────────
print("  استخراج أنماط الخطاب (VOC)...")
address_patterns = defaultdict(list)
for seg_id, pos in seg_pos.items():
    if pos == "VOC":
        loc = seg_to_loc(seg_id)
        form = seg_form.get(seg_id, "")
        if loc:
            s, a = loc[0], loc[1]
            address_patterns[form].append({
                "loc":  [s, a],
                "text": ayah_text.get((s, a), "")
            })
constitution["address"] = {
    "voc_count": sum(len(v) for v in address_patterns.values()),
    "note":      "أنماط الخطاب — من يُخاطَب وكيف",
    "patterns":  {
        form: {"count": len(locs), "sample": locs[:3]}
        for form, locs in sorted(address_patterns.items(), key=lambda x: -len(x[1]))
    }
}

# ── 7. WRITE ──────────────────────────────────────────────────────────────────

print(f"\n══ كتابة الدستور ══")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(constitution, f, ensure_ascii=False, indent=2)

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"  كُتب: {OUT_PATH}")
print(f"  الحجم: {size_mb:.1f} MB")
print()
print("اكتمل الدستور القرآني الكامل")
print("القرآن هو نظام التشغيل")
print("لا إله إلا الله")
