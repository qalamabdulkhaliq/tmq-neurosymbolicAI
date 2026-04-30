import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.epistemic_clusters import get_cluster, get_cluster_uri
from core.quran_rdf_compiler import parse_qac_line, VALID_TAGS

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


# ── QAC Parser Tests ────────────────────────────────────────────────────────────

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


# ── Buckwalter→Arabic Conversion Tests ──────────────────────────────────────

def test_buckwalter_root_conversion():
    from core.quran_rdf_compiler import to_arabic_root
    assert to_arabic_root("Elm") == "علم"
    assert to_arabic_root("rHm") == "رحم"
    assert to_arabic_root("mlk") == "ملك"


# ── Numeral Lexicon Tests ───────────────────────────────────────────────────

def test_numeral_root_detected():
    from core.quran_rdf_compiler import NUMERAL_ROOTS
    assert "وحد" in NUMERAL_ROOTS  # واحد family
    assert "ثلث" in NUMERAL_ROOTS  # ثلاثة family


def test_non_numeral_not_in_lexicon():
    from core.quran_rdf_compiler import NUMERAL_ROOTS
    assert "علم" not in NUMERAL_ROOTS
