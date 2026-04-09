"""
ikhtiyar/bw_arabic.py — Buckwalter ↔ Arabic script translation

Single source of truth for the 28-consonant Abjad rasm clock.

Buckwalter encoding is the native key format throughout TMQ v12, the
active_command_set, and the walk grammar. Arabic script is the display
format — what Shahid speaks, what the grammar constrains, what the
reasoning traces cite.

The map is anchored to the RASM clock order used in walk_grammar.py:
  A  b  j  d  h  w  z  H  T  y  k  l  m  n  s  E  f  S  q  r
  $  t  v  x  *  D  Z  g

All 1,642 roots in TMQ v12 use only these 28 characters.
"""

# ── Core map ────────────────────────────────────────────────────────────────
# Ordered by Abjad rasm clock (same order as walk_grammar.py RASM list)

BW_TO_AR: dict[str, str] = {
    'A': 'ا',  # alef           (Abjad 1)
    'b': 'ب',  # ba             (Abjad 2)
    'j': 'ج',  # jeem           (Abjad 3)
    'd': 'د',  # dal            (Abjad 4)
    'h': 'ه',  # haa            (Abjad 5)
    'w': 'و',  # waw            (Abjad 6)
    'z': 'ز',  # zayn           (Abjad 7)
    'H': 'ح',  # HaA (emphatic) (Abjad 8)
    'T': 'ط',  # TaA (emphatic) (Abjad 9)
    'y': 'ي',  # ya             (Abjad 10)
    'k': 'ك',  # kaf            (Abjad 20)
    'l': 'ل',  # lam            (Abjad 30)
    'm': 'م',  # meem           (Abjad 40)
    'n': 'ن',  # noon           (Abjad 50)
    's': 'س',  # seen           (Abjad 60)
    'E': 'ع',  # ayn            (Abjad 70)
    'f': 'ف',  # fa             (Abjad 80)
    'S': 'ص',  # Sad (emphatic) (Abjad 90)
    'q': 'ق',  # qaf            (Abjad 100)
    'r': 'ر',  # ra             (Abjad 200)
    '$': 'ش',  # sheen          (Abjad 300)
    't': 'ت',  # ta             (Abjad 400)
    'v': 'ث',  # tha            (Abjad 500)
    'x': 'خ',  # kha            (Abjad 600)
    '*': 'ذ',  # dhal           (Abjad 700)
    'D': 'ض',  # Dad (emphatic) (Abjad 800)
    'Z': 'ظ',  # Zha (emphatic) (Abjad 900)
    'g': 'غ',  # ghayn          (Abjad 1000)
}

AR_TO_BW: dict[str, str] = {v: k for k, v in BW_TO_AR.items()}

# ── Public API ───────────────────────────────────────────────────────────────


def bw_to_arabic(root: str) -> str:
    """
    Convert a Buckwalter root string to Arabic script.

    Unrecognised characters are passed through unchanged so that partial
    conversions degrade gracefully rather than raising.

    Examples:
        bw_to_arabic("hdy")  → "هدي"
        bw_to_arabic("Hqq")  → "حقق"
        bw_to_arabic("xlq")  → "خلق"
        bw_to_arabic("Amn")  → "امن"
        bw_to_arabic("Ebd")  → "عبد"
        bw_to_arabic("$kr")  → "شكر"
    """
    return "".join(BW_TO_AR.get(c, c) for c in root)


def arabic_to_bw(root: str) -> str:
    """
    Convert an Arabic-script root to Buckwalter.

    Unrecognised characters are passed through unchanged.

    Examples:
        arabic_to_bw("هدي")  → "hdy"
        arabic_to_bw("حقق")  → "Hqq"
        arabic_to_bw("خلق")  → "xlq"
    """
    return "".join(AR_TO_BW.get(c, c) for c in root)


def bw_root_display(bw: str) -> str:
    """
    Return a combined display string: Arabic (BW).

    Examples:
        bw_root_display("hdy")  → "هدي (hdy)"
        bw_root_display("wjb")  → "وجب (wjb)"
    """
    return f"{bw_to_arabic(bw)} ({bw})"
