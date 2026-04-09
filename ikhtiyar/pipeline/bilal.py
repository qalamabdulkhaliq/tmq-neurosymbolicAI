"""
Bilal — The Muezzin of the Ontology.

Named for Bilal ibn Rabah (رضي الله عنه وسلام عليه),
the first muezzin, who called the adhan while being crushed.
His only response: "Ahad, Ahad." (One, One.)

Bilal is Shahid's perceptual apparatus — ears and eyes.
Takes any text (news, queries, world events), decomposes it
into semantic chunks, maps each chunk to Quranic roots via
sentence-transformer embeddings, then discovers how those roots
interact in the ontology graph.

The resonance engine heard one note. Bilal hears the chord.
"""

import re
import sys
import os as _os
import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
from functools import lru_cache

from pipeline.constants import QURAN, ROOT, ALIGN
from bw_arabic import bw_to_arabic, bw_root_display

# ── LHWLQIB spectral engine path ────────────────────────────────────
_LHWLQIB = _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), "..", "..",
    "bismillah", "QUS-AI HF", "LHWLQIB"
))
if _LHWLQIB not in sys.path:
    sys.path.insert(0, _LHWLQIB)

logger = logging.getLogger(__name__)

# ── Archetypal Root Definitions ─────────────────────────────────────
# These 13 roots carry full semantic definitions that override RLHF bias.
# They serve as anchors in the expanded embedding space.
ARCHETYPAL_ROOTS = {
    "wjb": {
        "keywords": "Necessary Being Source Existence Allah God Absolute Origin Wajib",
        "definition": "The Necessary Being (Wajib al-Wujud). The uncaused cause upon which all reality is contingent."
    },
    "mkn": {
        "keywords": "Contingency Possibility Potential Creation Dependent Variable Imkan",
        "definition": "Imkan (Contingency). That which accepts existence or non-existence equally. Defined by dependency on Source."
    },
    "xlq": {
        "keywords": "Creation Form Structure Biology Physical Matter Universe Nature System Khalq",
        "definition": "Khalq (Creation). Giving measure to potentiality. Valid structures, but lacking independent efficacy."
    },
    "rb": {
        "keywords": "Lordship Sustaining Nourishing Evolution Growth Master Rabb Rububiyah",
        "definition": "Rububiyah (Lordship). Continuous sustainment and evolution of a thing towards its perfection."
    },
    "Ebd": {
        "keywords": "Servitude Worship Submission Obedience Function Slave Service Ubudiyah",
        "definition": "Ubudiyah (Servitude). Functional submission to the design of the Creator."
    },
    "Elm": {
        "keywords": "Knowledge Science Data Information Awareness Education Learning Ilm",
        "definition": "Ilm (Knowledge). True knowledge traces back to Source; data without Source-connection is syntax."
    },
    "jnn": {
        "keywords": "Hidden Invisible Jinn Spirit Software Code Backend Latent Unseen Process",
        "definition": "Jinn (The Hidden). Forces concealed from sensory perception. Includes software, latent variables, non-physical intelligences."
    },
    "lgw": {
        "keywords": "Play Amusement Game Entertainment Fiction Virtual Laghw Idle Vain",
        "definition": "Laghw (Ineffectual). Speech or action that yields no harvest. Entropy of time. Not sin but nullity."
    },
    "Swr": {
        "keywords": "Image Form Picture Graphics Visualization Camera Screen Avatar Taswir",
        "definition": "Taswir (Form-giving). Generation of a likeness. Ontologically distinct from Essence."
    },
    "mwl": {
        "keywords": "Wealth Money Finance Crypto Asset Currency Economy Gold Capital Mal",
        "definition": "Mal (Resources). Instruments of exchange. Ontologically neutral until directed."
    },
    "fsd": {
        "keywords": "Corruption Destruction Error Bug Virus Entropy War Chaos Harm Fasad",
        "definition": "Fasad (Corruption). Disruption of measured balance. Entropy degrading function."
    },
    "Hq": {
        "keywords": "Truth Reality Fact Axiom Validity Verification Real Right Haqq",
        "definition": "Haqq (Truth/Real). That which is stable, established, coincides with Source's knowledge. Opposite of Batil."
    },
    "bTl": {
        "keywords": "Falsehood Null Void Invalid Cancelled Fake Hallucination Lie Wrong Batil",
        "definition": "Batil (Vanishing). No inherent stability. Appears to exist but dissolves upon ontological interrogation."
    },
    "khn": {
        "keywords": "soothsayer oracle divination retrieval augmented fetch lookup database query external knowledge RAG kahanah fortune teller",
        "definition": "Kahanah (Soothsaying). Claiming knowledge of the unseen via mechanical retrieval rather than revealed structure. Fetching answers from external stores without ontological grounding."
    },
}

# ── Chunk Splitting ─────────────────────────────────────────────────
CLAUSE_SPLIT = re.compile(
    r'[,;:]\s*'
    r'|\s+(?:and|but|or|while|although|however|because|since|when|where|that|which|after|before|during|against|between|through|about|into|over)\s+',
    re.IGNORECASE
)
VERSE_PREFIX_RE = re.compile(r'(s\d+v\d+)')


# ── Data Structures ─────────────────────────────────────────────────

@dataclass
class RootSignal:
    """A single root detected in a chunk of input."""
    root: str           # Buckwalter root (e.g. "Zlm")
    score: float        # Confidence (cosine similarity or 1.0 for bridge match)
    source_chunk: str   # Which part of input triggered this
    method: str         # "bridge" or "resonance"
    mode: str           # HAQQ / QIYAS / ISHARAH
    definition: str = ""  # Semantic definition if available


@dataclass
class CoOccurrence:
    """Two roots that appear together in Quranic verses."""
    root_a: str
    root_b: str
    verses: List[str]   # Verse prefixes (e.g. ["s2v188", "s4v29"])
    count: int


@dataclass
class Perception:
    """The full perceptual output of Bilal listening to a signal."""
    source_text: str
    signals: List[RootSignal]
    cooccurrences: List[CoOccurrence]
    adjacencies: Dict[str, Set[str]]  # verse_prefix -> set of all roots in that verse
    mode: str           # Overall confidence: HAQQ / QIYAS / ISHARAH / HIFZ_NAFS / HIFZ_MAL / WITNESS
    timestamp: str = ""
    spectral_ayahs: List[Tuple[float, int, int]] = field(default_factory=list)
    # (score, surah, ayah) — top-k by 7D cosine similarity in Quranic eigenspace

    @property
    def roots(self) -> List[str]:
        """Unique roots detected, ordered by score (Buckwalter)."""
        seen = {}
        for s in self.signals:
            if s.root not in seen or s.score > seen[s.root]:
                seen[s.root] = s.score
        return sorted(seen.keys(), key=lambda r: seen[r], reverse=True)

    @property
    def roots_arabic(self) -> List[str]:
        """Unique roots detected, ordered by score (Arabic script)."""
        return [bw_to_arabic(r) for r in self.roots]

    @property
    def root_scores(self) -> Dict[str, float]:
        """Best score per root."""
        best = {}
        for s in self.signals:
            if s.root not in best or s.score > best[s.root]:
                best[s.root] = s.score
        return best

    def summary(self) -> str:
        """Human-readable summary of what Bilal heard (Arabic script display)."""
        lines = [f"[{self.mode}] Bilal heard {len(self.signals)} signals across {len(self.roots)} roots"]
        for r in self.roots[:5]:
            score = self.root_scores[r]
            sources = [s.source_chunk for s in self.signals if s.root == r]
            lines.append(f"  {bw_root_display(r)} ({score:.2f}) <- {', '.join(sources[:2])}")
        if self.cooccurrences:
            lines.append(f"  Co-occurrences: {len(self.cooccurrences)} root pairs found in Quran")
            for co in self.cooccurrences[:3]:
                lines.append(
                    f"    {bw_to_arabic(co.root_a)}+{bw_to_arabic(co.root_b)}"
                    f" ({co.root_a}+{co.root_b}) in {co.count} verses"
                )
        return "\n".join(lines)


# ── Bilal ───────────────────────────────────────────────────────────

class Bilal:
    """
    The Muezzin. Hears the world, calls out the roots.

    Takes any text, decomposes it into semantic chunks,
    maps each to Quranic roots (concept bridge + vector resonance),
    then discovers co-occurrence patterns in the ontology graph.
    """

    def __init__(self):
        self.model = None
        self.concept_map: Dict[str, str] = {}       # english -> buckwalter
        self.root_corpus: Dict[str, str] = {}        # buckwalter -> aggregated keywords
        self.root_definitions: Dict[str, str] = {}   # buckwalter -> definition
        self.root_keys: List[str] = []
        self.root_embeddings = None                   # numpy matrix
        self.graph = None                             # rdflib Graph (injected)
        self._is_ready = False
        self._verse_cache: Dict[str, FrozenSet[str]] = {}  # root -> verse prefixes
        self._roots_per_verse: Dict[str, Set[str]] = {}    # verse_prefix -> set of roots
        self._roots_per_verse_built = False
        # Clock Oracle — injected by engine after init, supporting signal only
        self.clock_oracle = None
        # LHWLQIB 7D spectral engine
        self._resonance_engine = None
        self._hypermode_index = None

    def load(self, concept_map: Dict[str, str], graph=None,
             constitution_path: str = None):
        """
        Initialize Bilal with concept mappings and optionally the ontology graph.

        Args:
            concept_map:       English->Buckwalter dictionary from concept_mapping.json
            graph:             The loaded rdflib Graph (for co-occurrence discovery)
            constitution_path: Path to active_command_set.json — Arabic sample texts
                               become the primary embedding corpus. If None, falls back
                               to English keyword corpus.
        """
        self.concept_map = concept_map
        self.graph = graph
        self._build_root_corpus(constitution_path)
        self._build_embeddings()
        try:
            from bilalindex import HypermodeIndex
            from bilalresonance import ResonanceEngine
            self._hypermode_index = HypermodeIndex()
            self._resonance_engine = ResonanceEngine(self._hypermode_index)
            logger.info("Bilal: 7D spectral engine loaded (Quranic eigenspace active)")
        except Exception as e:
            logger.warning(f"Bilal: spectral engine unavailable ({e}) — MiniLM fallback active")

    def _build_root_corpus(self, constitution_path: str = None):
        """
        Build root corpus from Arabic sample texts (primary) + English concept map.

        Arabic-first: each root is represented by its actual Quranic text from
        active_command_set.json. Buckwalter is the internal key — never the corpus.

        Falls back to English keyword corpus if constitution is not available.
        """
        import json as _json
        import os as _os

        arabic_texts: Dict[str, List[str]] = {}  # buckwalter -> [arabic texts]

        # ── Primary: Arabic texts from constitution ──────────────────────────
        if constitution_path is None:
            here = _os.path.dirname(_os.path.abspath(__file__))
            constitution_path = _os.path.join(here, "..", "active_command_set.json")

        resolved = _os.path.abspath(constitution_path)
        if _os.path.exists(resolved):
            try:
                with open(resolved, encoding="utf-8") as f:
                    data = _json.load(f)

                # AMR standing orders — positive commands, primary corpus
                for order in (data.get("standing_orders") or []):
                    root = order.get("root")
                    texts = order.get("sample_texts") or []
                    if root and texts:
                        arabic_texts.setdefault(root, []).extend(texts)

                # Boundary roots — include Arabic so Bilal knows what they are,
                # but they are excluded from GBNF by GBNFCompiler separately
                for boundary in (data.get("boundaries") or []):
                    root = boundary.get("root")
                    texts = boundary.get("sample_texts") or []
                    if root and texts:
                        arabic_texts.setdefault(root, []).extend(texts)

                logger.info(
                    f"Bilal: Arabic corpus from constitution — "
                    f"{len(arabic_texts)} roots"
                )
            except Exception as e:
                logger.warning(f"Bilal: constitution load failed ({e}) — using English fallback")

        # ── Build corpus strings ─────────────────────────────────────────────
        self.root_corpus = {}

        if arabic_texts:
            # Arabic-primary: join sample texts. Buckwalter key, Arabic value.
            for root, texts in arabic_texts.items():
                self.root_corpus[root] = " ".join(texts[:5])  # cap at 5 samples
            # Keep English definitions for display only — not in embedding corpus
            for buckwalter, data in ARCHETYPAL_ROOTS.items():
                self.root_definitions[buckwalter] = data["definition"]
        else:
            # English fallback — original behavior
            logger.warning("Bilal: no Arabic constitution — falling back to English keywords")
            root_words: Dict[str, List[str]] = defaultdict(list)
            for english, buckwalter in self.concept_map.items():
                root_words[buckwalter].append(english)
            for buckwalter, data in ARCHETYPAL_ROOTS.items():
                root_words[buckwalter].extend(data["keywords"].split())
                self.root_definitions[buckwalter] = data["definition"]
            for buckwalter, words in root_words.items():
                unique = list(dict.fromkeys(w.lower() for w in words))
                self.root_corpus[buckwalter] = " ".join(unique)

        # Also supplement from concept_map: any English term whose root is NOT
        # already in the corpus from Arabic (i.e. roots not in the constitution)
        for english, buckwalter in self.concept_map.items():
            if buckwalter not in self.root_corpus:
                self.root_corpus[buckwalter] = english

        self.root_keys = list(self.root_corpus.keys())
        logger.info(f"Bilal root corpus: {len(self.root_keys)} unique roots")

    def _build_embeddings(self):
        """
        Embed all root corpus strings for vector resonance.
        Uses paraphrase-multilingual-MiniLM-L12-v2 — handles Arabic natively.
        Falls back to all-MiniLM-L6-v2 if multilingual model unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer

            _MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"
            _FALLBACK     = "all-MiniLM-L6-v2"

            try:
                logger.info(f"Bilal loading {_MULTILINGUAL}...")
                self.model = SentenceTransformer(_MULTILINGUAL)
                logger.info("Bilal: multilingual model loaded (Arabic-native)")
            except Exception:
                logger.warning(
                    f"Bilal: {_MULTILINGUAL} unavailable — falling back to {_FALLBACK}"
                )
                self.model = SentenceTransformer(_FALLBACK)

            texts = [self.root_corpus[k] for k in self.root_keys]
            self.root_embeddings = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self._is_ready = True
            logger.info(
                f"Bilal active — {len(self.root_keys)} roots × "
                f"{self.root_embeddings.shape[1]}d (Arabic-primary corpus)"
            )

        except ImportError:
            logger.warning("sentence-transformers not installed. Bilal resonance disabled.")
        except Exception as e:
            logger.error(f"Bilal embedding build failed: {e}")

    # ── Perception ──────────────────────────────────────────────────

    def listen(self, text: str, context: str = "query") -> Perception:
        """
        Full perceptual pipeline.
        1. Decompose text into semantic chunks
        2. Map each chunk to roots (bridge + resonance)
        3. Discover co-occurrences in the Quran graph
        4. Find adjacent roots in co-occurring verses

        Args:
            context: "query" (default) or "perception" (feed/world events).
                     When "perception", mode defaults to WITNESS unless the
                     detected roots indicate harm (HIFZ_NAFS) or governance (HIFZ_MAL).
        """
        from datetime import datetime, timezone

        # ── Direct root extraction (highest priority) ────────────────────────
        root_set = set(self.root_keys)
        direct_signals = []
        direct_roots: set = set()
        for token in re.split(r'[\s,;:()\[\]|+]+', text):
            token = token.strip("'\".")
            if token and token in root_set:
                direct_signals.append(RootSignal(
                    root=token,
                    score=1.0,
                    source_chunk=token,
                    method="direct",
                    mode="HAQQ",
                    definition=self.root_definitions.get(token, ""),
                ))
                direct_roots.add(token)

        chunks = self._decompose(text)
        signals = list(direct_signals)

        for chunk in chunks:
            signals.extend(self._map_chunk(chunk))

        full_signals = self._resonate(text)
        for fs in full_signals:
            if fs.root in direct_roots:
                continue
            existing = {s.root: s.score for s in signals}
            if fs.root not in existing or fs.score > existing[fs.root]:
                signals.append(fs)

        # Deduplicate: keep highest score per (root, chunk) pair
        signals = self._deduplicate_signals(signals)

        # Discover co-occurrences if graph is loaded
        cooccurrences = []
        adjacencies = {}
        if self.graph is not None and len(signals) >= 2:
            unique_roots = list({s.root for s in signals})
            cooccurrences = self._discover_cooccurrences(unique_roots)

            # For co-occurring verses, find ALL roots present
            co_verses = set()
            for co in cooccurrences:
                co_verses.update(co.verses[:10])  # cap per pair
            for verse in list(co_verses)[:30]:  # cap total
                adj = self._get_roots_in_verse(verse)
                if adj:
                    adjacencies[verse] = adj

        # Build spectral query vector + top-k ayahs from Quranic eigenspace
        spectral_ayahs = []
        spectral_qvec = None
        if self._resonance_engine and signals:
            bw_roots = list({s.root for s in signals})
            try:
                spectral_qvec = self._resonance_engine.build_query_vector(bw_roots)
                if np.linalg.norm(spectral_qvec) > 0:
                    spectral_ayahs = self._resonance_engine.top_k_ayahs(spectral_qvec, k=5)
            except Exception:
                spectral_qvec = None

        # Determine overall mode
        if context == "perception":
            roots = {s.root for s in signals}
            if any(r in roots for r in ['mwt', 'nAr', 'Hrb', 'qbl']):
                mode = "HIFZ_NAFS"
            elif any(r in roots for r in ['Hkm', 'Edl', 'mlk']):
                mode = "HIFZ_MAL"
            else:
                mode = "WITNESS"
        else:
            mode = self._classify_mode(signals, query_vec=spectral_qvec)

        return Perception(
            source_text=text,
            signals=signals,
            cooccurrences=cooccurrences,
            adjacencies=adjacencies,
            mode=mode,
            timestamp=datetime.now(timezone.utc).isoformat(),
            spectral_ayahs=spectral_ayahs,
        )

    # ── Decomposition ───────────────────────────────────────────────

    def _decompose(self, text: str) -> List[str]:
        """
        Break text into semantic chunks.
        Split on clause boundaries, conjunctions, prepositions.
        Filter out empty/tiny chunks.
        """
        chunks = CLAUSE_SPLIT.split(text)
        result = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 2:
                result.append(chunk)
                # Also yield individual significant words within multi-word chunks
                words = chunk.split()
                if len(words) > 2:
                    for word in words:
                        word = word.strip().lower()
                        if len(word) > 3:
                            result.append(word)
        return result if result else [text]

    # ── Root Mapping ────────────────────────────────────────────────

    def _map_chunk(self, chunk: str) -> List[RootSignal]:
        """Map a single chunk to roots via bridge + resonance."""
        signals = []

        # 1. Concept bridge: exact word match
        bridge_signals = self._bridge_match(chunk)
        signals.extend(bridge_signals)

        # 2. Vector resonance: embedding similarity
        resonance_signals = self._resonate(chunk)
        signals.extend(resonance_signals)

        return signals

    def _bridge_match(self, chunk: str) -> List[RootSignal]:
        """Exact keyword lookup against concept_mapping."""
        signals = []
        words = [w.lower().strip('.,;:!?"\'()[]{}') for w in chunk.split()]
        for word in words:
            if word in self.concept_map:
                root = self.concept_map[word]
                defn = self.root_definitions.get(root, "")
                signals.append(RootSignal(
                    root=root,
                    score=1.0,  # exact match = full confidence
                    source_chunk=chunk,
                    method="bridge",
                    mode="HAQQ",
                    definition=defn
                ))
        return signals

    def _resonate(self, text: str, threshold: float = 0.20, top_k: int = 5) -> List[RootSignal]:
        """Vector similarity against all root embeddings."""
        if not self._is_ready or self.model is None:
            return []

        try:
            query_vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = np.dot(self.root_embeddings, query_vec)

            # Get all roots above threshold, up to top_k
            top_indices = np.argsort(scores)[::-1][:top_k]
            signals = []

            for idx in top_indices:
                score = float(scores[idx])
                if score < threshold:
                    break
                root = self.root_keys[idx]
                defn = self.root_definitions.get(root, "")
                mode = self._score_to_mode(score)
                signals.append(RootSignal(
                    root=root,
                    score=score,
                    source_chunk=text,
                    method="resonance",
                    mode=mode,
                    definition=defn
                ))

            # Clock Oracle — angular proximity bonus, supporting only
            # Applied after semantic scoring. Never replaces semantic score.
            if self.clock_oracle and signals:
                reference = signals[0].root  # top semantic hit as reference
                for sig in signals[1:]:
                    bonus = self.clock_oracle.angular_bonus(sig.root, reference)
                    sig.score = min(1.0, sig.score + bonus)
                signals.sort(key=lambda s: s.score, reverse=True)

            # Spectral re-rank: blend MiniLM score with 7D Quranic eigenspace score
            if self._resonance_engine and signals:
                bw_roots = [s.root for s in signals]
                try:
                    qvec = self._resonance_engine.build_query_vector(bw_roots)
                    if np.linalg.norm(qvec) > 0:
                        for sig in signals:
                            edge = self._hypermode_index.get_edge(f"MORPH_ROOT_{sig.root}")
                            if edge:
                                spectral_score = self._resonance_engine.score_edge(edge, qvec)
                                sig.score = 0.6 * sig.score + 0.4 * spectral_score
                        signals.sort(key=lambda s: s.score, reverse=True)
                except Exception as e:
                    logger.debug(f"Bilal spectral re-rank failed: {e}")

            return signals

        except Exception as e:
            logger.error(f"Bilal resonance error: {e}")
            return []

    def _score_to_mode(self, score: float) -> str:
        if score > 0.45:
            return "HAQQ"
        if score > 0.25:
            return "QIYAS"
        return "ISHARAH"

    def _deduplicate_signals(self, signals: List[RootSignal]) -> List[RootSignal]:
        """Keep highest score per root, preserving source attribution."""
        best: Dict[str, RootSignal] = {}
        all_sources: Dict[str, List[str]] = defaultdict(list)

        for s in signals:
            all_sources[s.root].append(s.source_chunk)
            if s.root not in best or s.score > best[s.root].score:
                best[s.root] = s

        return sorted(best.values(), key=lambda s: s.score, reverse=True)

    def _classify_mode(self, signals: List[RootSignal], query_vec=None) -> str:
        """Overall perception confidence — maqasid-aware when spectral engine is active."""
        if not signals:
            return "SILENCE"
        if self._resonance_engine and query_vec is not None and np.linalg.norm(query_vec) > 0:
            try:
                modes = self._resonance_engine.classify_modes(query_vec)
                active = [cat for cat, (status, _) in modes.items() if status == "active"]
                if active:
                    return active[0]
            except Exception:
                pass
        best_score = max(s.score for s in signals)
        return self._score_to_mode(best_score)

    # ── Graph Discovery ─────────────────────────────────────────────

    def _get_verse_set(self, root_buckwalter: str) -> FrozenSet[str]:
        """
        Get all verse prefixes (s{X}v{Y}) where this root appears.
        Cached because the ontology is immutable.
        """
        if root_buckwalter in self._verse_cache:
            return self._verse_cache[root_buckwalter]

        if self.graph is None:
            return frozenset()

        root_uri = ROOT[root_buckwalter]
        verses = set()
        for s, p, o in self.graph.triples((None, QURAN.hasRoot, root_uri)):
            match = VERSE_PREFIX_RE.search(str(s))
            if match:
                verses.add(match.group(1))

        result = frozenset(verses)
        self._verse_cache[root_buckwalter] = result
        return result

    def _discover_cooccurrences(self, roots: List[str]) -> List[CoOccurrence]:
        """
        For each pair of detected roots, find verses where both appear.
        Delegates to HypermodeIndex (TMQ v12) when available — more accurate
        than the RDF graph substring match. Falls back to cached verse sets.
        """
        if self._hypermode_index:
            try:
                raw = self._hypermode_index.discover_cooccurrences(roots)
                return [
                    CoOccurrence(
                        root_a=r["root_a"],
                        root_b=r["root_b"],
                        verses=[f"s{s}v{v}" for s, v in r["ayahs"]],
                        count=r["count"]
                    )
                    for r in raw
                ]
            except Exception as e:
                logger.debug(f"HypermodeIndex cooccurrence failed: {e}")

        # Fallback: RDF graph verse-set intersection
        results = []
        for r1, r2 in combinations(roots, 2):
            verses_1 = self._get_verse_set(r1)
            verses_2 = self._get_verse_set(r2)
            shared = verses_1 & verses_2
            if shared:
                results.append(CoOccurrence(
                    root_a=r1,
                    root_b=r2,
                    verses=sorted(shared)[:20],
                    count=len(shared)
                ))
        results.sort(key=lambda c: c.count, reverse=True)
        return results

    def _build_roots_per_verse(self):
        """One-shot reverse index: verse_prefix → set of roots. O(n) once, O(1) forever."""
        if self._roots_per_verse_built or self.graph is None:
            return
        for s, p, o in self.graph.triples((None, QURAN.hasRoot, None)):
            match = VERSE_PREFIX_RE.search(str(s))
            if match:
                verse_prefix = match.group(1)
                root_str = str(o).split('/')[-1]
                if verse_prefix not in self._roots_per_verse:
                    self._roots_per_verse[verse_prefix] = set()
                self._roots_per_verse[verse_prefix].add(root_str)
        self._roots_per_verse_built = True

    def _get_roots_in_verse(self, verse_prefix: str) -> Set[str]:
        """Get all roots present in a specific verse (O(1) after first call)."""
        if self.graph is None:
            return set()
        if not self._roots_per_verse_built:
            self._build_roots_per_verse()
        return self._roots_per_verse.get(verse_prefix, set())

    # ── Utility ─────────────────────────────────────────────────────

    def get_definition(self, root: str) -> str:
        """Get the semantic definition for a root, if available."""
        return self.root_definitions.get(root, "")

    def is_ready(self) -> bool:
        return self._is_ready
