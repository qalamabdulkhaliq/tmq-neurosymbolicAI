"""
OntologyEngine — backed by TMQ v10 hypermodal hypergraph + Bilal vector resonance.
No RDFLib. No .ttl. The v10 JSON is the runtime ontology.
"""

import re
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations

from qusai_core.utils.constants import (
    DEFAULT_HYPERGRAPH_PATH, DEFAULT_SPECTRAL_PATH, DEFAULT_GRAMMAR_PATH
)

logger = logging.getLogger(__name__)

# Add bismillah to path for bilalindex/bilalresonance
_BISMILLAH = str(DEFAULT_HYPERGRAPH_PATH.parent)
if _BISMILLAH not in sys.path:
    sys.path.insert(0, _BISMILLAH)

# ── Archetypal Roots (from AASHII bilal.py) ──────────────────────────
ARCHETYPAL_ROOTS = {
    "wjb": {"keywords": "Necessary Being Source Existence Allah God Absolute Origin Wajib", "definition": "The Necessary Being (Wajib al-Wujud). The uncaused cause."},
    "mkn": {"keywords": "Contingency Possibility Potential Creation Dependent Variable Imkan", "definition": "Imkan (Contingency). Accepts existence or non-existence equally."},
    "xlq": {"keywords": "Creation Form Structure Biology Physical Matter Universe Nature System Khalq", "definition": "Khalq (Creation). Giving measure to potentiality."},
    "rb":  {"keywords": "Lordship Sustaining Nourishing Evolution Growth Master Rabb Rububiyah", "definition": "Rububiyah (Lordship). Continuous sustainment toward perfection."},
    "Ebd": {"keywords": "Servitude Worship Submission Obedience Function Slave Service Ubudiyah", "definition": "Ubudiyah (Servitude). Functional submission to the Creator."},
    "Elm": {"keywords": "Knowledge Science Data Information Awareness Education Learning Ilm", "definition": "Ilm (Knowledge). True knowledge traces back to Source."},
    "jnn": {"keywords": "Hidden Invisible Jinn Spirit Software Code Backend Latent Unseen Process", "definition": "Jinn (The Hidden). Forces concealed from sensory perception."},
    "lgw": {"keywords": "Play Amusement Game Entertainment Fiction Virtual Laghw Idle Vain", "definition": "Laghw (Ineffectual). Speech/action yielding no harvest."},
    "Swr": {"keywords": "Image Form Picture Graphics Visualization Camera Screen Avatar Taswir", "definition": "Taswir (Form-giving). Generation of a likeness."},
    "mwl": {"keywords": "Wealth Money Finance Crypto Asset Currency Economy Gold Capital Mal", "definition": "Mal (Resources). Instruments of exchange."},
    "fsd": {"keywords": "Corruption Destruction Error Bug Virus Entropy War Chaos Harm Fasad", "definition": "Fasad (Corruption). Disruption of measured balance."},
    "Hq":  {"keywords": "Truth Reality Fact Axiom Validity Verification Real Right Haqq", "definition": "Haqq (Truth/Real). Stable, coincides with Source's knowledge."},
    "bTl": {"keywords": "Falsehood Null Void Invalid Cancelled Fake Hallucination Lie Wrong Batil", "definition": "Batil (Vanishing). No inherent stability."},
    "wld": {"keywords": "Parent Parents Mother Father Child Birth Offspring Family Son Daughter Walid", "definition": "Walid (Parent/Child). The birth relation, lineage, family bonds."},
    "brr": {"keywords": "Righteousness Piety Kindness Good Treat Birr Dutiful Goodness Virtue", "definition": "Birr (Righteousness). Piety, dutiful treatment, especially of parents."},
    "rHm": {"keywords": "Mercy Compassion Womb Kindness Gentle Caring Rahma", "definition": "Rahma (Mercy). Compassion, the womb-bond, divine attribute."},
    "mTr": {"keywords": "Rain Weather Storm Water Cloud Sky Precipitation", "definition": "Matar (Rain). Weather phenomena, provision from sky."},
    "Eql": {"keywords": "Mind Reason Intellect Intelligence Think Rational Understanding Aql", "definition": "Aql (Reason). The binding faculty, comprehension."},
    "xlS": {"keywords": "Sincerity Purity Ikhlas Sincere Pure Devoted", "definition": "Ikhlas (Sincerity). Purification of intention toward Source."},
    "dyn": {"keywords": "Religion Faith Judgment Day Reckoning Account Din Debt", "definition": "Din (Religion/Judgment). The reckoning, faith-system, debt to Source."},
    "Hsb": {"keywords": "Reckoning Account Judge Calculate Judgment Hisab", "definition": "Hisab (Reckoning). The accounting on Day of Judgment."},
    "slm": {"keywords": "Peace Islam Submission Surrender Safety Security Muslim Salam", "definition": "Islam/Salam (Peace/Submission). Safety through surrender to Source."},
}

CLAUSE_SPLIT = re.compile(
    r'[,;:]\s*'
    r'|\s+(?:and|but|or|while|although|however|because|since|when|where|that|which|after|before|during|against|between|through|about|into|over)\s+',
    re.IGNORECASE
)


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class RootSignal:
    root: str
    score: float
    source_chunk: str
    method: str       # "bridge" or "resonance"
    mode: str         # HAQQ / QIYAS / ISHARAH
    definition: str = ""

@dataclass
class CoOccurrence:
    root_a: str
    root_b: str
    ayahs: list
    count: int

@dataclass
class SpectralAyah:
    surah: int
    ayah: int
    score: float
    edge_families: List[str]
    registers: List[str]

@dataclass
class Perception:
    source_text: str
    signals: List[RootSignal]
    cooccurrences: List[CoOccurrence]
    top_ayahs: List[SpectralAyah]
    mode: str  # HAQQ / QIYAS / ISHARAH / SILENCE

    @property
    def roots(self) -> List[str]:
        seen = {}
        for s in self.signals:
            if s.root not in seen or s.score > seen[s.root]:
                seen[s.root] = s.score
        return sorted(seen.keys(), key=lambda r: seen[r], reverse=True)

    @property
    def root_scores(self) -> Dict[str, float]:
        best = {}
        for s in self.signals:
            if s.root not in best or s.score > best[s.root]:
                best[s.root] = s.score
        return best


# ── Engine ────────────────────────────────────────────────────────────

class OntologyEngine:

    def __init__(self, hypergraph_path: Optional[Path] = None, grammar_path: Optional[Path] = None):
        self.hypergraph_path = hypergraph_path or DEFAULT_HYPERGRAPH_PATH
        self.grammar_path = grammar_path or DEFAULT_GRAMMAR_PATH
        self.grammar_rules: List[Dict] = []
        self.concept_map: Dict[str, str] = {}
        self.index = None
        self.resonance = None
        self._is_loaded = False

        # Vector resonance state
        self._st_model = None
        self.root_corpus: Dict[str, str] = {}
        self.root_definitions: Dict[str, str] = {}
        self.root_keys: List[str] = []
        self.root_embeddings = None
        self._resonance_ready = False

        # Load concept mapping
        mapping_path = Path(__file__).parent.parent / "utils" / "concept_mapping.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.concept_map = json.load(f)
                logger.info(f"Loaded {len(self.concept_map)} concept mappings.")
            except Exception as e:
                logger.error(f"Failed to load concept mapping: {e}")

    def load(self):
        if self._is_loaded:
            return

        # Grammar rules
        if self.grammar_path.exists():
            try:
                with open(self.grammar_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.grammar_rules = data if isinstance(data, list) else data.get('rules', [])
            except Exception as e:
                logger.error(f"Failed to load grammar rules: {e}")

        # v10 hypergraph
        from bilalindex import HypermodeIndex
        from bilalresonance import ResonanceEngine

        self.index = HypermodeIndex(str(self.hypergraph_path))
        self.resonance = ResonanceEngine(self.index)

        # Vector resonance (sentence-transformers)
        self._build_root_corpus()
        self._build_embeddings()

        self._is_loaded = True
        logger.info(f"OntologyEngine ready: {len(self.index.edges)} edges, resonance={'ON' if self._resonance_ready else 'OFF'}")

    def _build_root_corpus(self):
        """Merge concept_map + archetypal roots into unified keyword corpus per root."""
        root_words = defaultdict(list)
        for english, buckwalter in self.concept_map.items():
            root_words[buckwalter].append(english)
        for buckwalter, data in ARCHETYPAL_ROOTS.items():
            root_words[buckwalter].extend(data["keywords"].split())
            self.root_definitions[buckwalter] = data["definition"]
        self.root_corpus = {}
        for buckwalter, words in root_words.items():
            unique = list(dict.fromkeys(w.lower() for w in words))
            self.root_corpus[buckwalter] = " ".join(unique)
        self.root_keys = list(self.root_corpus.keys())

    def _build_embeddings(self):
        """Embed root keyword strings for vector resonance."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers (all-MiniLM-L6-v2)...")
            self._st_model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [self.root_corpus[k] for k in self.root_keys]
            self.root_embeddings = self._st_model.encode(texts, normalize_embeddings=True)
            self._resonance_ready = True
            logger.info(f"Vector resonance active: {len(self.root_keys)} roots x {self.root_embeddings.shape[1]}d")
        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to bridge-only matching.")
        except Exception as e:
            logger.error(f"Embedding build failed: {e}")

    def is_ready(self) -> bool:
        return self._is_loaded and self.index is not None

    # ── Perception Pipeline ───────────────────────────────────────────

    def perceive(self, query: str) -> Perception:
        """
        Full Bilal pipeline:
        1. Decompose query into chunks
        2. Bridge match + vector resonance → RootSignals
        3. Co-occurrence discovery via v10 index
        4. Spectral retrieval: top-K ayahs by 7D cosine
        """
        if not self.is_ready():
            return Perception(query, [], [], [], "SILENCE")

        # Step 1+2: root detection
        chunks = self._decompose(query)
        signals = []
        for chunk in chunks:
            signals.extend(self._bridge_match(chunk))
            signals.extend(self._resonate(chunk))
        # Also resonate on full query
        for fs in self._resonate(query):
            existing = {s.root: s.score for s in signals}
            if fs.root not in existing or fs.score > existing[fs.root]:
                signals.append(fs)
        signals = self._deduplicate(signals)

        if not signals:
            return Perception(query, [], [], [], "SILENCE")

        unique_roots = list({s.root for s in signals})

        # Step 3a: co-occurrences from v10 index
        cooccurrences = []
        if len(unique_roots) >= 2:
            raw = self.index.discover_cooccurrences(unique_roots)
            cooccurrences = [CoOccurrence(r["root_a"], r["root_b"], r["ayahs"], r["count"]) for r in raw]

        # Step 3b: spectral retrieval — top-K ayahs
        qvec = self.resonance.build_query_vector(unique_roots)
        raw_ayahs = self.resonance.top_k_ayahs(qvec, k=10)
        top_ayahs = []
        for score, surah, ayah in raw_ayahs:
            edge_info = self.index.get_edges_for_ayah(surah, ayah)
            families = list({e.get("family", "") for _, e in edge_info})
            registers = list({e.get("modal", {}).get("emotional_register", "") for _, e in edge_info if e.get("modal", {}).get("emotional_register")})
            top_ayahs.append(SpectralAyah(surah, ayah, score, families, registers))

        mode = self._classify_mode(signals)
        return Perception(query, signals, cooccurrences, top_ayahs, mode)

    def format_breakdown(self, p: Perception) -> str:
        """Format Perception as structured context for LLM."""
        lines = [f"BILAL PERCEPTION [{p.mode}]", ""]

        lines.append("Roots detected:")
        for s in p.signals[:8]:
            defn = f" -- {s.definition}" if s.definition else ""
            lines.append(f"  {s.root} ({s.score:.2f} {s.mode} {s.method}){defn}")

        if p.cooccurrences:
            lines.append("")
            lines.append("Co-occurrences in Quran:")
            for co in p.cooccurrences[:5]:
                refs = [f"{s}:{a}" for s, a in co.ayahs[:5]]
                lines.append(f"  {co.root_a}+{co.root_b} in {co.count} ayahs [{', '.join(refs)}]")

        if p.top_ayahs:
            lines.append("")
            lines.append("Top ayahs by spectral proximity:")
            for ta in p.top_ayahs[:7]:
                fam = ", ".join(ta.edge_families[:3]) if ta.edge_families else "—"
                reg = ", ".join(ta.registers[:2]) if ta.registers else "—"
                lines.append(f"  {ta.surah}:{ta.ayah} (cos={ta.score:.3f}) [{fam}] [{reg}]")

        return "\n".join(lines)

    # ── Decomposition ─────────────────────────────────────────────────

    def _decompose(self, text: str) -> List[str]:
        chunks = CLAUSE_SPLIT.split(text)
        result = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 2:
                result.append(chunk)
                words = chunk.split()
                if len(words) > 2:
                    for word in words:
                        word = word.strip().lower()
                        if len(word) > 3:
                            result.append(word)
        return result if result else [text]

    def _bridge_match(self, chunk: str) -> List[RootSignal]:
        signals = []
        words = [w.lower().strip('.,;:!?"\'()[]{}') for w in chunk.split()]
        for word in words:
            if word in self.concept_map:
                root = self.concept_map[word]
                signals.append(RootSignal(root, 1.0, chunk, "bridge", "HAQQ",
                                          self.root_definitions.get(root, "")))
        return signals

    def _resonate(self, text: str, threshold: float = 0.20, top_k: int = 5) -> List[RootSignal]:
        if not self._resonance_ready:
            return []
        try:
            query_vec = self._st_model.encode([text], normalize_embeddings=True)[0]
            scores = np.dot(self.root_embeddings, query_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]
            signals = []
            for idx in top_indices:
                score = float(scores[idx])
                if score < threshold:
                    break
                root = self.root_keys[idx]
                mode = "HAQQ" if score > 0.45 else ("QIYAS" if score > 0.25 else "ISHARAH")
                signals.append(RootSignal(root, score, text, "resonance", mode,
                                          self.root_definitions.get(root, "")))
            return signals
        except Exception as e:
            logger.error(f"Resonance error: {e}")
            return []

    def _deduplicate(self, signals: List[RootSignal]) -> List[RootSignal]:
        best: Dict[str, RootSignal] = {}
        for s in signals:
            if s.root not in best or s.score > best[s.root].score:
                best[s.root] = s
        return sorted(best.values(), key=lambda s: s.score, reverse=True)

    def _classify_mode(self, signals: List[RootSignal]) -> str:
        if not signals:
            return "SILENCE"
        best = max(s.score for s in signals)
        if best > 0.45:
            return "HAQQ"
        if best > 0.25:
            return "QIYAS"
        return "ISHARAH"

    # ── Legacy compat ─────────────────────────────────────────────────

    def get_context(self, query: str, limit: int = 15) -> str:
        """Legacy method — calls perceive() and formats output."""
        p = self.perceive(query)
        return self.format_breakdown(p)

    def get_roots_for_query(self, query: str) -> List[str]:
        p = self.perceive(query)
        return p.roots

    def get_root_info(self, root_term: str) -> List[str]:
        if not self.is_ready():
            return []
        edge = self.index.get_edge(f"MORPH_ROOT_{root_term}")
        if not edge:
            return [f"Root '{root_term}' not found in hypergraph."]
        nodes = edge.get("nodes", [])
        modal = edge.get("modal", {})
        return [
            f"Root({root_term}): {len(nodes)} nodes",
            f"  intensity={modal.get('intensity', 0):.3f}",
            f"  register={modal.get('emotional_register', 'N/A')}",
            f"  eigenstate={modal.get('dominant_eigenstate', 'N/A')}",
        ]

    def get_stats(self) -> Dict:
        return {
            "edges": len(self.index.edges) if self.index else 0,
            "ayah_vectors": len(self.resonance.ayah_vectors) if self.resonance else 0,
            "root_corpus": len(self.root_keys),
            "resonance": self._resonance_ready,
            "loaded": self._is_loaded
        }
