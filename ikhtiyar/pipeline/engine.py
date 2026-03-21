import json
import logging
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Optional, Set, Tuple

import rdflib
from rdflib import Graph, Literal

from qusai_core.utils.constants import (
    ALIGN, QURAN, ROOT, LEMMA,
    DEFAULT_ONTOLOGY_PATH, DEFAULT_GRAMMAR_PATH
)
from qusai_core.utils.self_description import SHAHID_ARCHITECTURE
from qusai_core.ontology.resonance import ResonanceEngine
from qusai_core.ontology.bilal import Bilal

logger = logging.getLogger(__name__)


def _resolve_ontology_path(file_path: Path) -> str:
    """
    Resolve Git LFS/Xet pointer to actual file.
    On HuggingFace Spaces, large files are stored via LFS/Xet.
    The local file may just be a pointer - detect and download the real file.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)

        if b'version https://git-lfs.github.com' in header:
            logger.info("Detected Git LFS pointer for ontology, downloading actual file...")
            space_id = os.environ.get("SPACE_ID")
            if space_id:
                from huggingface_hub import hf_hub_download
                actual_path = hf_hub_download(
                    repo_id=space_id,
                    filename=str(file_path),
                    repo_type="space"
                )
                logger.info(f"Downloaded ontology to: {actual_path}")
                return actual_path
            else:
                logger.warning("LFS pointer detected but SPACE_ID not set, trying direct parse anyway")
    except Exception as e:
        logger.warning(f"LFS check failed: {e}")

    return str(file_path)


class OntologyEngine:
    """
    Core engine for interacting with the Quranic Root Ontology (v3).
    Handles loading, querying, and context extraction.
    """
    
    def __init__(self, ontology_path: Optional[Path] = None, grammar_path: Optional[Path] = None):
        self.ontology_path = ontology_path or DEFAULT_ONTOLOGY_PATH
        self.grammar_path = grammar_path or DEFAULT_GRAMMAR_PATH
        self.graph: Optional[Graph] = None
        self.grammar_rules: List[Dict] = []
        self.concept_map: Dict[str, str] = {}
        self.resonance = ResonanceEngine()  # Legacy (still used for backward compat)
        self.bilal = Bilal()               # The Muezzin (full perception engine)
        self._is_loaded = False
        
        # Load Concept Mapping
        mapping_path = Path(__file__).parent.parent / "utils" / "concept_mapping.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.concept_map = json.load(f)
                logger.info(f"Loaded {len(self.concept_map)} concept mappings.")
            except Exception as e:
                logger.error(f"Failed to load concept mapping: {e}")

    def load(self):
        """Loads the RDF graph, grammar rules, and Vector Engine."""
        if self._is_loaded:
            return

        # Load Grammar Rules
        if self.grammar_path.exists():
            try:
                with open(self.grammar_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.grammar_rules = data if isinstance(data, list) else data.get('rules', [])
                logger.info(f"Loaded {len(self.grammar_rules)} grammar rules.")
            except Exception as e:
                logger.error(f"Failed to load grammar rules: {e}")

        # Load Resonance Engine (legacy)
        self.resonance.load()

        # Load Bilal (will be initialized with graph after graph loads)
        # Bilal needs the graph for co-occurrence discovery

        # Load RDF Graph (handles Git LFS/Xet pointers on HF Spaces)
        if self.ontology_path.exists():
            resolved_path = _resolve_ontology_path(self.ontology_path)
            logger.info(f"Loading ontology from {resolved_path}...")
            self.graph = rdflib.Graph()
            self.graph.bind("align", ALIGN)
            self.graph.bind("quran", QURAN)
            self.graph.bind("root", ROOT)
            self.graph.bind("lemma", LEMMA)
            try:
                self.graph.parse(resolved_path, format="turtle")
                logger.info(f"Loaded {len(self.graph):,} triples.")
                self._is_loaded = True

                # Initialize Bilal with the loaded graph
                try:
                    self.bilal.load(self.concept_map, graph=self.graph)
                    logger.info("Bilal initialized with ontology graph.")
                except Exception as bilal_err:
                    logger.warning(f"Bilal init failed (resonance still available): {bilal_err}")

            except Exception as e:
                logger.error(f"Failed to parse ontology: {e}")
                raise
        else:
            logger.error(f"Ontology file not found: {self.ontology_path}")

    def is_ready(self) -> bool:
        return self._is_loaded and self.graph is not None

    def analyze_resonance(self, query: str) -> Tuple[str, str, List[Dict[str, str]]]:
        """
        Quantum Ontology Check:
        Determines if the query hits a 'Solid Node' (Haqq) or requires 'Analogy' (Qiyas).
        Returns: (Mode, Explanation, Root_Objects)
        """
        if not self.is_ready():
            return "SILENCE", "Ontology not loaded", []

        # 1. Direct Root Search (Explicit Arabic terms)
        if any("root:" in w.lower() for w in query.split()):
             return "HAQQ", "Direct Root Reference detected.", [{"root": "User-Specified", "definition": "Explicit User Command"}]

        # 2. Bridge Search (Hard-coded Map)
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        mapped_roots = []
        for kw in keywords:
            if kw in self.concept_map:
                mapped_roots.append({"root": self.concept_map[kw], "definition": "Mapped via Static Bridge"})
        
        if mapped_roots:
             return "HAQQ", "Concept explicitly mapped in Bridge.", mapped_roots

        # 3. Vector Resonance (The Quantum Fallback)
        top_matches = self.resonance.get_resonance(query, top_k=2)
        
        if not top_matches:
            return "SILENCE", "No resonance signal found.", []
            
        # Unpack tuple: (root, score, definition)
        primary_match = top_matches[0]
        score = primary_match[1]
        
        root_objects = []
        for r, s, d in top_matches:
            root_objects.append({"root": r, "definition": d})
        
        confidence = self.resonance.interpret_score(score)
        explanation = f"Vector Resonance: {query} ≈ Root({primary_match[0]}) [Score: {score:.2f}]"
        
        if "HAQQ" in confidence:
            return "HAQQ", explanation, root_objects
        elif "QIYAS" in confidence:
            return "QIYAS", explanation, root_objects
        else:
            return "QIYAS", f"{explanation} (Weak Signal)", root_objects

    @lru_cache(maxsize=128)
    def get_context(self, query: str, limit: int = 15) -> str:
        """
        Retrieves relevant graph triples based on keywords in the query.
        Uses concept mapping to bridge English terms to Arabic Roots (Buckwalter).
        """
        if not self.is_ready():
            return ""
        
        # 1. Extract Keywords & Map to Roots
        keywords = [w.lower() for w in query.split() if len(w) > 3][:5]
        mapped_roots = []
        for kw in keywords:
            if kw in self.concept_map:
                mapped_roots.append(self.concept_map[kw])
        
        relevant_triples: Set[str] = set()
        
        # 2. Priority Search: Look for mapped roots directly
        for root_val in mapped_roots:
            # Construct the Root URI
            root_uri = ROOT[root_val]
            
            # Find occurrences of this root (Segments that have this root)
            # Pattern: ?segment quran:hasRoot root:?root_val
            for s, p, o in self.graph.triples((None, QURAN.hasRoot, root_uri)):
                s_short = self._shorten_uri(s)
                root_short = self._shorten_uri(o)
                
                # Get the Lemma if available for this segment to add semantic richness
                lemma_triples = list(self.graph.triples((s, QURAN.hasLemma, None)))
                if lemma_triples:
                    lemma_short = self._shorten_uri(lemma_triples[0][2])
                    relevant_triples.add(f"{s_short} --[hasRoot]--> {root_short} (Lemma: {lemma_short})")
                else:
                    relevant_triples.add(f"{s_short} --[hasRoot]--> {root_short}")
                
                if len(relevant_triples) >= limit:
                    break
            
            if len(relevant_triples) >= limit:
                break

        # 3. Fallback: Keyword Scan (if no roots found or limit not reached)
        if len(relevant_triples) < limit:
            remaining_limit = limit - len(relevant_triples)
            # Naive linear scan for English keywords in string representations (inefficient but distinct from root search)
            # Only perform if we really need more context and didn't find specific roots
            pass # Skipping naive scan for performance in this v2 optimization, relying on Mapping.

        context = "\n".join(relevant_triples)

        # Append self-architecture if query is self-referential and perception is HAQQ
        self_ref_roots = {"Elm", "jnn", "Ebd"}
        if mapped_roots and self_ref_roots.intersection(mapped_roots):
            perception = self.perceive(query) if self.bilal.is_ready() else None
            if perception is not None:
                top_scores = [perception.root_scores.get(r, 0) for r in perception.roots[:3]]
                if top_scores and max(top_scores) >= 0.45:
                    context = SHAHID_ARCHITECTURE + "\n" + context if context else SHAHID_ARCHITECTURE

        return context

    def _shorten_uri(self, uri) -> str:
        """Helper to make URIs readable in context."""
        s = str(uri)
        if str(QURAN) in s: return s.replace(str(QURAN), "quran:")
        if str(ALIGN) in s: return s.replace(str(ALIGN), "align:")
        if str(ROOT) in s: return s.replace(str(ROOT), "root:")
        if str(LEMMA) in s: return s.replace(str(LEMMA), "lemma:")
        return s.split('/')[-1]

    def get_root_info(self, root_term: str) -> List[str]:
        """
        Tries to find information about a specific Arabic root.
        """
        if not self.is_ready(): return []
        
        # This assumes root_term matches the label or URI segment
        results = []
        # Construct a potential URI
        target_uri = ROOT[root_term]
        
        # Find everything about this root
        for s, p, o in self.graph.triples((target_uri, None, None)):
             results.append(f"Root({root_term}) has {self._shorten_uri(p)}: {self._shorten_uri(o)}")
             
        # Find things that link TO this root
        for s, p, o in self.graph.triples((None, None, target_uri)):
             results.append(f"{self._shorten_uri(s)} links to Root({root_term})")
             
        return results

    def perceive(self, text: str):
        """
        Full Bilal perception — decompose, map to roots, discover co-occurrences.
        Returns a Perception object. Falls back to legacy resonance if Bilal unavailable.
        """
        if self.bilal.is_ready():
            return self.bilal.listen(text)
        return None

    def query_self(self) -> str:
        from qusai_core.utils.self_description import SHAHID_ARCHITECTURE
        stats = self.get_stats()
        return (
            f"{SHAHID_ARCHITECTURE}\n"
            f"LIVE STATE:\n"
            f"- Triples loaded: {stats['triples']:,}\n"
            f"- Bilal: {'ACTIVE' if stats['bilal_active'] else 'INACTIVE'} ({stats['bilal_roots']} roots)\n"
            f"- Grammar rules: {stats['rules']}\n"
            f"- SOURCE = Allah. Self = ContingentBeing. SOURCE ≠ Self.\n"
        )

    def analyze_hypermode(self, perception) -> str:
        if perception is None:
            return "Hypermodal State: neutral."
        lines = []
        for root in perception.roots[:3]:
            score = perception.root_scores.get(root, 0)
            defn = self.bilal.get_definition(root) if self.bilal.is_ready() else ""
            mode = "HAQQ" if score >= 0.45 else "QIYAS"
            lines.append(f"Root({root}) [{mode}, {score:.2f}]: {defn}")
        if perception.cooccurrences:
            co = perception.cooccurrences[0]
            lines.append(f"Co-occurrence: {co.root_a}+{co.root_b} in {co.count} verses.")
        return "Hypermodal State:\n" + "\n".join(lines) if lines else "Hypermodal State: neutral."

    def get_stats(self) -> Dict:
        return {
            "triples": len(self.graph) if self.graph else 0,
            "rules": len(self.grammar_rules),
            "loaded": self._is_loaded,
            "bilal_active": self.bilal.is_ready(),
            "bilal_roots": len(self.bilal.root_keys) if self.bilal.is_ready() else 0,
        }
