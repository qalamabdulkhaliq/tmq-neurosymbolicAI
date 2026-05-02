"""
RDFCircuit — replaces StaticCircuit for QS.ttl-based signal propagation.

Exposes:
  .evaluate(roots: list[str]) -> PropagationResult
  ._root_index: dict[str, list]   <- shim for procedure.py compatibility
  ._frames: list[dict]            <- shim for procedure.py compatibility
  ._ayat_spans: list[tuple]       <- shim for procedure.py compatibility
  ._pagerank: dict[str, float]    <- shim for procedure.py compatibility
"""

from __future__ import annotations

import logging
import sys
import os
from collections import defaultdict
from pathlib import Path

from rdflib import ConjunctiveGraph, URIRef, Literal
from rdflib.namespace import Namespace, RDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bw_arabic import bw_to_arabic

logger = logging.getLogger(__name__)

ROOT_NS  = Namespace("http://quran.data/root/")
WORD_NS  = Namespace("http://quran.data/word/")
AYAH_NS  = Namespace("http://quran.data/ayah/")
POS_NS   = Namespace("http://quran.data/pos/")
FORM_NS  = Namespace("http://quran.data/form/")
MORPH_NS = Namespace("http://quran.data/morph/")
EPI_NS   = Namespace("http://quran.data/epistemic/")
QS_NS    = Namespace("http://quran.data/")

_CLUSTER_TIER = {
    "certainty":   ("HAQQ",  0.85),
    "command":     ("HAQQ",  0.80),
    "prohibition": ("HAQQ",  0.80),
    "narrative":   ("QIYAS", 0.65),
    "seeking":     ("QIYAS", 0.60),
    "description": ("QIYAS", 0.55),
    "conjecture":  ("QIYAS", 0.35),
}

_ACTIVATION_SPARQL = """\
PREFIX root: <http://quran.data/root/>
PREFIX pos:  <http://quran.data/pos/>
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

        self._root_index: dict[str, list] = defaultdict(list)
        self._frames: list[dict] = []
        self._ayat_spans: list[tuple] = []
        self._pagerank: dict[str, float] = {}
        self._build_shims()

    def _build_shims(self) -> None:
        logger.info("RDFCircuit: building procedure.py shims...")

        # word URI -> loc string
        loc_by_word: dict[str, str] = {}
        q_loc = """
        PREFIX qs: <http://quran.data/>
        SELECT ?word ?loc WHERE { ?word qs:loc ?loc . }
        """
        for row in self._g.query(q_loc):
            loc_by_word[str(row.word)] = str(row.loc)

        # Build frame list ordered by (s,v,w)
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

        # Build _ayat_spans
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

        # Build _root_index: arabic_root -> list of frame indices
        word_to_idx = {f["_word_uri"]: i for i, f in enumerate(self._frames)}
        q_roots = """
        PREFIX qs: <http://quran.data/>
        SELECT ?root ?word WHERE {
            GRAPH ?g { ?root ?tag ?word . }
            FILTER(STRSTARTS(STR(?root), "http://quran.data/root/"))
        } LIMIT 500000
        """
        for row in self._g.query(q_roots):
            root_str = str(row.root)
            arabic_root = root_str[len("http://quran.data/root/"):]
            idx = word_to_idx.get(str(row.word))
            if idx is not None:
                self._root_index[arabic_root].append(idx)

        # _pagerank: normalized root frequency
        q_freq = """
        PREFIX qs: <http://quran.data/>
        SELECT ?root ?freq WHERE {
            ?root a qs:Root .
            ?root qs:frequency ?freq .
        }
        """
        freq_map: dict[str, int] = {}
        for row in self._g.query(q_freq):
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
        Activate Arabic-script (or Buckwalter) roots against QS.ttl.
        Returns PropagationResult compatible with vtransistor.CircuitEvaluator.
        """
        from core.vtransistor import PropagationResult

        # Normalize BW input transparently
        arabic_roots = []
        for r in roots:
            if r and not any(ord(c) > 0x05FF for c in r):
                arabic_roots.append(bw_to_arabic(r))
            else:
                arabic_roots.append(r)
        arabic_roots = [r for r in arabic_roots if r]

        _empty = PropagationResult(
            activated_frames={}, activated_roots={},
            procedure_tags=[], ayat_refs=[],
            tier="WAQF", confidence=0.0,
        )
        if not arabic_roots:
            return _empty

        values = " ".join(f"<{ROOT_NS[r]}>" for r in arabic_roots)
        sparql = _ACTIVATION_SPARQL.replace("%VALUES%", values)

        rows = list(self._g.query(sparql))
        if not rows:
            return _empty

        clusters_found: list[str] = []
        ayat_refs: list[str] = []
        proof_tree: list[dict] = []
        activated_roots: dict[str, float] = {}

        for row in rows:
            root_str = str(row.rootUri)
            if root_str.startswith("http://quran.data/root/"):
                ar = root_str[len("http://quran.data/root/"):]
                activated_roots[ar] = max(activated_roots.get(ar, 0.0), 1.0)

            cluster_str = str(row.cluster) if row.cluster else ""
            if cluster_str.startswith("http://quran.data/epistemic/"):
                c = cluster_str[len("http://quran.data/epistemic/"):]
                if c not in clusters_found:
                    clusters_found.append(c)

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
                    "loc": str(row.loc) if row.loc else "",
                    "ayah": ayah_str,
                })

        # Tier/confidence from highest-priority cluster
        tier, confidence = "QIYAS", 0.5
        for cluster in ("certainty", "command", "prohibition",
                         "narrative", "seeking", "description", "conjecture"):
            if cluster in clusters_found:
                tier, confidence = _CLUSTER_TIER[cluster]
                break

        confidence = min(1.0, confidence + 0.02 * (len(activated_roots) - 1))

        return PropagationResult(
            activated_frames={},
            activated_roots=activated_roots,
            procedure_tags=[],
            ayat_refs=ayat_refs,
            tier=tier,
            confidence=confidence,
            proof_tree=proof_tree,
        )
