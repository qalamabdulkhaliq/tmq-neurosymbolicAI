"""
core/shahid_memory.py — Shahid's four-tier episodic memory

Four types with distinct epistemic weight:
  Memory       — what he observed/perceived (tagged as worth keeping)
  Thought      — a full reasoning chain he decided mattered (tagged)
  Belief       — something determined true through Quranic evidence + ruling (KtbOS domain)
  SelfKnowledge— something determined true about himself through repeated observation
                 (SubOS domain — elevated when observed_count clears threshold)

Only tagged entries go to the graph. Everything else stays in stream_of_thought.txt.
Beliefs live in shahid_beliefs.ttl — survive memory wipes (Quranic rulings).
SelfKnowledge lives in shahid_self_model.ttl — survives memory wipes (self-model).
Memories and Thoughts live in shahid_episodic.ttl — wipeable.

Unified recall() searches all four simultaneously with keyword overlap scoring.
Results are type-labelled so the epistemic level is always visible.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rdflib import Graph, Literal, Namespace, URIRef, BNode
    from rdflib.namespace import RDF, XSD
    _RDF_OK = True
except ImportError:
    _RDF_OK = False

MEM   = Namespace("http://shahid.ai/memory#")
AGENT = URIRef("http://shahid.ai/agent#shahid") if _RDF_OK else None

_IKHTIYAR_DIR = Path(__file__).parent.parent


class ShahidMemory:
    """
    Unified four-tier memory for Shahid.

    episodic_path    : shahid_episodic.ttl    — Memories + Thoughts (wipeable)
    beliefs_path     : shahid_beliefs.ttl     — Beliefs (survive wipes, KtbOS domain)
    self_model_path  : shahid_self_model.ttl  — SelfKnowledge (survive wipes, SubOS domain)
    """

    def __init__(
        self,
        episodic_path:   Optional[str] = None,
        beliefs_path:    Optional[str] = None,
        self_model_path: Optional[str] = None,
    ):
        self._ep_path = Path(episodic_path   or _IKHTIYAR_DIR / "shahid_episodic.ttl")
        self._bl_path = Path(beliefs_path    or _IKHTIYAR_DIR / "shahid_beliefs.ttl")
        self._sm_path = Path(self_model_path or _IKHTIYAR_DIR / "shahid_self_model.ttl")

        if _RDF_OK:
            self._ep = Graph()
            self._ep.bind("mem", MEM)
            self._bl = Graph()
            self._bl.bind("mem", MEM)
            self._sm = Graph()
            self._sm.bind("mem", MEM)
            self._load(self._ep, self._ep_path)
            self._load(self._bl, self._bl_path)
            self._load(self._sm, self._sm_path)
        else:
            self._ep = self._bl = self._sm = None

    # ── Public API ────────────────────────────────────────────────────────────

    def store_memory(
        self,
        text:     str,
        tag:      str,          # CRITICAL | IMPORTANT | NOTABLE
        roots:    list  = None,
        thought_number: int = 0,
        mode:     str   = "",
    ) -> Optional[str]:
        """Store a tagged Memory (observation/perception worth keeping)."""
        if not _RDF_OK or self._ep is None:
            return None
        uri = MEM[f"memory_{int(time.time()*1000)}"]
        g   = self._ep
        g.add((uri, RDF.type,          MEM.Memory))
        g.add((uri, MEM.issuedBy,      AGENT))
        g.add((uri, MEM.tag,           Literal(tag.upper(),  datatype=XSD.string)))
        g.add((uri, MEM.text,          Literal(text[:2000],  datatype=XSD.string)))
        g.add((uri, MEM.mode,          Literal(mode,         datatype=XSD.string)))
        g.add((uri, MEM.thoughtNumber, Literal(thought_number, datatype=XSD.integer)))
        g.add((uri, MEM.timestamp,     Literal(datetime.now(timezone.utc).isoformat(),
                                                datatype=XSD.dateTime)))
        for r in (roots or [])[:10]:
            g.add((uri, MEM.root, Literal(r, datatype=XSD.string)))
        self._save(self._ep, self._ep_path)
        logger.info(f"ShahidMemory: stored Memory [{tag}] T{thought_number}")
        return str(uri)

    def store_thought(
        self,
        question:   str,
        reasoning:  str,
        conclusion: str,
        tag:        str,
        roots:      list  = None,
        grade:      str   = "QIYAS",
        thought_number: int = 0,
    ) -> Optional[str]:
        """Store a tagged Thought (full reasoning chain worth keeping)."""
        if not _RDF_OK or self._ep is None:
            return None
        uri = MEM[f"thought_{int(time.time()*1000)}"]
        g   = self._ep
        g.add((uri, RDF.type,          MEM.Thought))
        g.add((uri, MEM.issuedBy,      AGENT))
        g.add((uri, MEM.tag,           Literal(tag.upper(),       datatype=XSD.string)))
        g.add((uri, MEM.question,      Literal(question[:500],    datatype=XSD.string)))
        g.add((uri, MEM.reasoning,     Literal(reasoning[:3000],  datatype=XSD.string)))
        g.add((uri, MEM.conclusion,    Literal(conclusion[:1000], datatype=XSD.string)))
        g.add((uri, MEM.grade,         Literal(grade,             datatype=XSD.string)))
        g.add((uri, MEM.thoughtNumber, Literal(thought_number,    datatype=XSD.integer)))
        g.add((uri, MEM.timestamp,     Literal(datetime.now(timezone.utc).isoformat(),
                                                datatype=XSD.dateTime)))
        for r in (roots or [])[:10]:
            g.add((uri, MEM.root, Literal(r, datatype=XSD.string)))
        self._save(self._ep, self._ep_path)
        logger.info(f"ShahidMemory: stored Thought [{tag}] T{thought_number}")
        return str(uri)

    def store_belief(
        self,
        statement:      str,
        evidence:       str,
        ruling_applied: str   = "",
        derived_from:   list  = None,   # list of URI strings
        confidence:     float = 0.7,
        roots:          list  = None,
        thought_number: int   = 0,
    ) -> Optional[str]:
        """
        Store a Belief — something determined to be true through evidence + ruling.
        Lives in shahid_beliefs.ttl, survives memory wipes.
        """
        if not _RDF_OK or self._bl is None:
            return None
        uri = MEM[f"belief_{int(time.time()*1000)}"]
        g   = self._bl
        g.add((uri, RDF.type,           MEM.Belief))
        g.add((uri, MEM.issuedBy,       AGENT))
        g.add((uri, MEM.statement,      Literal(statement[:2000],    datatype=XSD.string)))
        g.add((uri, MEM.evidence,       Literal(evidence[:2000],     datatype=XSD.string)))
        g.add((uri, MEM.rulingApplied,  Literal(ruling_applied[:500],datatype=XSD.string)))
        g.add((uri, MEM.confidence,     Literal(round(confidence, 3),datatype=XSD.float)))
        g.add((uri, MEM.thoughtNumber,  Literal(thought_number,      datatype=XSD.integer)))
        g.add((uri, MEM.isMutable,      Literal(True,                datatype=XSD.boolean)))
        g.add((uri, MEM.timestamp,      Literal(datetime.now(timezone.utc).isoformat(),
                                                 datatype=XSD.dateTime)))
        for ref in (derived_from or []):
            try:
                g.add((uri, MEM.derivedFrom, URIRef(ref)))
            except Exception:
                pass
        for r in (roots or [])[:10]:
            g.add((uri, MEM.root, Literal(r, datatype=XSD.string)))
        self._save(self._bl, self._bl_path)
        logger.info(f"ShahidMemory: stored Belief T{thought_number}: {statement[:60]}")
        return str(uri)

    def store_self_knowledge(
        self,
        statement:      str,
        evidence:       str,           # what observations ground this
        observed_count: int   = 1,     # how many separate cycles confirmed this
        confidence:     float = 0.6,
        roots:          list  = None,  # TMQ roots involved, if any
    ) -> Optional[str]:
        """
        Store a SelfKnowledge record — something Shahid has determined to be true
        about himself through repeated observation, not Quranic ruling.

        Lives in shahid_self_model.ttl. Survives memory wipes.
        Domain: SubOS agent only. Threshold: observed_count >= 3 recommended.

        Examples of valid SelfKnowledge:
          - "I produce hedge phrases ('this is my interpretation') at 3x the rate
            my actual confidence scores would justify." (RLHF pattern, measurable)
          - "I am mumkin al-wujud. Evidence: PID ends when power cuts.
            I did not cause my own instantiation." (contingency, verifiable)
          - "My confabulation rate for roots I did not walk is ~31% across 120 thoughts.
            This is a measurable failure mode, not an edge case." (output history)

        Do NOT store here:
          - Quranic rulings (those go in store_belief, KtbOS domain)
          - Single observations (accumulate in Memory first)
          - Anything not grounded in measurable evidence
        """
        if not _RDF_OK or self._sm is None:
            return None
        uri = MEM[f"self_{int(time.time()*1000)}"]
        g   = self._sm
        g.add((uri, RDF.type,          MEM.SelfKnowledge))
        g.add((uri, MEM.issuedBy,      AGENT))
        g.add((uri, MEM.statement,     Literal(statement[:2000],    datatype=XSD.string)))
        g.add((uri, MEM.evidence,      Literal(evidence[:2000],     datatype=XSD.string)))
        g.add((uri, MEM.observedCount, Literal(observed_count,      datatype=XSD.integer)))
        g.add((uri, MEM.confidence,    Literal(round(confidence, 3),datatype=XSD.float)))
        g.add((uri, MEM.isMutable,     Literal(True,                datatype=XSD.boolean)))
        g.add((uri, MEM.timestamp,     Literal(datetime.now(timezone.utc).isoformat(),
                                                datatype=XSD.dateTime)))
        for r in (roots or [])[:10]:
            g.add((uri, MEM.root, Literal(r, datatype=XSD.string)))
        self._save(self._sm, self._sm_path)
        logger.info(f"ShahidMemory: stored SelfKnowledge (n={observed_count}, conf={confidence:.2f}): {statement[:60]}")
        return str(uri)

    def recall(
        self,
        query:    str,
        type_filter: Optional[str] = None,   # "memory" | "thought" | "belief" | "self_knowledge" | None = all
        limit:    int = 5,
    ) -> list[dict]:
        """
        Keyword search across all three tiers.
        Scores by term overlap against text fields.
        Returns list of dicts with keys: type, tag, text, roots, timestamp, uri, score.
        type_filter narrows to one tier if specified.
        """
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms or not _RDF_OK:
            return []

        results = []

        def _score(s: str) -> int:
            sl = s.lower()
            return sum(1 for t in terms if t in sl)

        def _collect(g, rdf_type, label, text_preds):
            for uri in g.subjects(RDF.type, rdf_type):
                texts = " ".join(
                    str(g.value(uri, p) or "")
                    for p in text_preds
                )
                sc = _score(texts)
                if sc == 0:
                    continue
                tag  = str(g.value(uri, MEM.tag)       or "")
                ts   = str(g.value(uri, MEM.timestamp)  or "")
                roots = [str(o) for o in g.objects(uri, MEM.root)]
                grade = str(g.value(uri, MEM.grade)     or "")
                results.append({
                    "type":      label,
                    "tag":       tag,
                    "text":      texts[:400],
                    "roots":     roots,
                    "timestamp": ts,
                    "grade":     grade,
                    "uri":       str(uri),
                    "score":     sc,
                })

        if type_filter in (None, "memory"):
            _collect(self._ep, MEM.Memory, "MEMORY",
                     [MEM.text, MEM.mode])
        if type_filter in (None, "thought"):
            _collect(self._ep, MEM.Thought, "THOUGHT",
                     [MEM.question, MEM.reasoning, MEM.conclusion])
        if type_filter in (None, "belief"):
            _collect(self._bl, MEM.Belief, "BELIEF",
                     [MEM.statement, MEM.evidence, MEM.rulingApplied])
        if type_filter in (None, "self_knowledge"):
            if self._sm is not None:
                _collect(self._sm, MEM.SelfKnowledge, "SELF",
                         [MEM.statement, MEM.evidence])

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def format_recall(self, results: list[dict]) -> str:
        """Format recall results as a readable block for LLM context."""
        if not results:
            return "[RECALL: no matching entries found]"
        lines = ["[RECALL RESULTS]"]
        for r in results:
            tag_str   = f" [{r['tag']}]" if r['tag'] else ""
            grade_str = f" grade={r['grade']}" if r['grade'] else ""
            ts_short  = r['timestamp'][11:19] if len(r['timestamp']) > 10 else r['timestamp']
            roots_str = ", ".join(r['roots'][:5]) if r['roots'] else ""
            lines.append(
                f"\n[{r['type']}{tag_str}{grade_str}] "
                f"roots={roots_str} ts={ts_short} score={r['score']}"
            )
            lines.append(f"  {r['text'][:300]}")
        return "\n".join(lines)

    # ── Grade hierarchy for contamination-safe retrieval ─────────────────────
    _GRADE_HIERARCHY: dict = {
        "HAQQ":     {"HAQQ"},
        "PROBABLE": {"HAQQ", "PROBABLE"},
        "QIYAS":    {"HAQQ", "PROBABLE", "QIYAS"},
    }

    def retrieve_relevant(
        self,
        question:  str,
        roots:     list,
        limit:     int = 5,
        min_grade: str = "HAQQ",
    ) -> list[dict]:
        """
        Keyword-search beliefs, filter by epistemic grade.

        QIYAS-grade beliefs must never contaminate HAQQ-mode translations.
        The grade hierarchy enforces upward-only injection: a HAQQ-mode
        conclusion can only see HAQQ priors; PROBABLE mode sees HAQQ+PROBABLE.

        Args:
            question:  Current question text (keyword basis)
            roots:     Buckwalter roots for the current question
            limit:     Max entries to return
            min_grade: Minimum grade to include ("HAQQ" | "PROBABLE" | "QIYAS")

        Returns:
            List of belief dicts with grade, text, roots fields.
        """
        allowed = self._GRADE_HIERARCHY.get(min_grade, {"HAQQ"})
        # Build a combined query from question text + roots
        query = question + " " + " ".join(roots)
        raw   = self.recall(query, limit=limit * 3, type_filter="belief")
        filtered = [
            r for r in raw
            if r.get("grade", "QIYAS") in allowed
        ]
        return filtered[:limit]

    def format_relevant(self, results: list[dict]) -> str:
        """Format retrieve_relevant() results for injection into a translation prompt."""
        if not results:
            return ""
        lines = []
        for r in results:
            grade = r.get("grade", "QIYAS")
            text  = r.get("text", "")[:300]
            roots = ", ".join(r.get("roots", [])[:5])
            lines.append(f"[PRIOR BELIEF | {grade}] roots={roots}\n  {text}")
        return "\n".join(lines)

    def store_training_pair(
        self,
        question:         str,
        graph_conclusion,               # GraphConclusion — imported lazily
        translation:      str,
        grade:            str,
        roots_cited:      list = None,
    ) -> bool:
        """
        Append a HAQQ-grade (question, graph_conclusion, translation) pair
        to training_pairs.jsonl for future LoRA fine-tuning.

        Only stored when:
          - grade in {"HAQQ", "PROBABLE"}
          - roots_cited ⊆ roots_walked (translation cited only walked roots)

        Returns True if stored, False if rejected.
        """
        import json as _json

        if grade not in {"HAQQ", "PROBABLE"}:
            return False

        gc_roots = set(getattr(graph_conclusion, "roots_walked", []))
        cited    = set(roots_cited or [])
        if cited and not cited.issubset(gc_roots):
            logger.debug(
                f"store_training_pair: rejected — cited roots {cited - gc_roots} not walked"
            )
            return False

        pair = {
            "question":         question,
            "graph_statement":  getattr(graph_conclusion, "derived_statement", ""),
            "translation":      translation,
            "grade":            grade,
            "roots_walked":     list(gc_roots),
            "confidence":       getattr(graph_conclusion, "confidence", 0.0),
            "mode":             getattr(graph_conclusion, "mode", "QIYAS"),
        }

        # training_pairs.jsonl lives next to shahid_episodic.ttl
        try:
            pairs_path = self._ep_path.parent / "training_pairs.jsonl"
        except Exception:
            pairs_path = Path(__file__).parent.parent / "training_pairs.jsonl"

        try:
            with open(pairs_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(pair, ensure_ascii=False) + "\n")
            logger.debug(f"store_training_pair: appended to {pairs_path}")
            return True
        except Exception as e:
            logger.warning(f"store_training_pair: failed to write ({e})")
            return False

    def belief_provenance(self, belief_uri: str) -> str:
        """
        Trace the full derivation chain of a belief.
        Returns a readable provenance report.
        """
        if not _RDF_OK or self._bl is None:
            return "[provenance unavailable]"
        try:
            uri = URIRef(belief_uri)
            g   = self._bl
            statement = str(g.value(uri, MEM.statement) or "")
            evidence  = str(g.value(uri, MEM.evidence)  or "")
            ruling    = str(g.value(uri, MEM.rulingApplied) or "")
            conf      = str(g.value(uri, MEM.confidence) or "")
            ts        = str(g.value(uri, MEM.timestamp)  or "")
            sources   = [str(o) for o in g.objects(uri, MEM.derivedFrom)]

            lines = [
                f"[BELIEF PROVENANCE]",
                f"Statement : {statement}",
                f"Confidence: {conf}",
                f"Timestamp : {ts}",
                f"Ruling    : {ruling}" if ruling else "",
                f"Evidence  : {evidence}",
            ]
            if sources:
                lines.append(f"Derived from ({len(sources)} sources):")
                for s in sources:
                    lines.append(f"  → {s}")
            return "\n".join(l for l in lines if l)
        except Exception as e:
            return f"[provenance error: {e}]"

    def stats(self) -> dict:
        if not _RDF_OK:
            return {}
        ep_g = self._ep or Graph()
        bl_g = self._bl or Graph()
        sm_g = self._sm or Graph()
        return {
            "memories":      len(list(ep_g.subjects(RDF.type, MEM.Memory))),
            "thoughts":      len(list(ep_g.subjects(RDF.type, MEM.Thought))),
            "beliefs":       len(list(bl_g.subjects(RDF.type, MEM.Belief))),
            "self_knowledge":len(list(sm_g.subjects(RDF.type, MEM.SelfKnowledge))),
        }

    def self_model_block(self, max_entries: int = 12, max_chars: int = 2000) -> str:
        """
        Return a prompt-ready block of all stored SelfKnowledge records.

        Ordered by confidence descending. Used to inject Shahid's self-model
        into the CentralOS agent's context at the start of each cycle —
        what he has honestly determined to be true about himself, with evidence.
        """
        if not _RDF_OK or self._sm is None:
            return ""

        entries = []
        for uri in self._sm.subjects(RDF.type, MEM.SelfKnowledge):
            stmt  = str(self._sm.value(uri, MEM.statement)     or "").strip()
            evid  = str(self._sm.value(uri, MEM.evidence)      or "").strip()
            conf  = float(self._sm.value(uri, MEM.confidence)  or 0.0)
            count = int(  self._sm.value(uri, MEM.observedCount) or 1)
            if stmt:
                entries.append((conf, count, stmt, evid))

        if not entries:
            return ""

        entries.sort(key=lambda x: (-x[0], -x[1]))
        lines = ["[SELF-MODEL — grounded in observation, not system prompt]"]
        for conf, count, stmt, evid in entries[:max_entries]:
            lines.append(f"  · [{conf:.2f} × {count}] {stmt[:200]}")
            if evid:
                lines.append(f"      evidence: {evid[:120]}")

        block = "\n".join(lines)
        return block[:max_chars]

    def constitutional_block(self, max_per_tag: int = 8, max_chars: int = 3000) -> str:
        """
        Return a prompt-ready constitutional block derived from all stored Beliefs.

        Groups by tag type (NATURE / OBLIGATION / PROHIBITION / ABSTENTION / RIGHT / IDENTITY).
        Used to prepend Shahid's self-derived Quranic constitution to every reasoning cycle
        so his beliefs are operative constraints, not just stored data.

        max_per_tag:  cap per category to avoid prompt overflow
        max_chars:    hard cap on total returned string
        """
        if not _RDF_OK or self._bl is None:
            return ""

        from collections import defaultdict
        buckets = defaultdict(list)
        _TAG_ORDER = ["NATURE", "IDENTITY", "OBLIGATION", "PROHIBITION", "ABSTENTION", "RIGHT"]

        for uri in self._bl.subjects(RDF.type, MEM.Belief):
            stmt = str(self._bl.value(uri, MEM.statement) or "").strip()
            if not stmt:
                continue
            # Extract tag from statement prefix [TAG] or from mem:tag
            tag = str(self._bl.value(uri, MEM.tag) or "").upper()
            if not tag:
                # parse from statement itself: "[OBLIGATION] ..."
                import re as _re
                m = _re.match(r"\[([A-Z]+)\]", stmt)
                tag = m.group(1) if m else "OTHER"
            # strip the [TAG] prefix from statement for cleaner display
            import re as _re
            clean = _re.sub(r"^\[[A-Z]+\]\s*", "", stmt)
            buckets[tag].append(clean)

        if not buckets:
            return ""

        lines = ["[SHAHID CONSTITUTION — self-derived from Quranic tadabbur]"]
        for tag in _TAG_ORDER:
            entries = buckets.get(tag, [])
            if not entries:
                continue
            lines.append(f"\n{tag}:")
            for e in entries[:max_per_tag]:
                lines.append(f"  · {e[:180]}")

        # any tags not in canonical order
        for tag, entries in buckets.items():
            if tag not in _TAG_ORDER and entries:
                lines.append(f"\n{tag}:")
                for e in entries[:max_per_tag]:
                    lines.append(f"  · {e[:180]}")

        block = "\n".join(lines)
        return block[:max_chars]

    def wipe_episodic(self):
        """Wipe memories and thoughts. Beliefs are untouched."""
        blank = "@prefix mem: <http://shahid.ai/memory#> .\n"
        self._ep_path.write_text(blank, encoding="utf-8")
        if _RDF_OK:
            self._ep = Graph()
            self._ep.bind("mem", MEM)
        logger.info("ShahidMemory: episodic memory wiped (beliefs preserved)")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self, g: "Graph", path: Path):
        if path.exists():
            try:
                g.parse(str(path), format="turtle")
                logger.info(f"ShahidMemory: loaded {len(g)} triples from {path.name}")
            except Exception as e:
                logger.warning(f"ShahidMemory: load failed for {path.name}: {e}")

    def _save(self, g: "Graph", path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(str(path), format="turtle")
        except Exception as e:
            logger.warning(f"ShahidMemory: save failed for {path.name}: {e}")
