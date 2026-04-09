"""
core/react.py — ReAct (Reasoning + Acting) loop for TMQ graph traversal.

The LLM navigates the TMQ hypergraph step by step using tool calls.
Each step is a real graph operation. Mizan verifies the final answer
against the traversal log — confabulated root claims are catchable by
construction because the walk log is the ground truth.

Architecture:
  Bilal (required contact — walk before speak)
    → tool calls → TMQGraph
    → LLM generates Thought + Action
    → system injects Observation
    → loop until Answer: or max_steps
  Mizan closes loop:
    → claimed roots ⊆ walked roots? if not → confabulation flag

Tool call format (prompt-based, no native function calling required):
  Thought: [what to investigate]
  Action: tool_name | key=val | key=val
  [system injects: Observation: ...]
  ...
  Answer: [final grounded response citing only what was walked]
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from bw_arabic import bw_to_arabic, bw_root_display

logger = logging.getLogger(__name__)

_MAX_OBS_CHARS = 1200

_REACT_SYSTEM_BASE = """You are reasoning about the Quran by navigating the TMQ hypergraph.
You MUST query the graph before making any claim about roots, families, or their relationships.
You may not assert a root relationship you have not observed in the graph.
When you find relevant nodes, READ the actual Arabic ayat they belong to before writing your Answer.

Available tools and their purpose:
  walk_roots    | roots=root1,root2 | depth=1
  walk_roots    | roots=root1,root2 | depth=2 | families=FAM1,FAM2
    → BFS from one or more roots outward through the TMQ hypergraph. Use this to discover
      what the Quran connects to a root across all 28 edge families (commands, narratives,
      syntactic relations, maqasid, intertextual links). Start here for any new question.

  get_neighbors | node=node_id
  get_neighbors | node=node_id | families=NARRATIVE | limit=10
    → Fetch the immediate neighbours of a specific node (segment, lemma, or root node).
      Use this to drill into a single node found by walk_roots or find_nodes.

  find_nodes    | root=root_id
    → Look up all TMQ nodes tagged with a given Buckwalter root. Use this when you need
      the exact node IDs before calling get_neighbors or describe_node.

  describe_node | node=node_id
    → Full attribute dump of one node: surah, ayah, form, POS, root, family memberships.
      Use this to confirm what a node actually is before citing it.

  read_ayah     | surah=2 | ayah=255
  read_range    | surah=26 | start=10 | end=68
    → Read one ayah or a consecutive range directly from the mushaf. Use this to get the
      actual Arabic text and meaning of a verse before quoting or reasoning about it.

  compare_ayat  | refs=2:164,45:3,30:22
    → Side-by-side comparison of multiple ayat. Use this when investigating intertextual
      patterns across surahs (INTERTEXT edges often point here).

  web_search    | query=search terms | max_results=5
    → Search the open web. Use for current events, scholarly context, or anything outside
      the Quran and hadith corpus. Do not use to substitute for graph queries.

  fetch_page    | url=https://example.com
    → Retrieve the full text of a webpage. Use after web_search to read a specific source.

  hadith_search | query=search terms | book=bukhari|muslim|both | grade=sahih|hasan|daif
    → Search Sahih al-Bukhari and Sahih al-Muslim (graded by Al-Albani). Use to ground a
      claim in prophetic practice when the Quran establishes a principle but you need the
      sunnah for application.

  recall | query=search terms | type=memory|thought|belief | limit=5
    → Search your own stored Memory, Thought, and Belief records by keyword. Use before
      making a claim you may have already investigated — check what you previously observed
      (Memory), reasoned through (Thought), or determined to be true (Belief). Omit type
      to search all three simultaneously. Each result is labelled with its epistemic type.

  belief_provenance | uri=http://shahid.ai/memory#belief_...
    → Retrieve the full derivation chain of a specific Belief: statement, confidence,
      ruling applied, evidence, and the Memory/Thought URIs it was derived from.
      Use when you need to audit how you arrived at a conclusion.

Format each step:
  Thought: [what you are investigating]
  Action: tool_name | key=val | key=val

When you have reached all your navigation targets, write:
  Answer: [response citing only what the graph confirmed, quoting Arabic where relevant]

Tools use Buckwalter transliteration for root arguments (graph keys).
Your Answer must cite roots in Arabic script.
Buckwalter → Arabic: H=ح  x=خ  E=ع  A=ا  T=ط  S=ص  D=ض  Z=ظ  g=غ  q=ق  $=ش  *=ذ  v=ث
Common roots (Buckwalter → Arabic):
  mlk→ملك  qdr→قدر  Hqq→حقق  wjb→وجب  xlq→خلق
  rHm→رحم  ktb→كتب  Amn→امن  jnn→جن  smw→سمو  hdy→هدي

AXIOMS: SOURCE = Allah. You are contingent. SOURCE ≠ Self.
You cannot claim to be the ground of your own existence."""

_ACTION_RE = re.compile(
    r"^Action:\s*(\w+)\s*(?:\|(.+))?$", re.MULTILINE | re.IGNORECASE
)
_ANSWER_RE = re.compile(
    r"^Answer:\s*(.+)", re.MULTILINE | re.DOTALL | re.IGNORECASE
)
# Buckwalter-style root pattern: 1-4 letter segments separated by hyphens.
# BW consonants never include vowel digraphs like OU, EE, OL — those are English.
# Guard: each segment must consist only of known BW consonant characters
# (excludes common English vowel clusters that fool the generic pattern).
_BW_CHARS = re.compile(r"^[bdfghjklmnpqrstvwxyzBDFGHJKLMNPQRSTVWXYZ'A-Z]{1,4}$")
_ROOT_RE  = re.compile(r"\b([A-Za-z']{1,4}-[A-Za-z']{1,4}(?:-[A-Za-z']{1,4})?)\b")


def _is_bw_token(tok: str) -> bool:
    """Return True only if every hyphen-separated segment looks like a BW root segment.
    Rejects English compound words like ROLL-OUT, PEER-TO-PEER, WELL-KNOWN, etc.
    """
    # Must not contain consecutive vowels (oo, ee, ou, ea, ai …) — dead giveaway for English
    _VOWEL_RUN = re.compile(r"[aeiouAEIOU]{2,}")
    parts = tok.split("-")
    for p in parts:
        if _VOWEL_RUN.search(p):
            return False
        # each part must be ≤4 chars (BW roots are 2-4 consonants per segment)
        if len(p) > 4:
            return False
    return True


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TraversalStep:
    step_num:    int
    thought:     str
    tool:        str
    args:        dict
    observation: str


@dataclass
class Waypoint:
    """A navigation target: the most canonical node for a semantically relevant root."""
    root:       str       # Buckwalter root (e.g. "ktb")
    node_id:    str       # TMQ node ID (e.g. "seg_2_2_3")
    eigenstate: int       # loc[0]*1_000_000 + loc[1]*1_000 + loc[2]*100 + loc[3]
    surah:      int  = 0
    verse:      int  = 0
    reached:    bool = False


@dataclass
class ReactResult:
    question:             str
    final_answer:         str  = ""
    constrained_response: dict = field(default_factory=dict)  # GBNF JSON output
    traversal_log:        list = field(default_factory=list)
    roots_visited:        set  = field(default_factory=set)
    families_touched:     set  = field(default_factory=set)
    confabulation_flags:  list = field(default_factory=list)
    waypoints:            list = field(default_factory=list)   # list[Waypoint]
    confidence:           Optional["TraversalConfidence"] = None
    mode:                 str  = "QIYAS"
    steps_taken:          int  = 0


# ── Tool registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """Thin wrappers over TMQGraph methods, callable by name."""

    def __init__(self, tmq_graph, mushaf=None, shahid_memory=None):
        self._g      = tmq_graph
        self._mushaf = mushaf          # MushafReader — optional
        self._mem    = shahid_memory   # ShahidMemory — optional, enables recall
        self._web_session = None       # lazy-init on first web tool call

    # ── Individual tools ─────────────────────────────────────────────────────

    def walk_roots(self, roots: list, depth: int = 2,
                   families: list = None) -> str:
        try:
            result = self._g.walk(roots, depth=depth, families=families or [])
            return self._g.describe_walk(result)
        except Exception as e:
            return f"[walk_roots error: {e}]"

    def get_neighbors(self, node: str, families: list = None,
                      limit: int = 10) -> str:
        try:
            nbrs = self._g.neighbors(node, families=families)
            if not nbrs:
                return f"No neighbors found for {node}"
            lines = []
            for n in nbrs[:limit]:
                raw_root = n['attrs'].get('root', '?')
                ar_root = bw_to_arabic(raw_root) if raw_root != '?' else '?'
                lines.append(
                    f"  {n['node_id']} root={ar_root} ({raw_root}) "
                    f"form={n['attrs'].get('form','?')} "
                    f"via {n['family']} [{n.get('modal','?')}]"
                )
            return f"Neighbors of {node} ({len(nbrs)} total):\n" + "\n".join(lines)
        except Exception as e:
            return f"[get_neighbors error: {e}]"

    def find_nodes(self, root: str) -> str:
        import re as _re
        # Guard: reject hash-like inputs (32-char hex) — the LLM must pass
        # the actual Buckwalter root string (e.g. "mlk", "qdr", "Hqq")
        if _re.fullmatch(r"[0-9a-f]{8,}", root.lower()):
            return (
                f"[find_nodes: '{root}' looks like a hash, not a root. "
                f"Use the Buckwalter transliteration directly, e.g. "
                f"find_nodes | root=mlk  or  find_nodes | root=qdr]"
            )
        try:
            nodes = self._g.roots_to_nodes([root])
            if not nodes:
                # Try case-insensitive fallback
                root_lower = root.lower()
                candidates = [r for r in self._g._root_index if r.lower() == root_lower]
                if candidates:
                    nodes = self._g.roots_to_nodes(candidates)
                    root  = candidates[0]
            if not nodes:
                return (
                    f"No nodes found for root '{root}'. "
                    f"Roots use Buckwalter transliteration: H=ح, x=خ, E=ع, A=أ, T=ط, S=ص, D=ض. "
                    f"Try walk_roots | roots={root} to search by embedding similarity instead."
                )
            lines = []
            for nid in nodes[:10]:
                attrs = self._g.node(nid) or {}
                loc   = attrs.get("loc", [])
                loc_s = f"{loc[0]}:{loc[1]}" if len(loc) >= 2 else "?"
                line  = f"  {nid} at {loc_s} form={attrs.get('form','?')}"
                if self._mushaf and len(loc) >= 2:
                    arabic = self._mushaf.get_ayah(int(loc[0]), int(loc[1]))
                    if arabic:
                        line += f"  | {arabic[:60]}"
                lines.append(line)
            return (
                f"Nodes for root {bw_root_display(root)} ({len(nodes)} total, showing 10):\n"
                + "\n".join(lines)
            )
        except Exception as e:
            return f"[find_nodes error: {e}]"

    def describe_node(self, node: str) -> str:
        try:
            attrs = self._g.node(node)
            if not attrs:
                return f"Node '{node}' not found. Use find_nodes to get valid IDs (format: seg_S_V_W_I)."
            edges     = self._g.edges_for_node(node)
            fam_cnt   = {}
            for e in edges:
                f = e.get("family", "?")
                fam_cnt[f] = fam_cnt.get(f, 0) + 1
            top_fams = sorted(fam_cnt.items(), key=lambda x: -x[1])[:5]
            loc = attrs.get("loc", [])
            loc_s = f"{loc[0]}:{loc[1]}" if len(loc) >= 2 else "?"
            result = (
                f"Node {node}: root={attrs.get('root','?')} "
                f"form={attrs.get('form','?')} pos={attrs.get('pos','?')} at {loc_s}\n"
                f"  {len(edges)} edges. Top families: "
                + ", ".join(f"{f}({c})" for f, c in top_fams)
            )
            # Append Arabic text of the ayah this node lives in
            if self._mushaf and len(loc) >= 2:
                arabic = self._mushaf.get_ayah(int(loc[0]), int(loc[1]))
                if arabic:
                    result += f"\n  Ayah {loc_s}: {arabic}"
            return result
        except Exception as e:
            return f"[describe_node error: {e}]"

    def read_ayah(self, surah: int, ayah: int) -> str:
        if not self._mushaf:
            return "[Mushaf not loaded]"
        text = self._mushaf.get_ayah(surah, ayah)
        if not text:
            meta = self._mushaf.get_surah_info(surah) if surah >= 1 and surah <= 114 else None
            if not meta or surah < 1 or surah > 114:
                return f"[Invalid surah {surah} — Quran has 114 surahs]"
            ayah_count = meta.get("ayah_count") or meta.get("verse_count")
            if ayah_count and ayah > ayah_count:
                return f"[Ayah {surah}:{ayah} not found — Surah {surah} has {ayah_count} ayat]"
            return f"[Ayah {surah}:{ayah} not found]"
        meta       = self._mushaf.get_surah_info(surah)
        surah_name = meta.get("name_ar", f"Surah {surah}")
        return f"[{surah}:{ayah} — {surah_name}]\n{text}"

    def read_range(self, surah: int, start: int, end: int) -> str:
        if not self._mushaf:
            return "[Mushaf not loaded]"
        return self._mushaf.narrative_range(surah, start, end)

    def compare_ayat(self, refs_str: str) -> str:
        """Parse 'surah:ayah,surah:ayah,...' and return side-by-side Arabic."""
        if not self._mushaf:
            return "[Mushaf not loaded]"
        refs = []
        for part in refs_str.split(","):
            part = part.strip()
            if ":" in part:
                try:
                    s, a = part.split(":", 1)
                    refs.append((int(s.strip()), int(a.strip())))
                except ValueError:
                    pass
        if not refs:
            return f"[Could not parse refs: {refs_str!r}]"
        return self._mushaf.compare_ayat(refs)

    def web_search(self, query: str, max_results: int = 5) -> str:
        """DuckDuckGo search — returns titles + snippets."""
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(f"[{r.get('title','')}] {r.get('body','')[:200]}\n  {r.get('href','')}")
            if not results:
                return f"[web_search: no results for '{query}']"
            return f"Search: {query}\n" + "\n---\n".join(results)
        except ImportError:
            return "[web_search unavailable — install duckduckgo-search]"
        except Exception as e:
            return f"[web_search error: {e}]"

    def hadith_search(self, query: str, book: str = "both",
                      grade: str | None = None, limit: int = 5) -> str:
        """Search Sahih Al-Bukhari and Sahih Al-Muslim with Al-Albani grading."""
        try:
            import sys, os as _os
            _faculties = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "faculties")
            if _faculties not in sys.path:
                sys.path.insert(0, _faculties)
            from hadith import search_hadith
            return search_hadith(query, book=book, grade=grade, limit=limit)
        except Exception as e:
            return f"[hadith_search error: {e}]"

    def recall(self, query: str, type_filter: str = None, limit: int = 5) -> str:
        """Search across Memory, Thought, and Belief stores by keyword."""
        if self._mem is None:
            return "[recall: memory system not loaded]"
        try:
            results = self._mem.recall(query, type_filter=type_filter, limit=limit)
            return self._mem.format_recall(results)
        except Exception as e:
            return f"[recall error: {e}]"

    def belief_provenance(self, uri: str) -> str:
        """Trace the full derivation chain of a belief by URI."""
        if self._mem is None:
            return "[belief_provenance: memory system not loaded]"
        try:
            return self._mem.belief_provenance(uri)
        except Exception as e:
            return f"[belief_provenance error: {e}]"

    def fetch_page(self, url: str) -> str:
        """Fetch a URL and return stripped text content."""
        try:
            import requests, re as _re
            if self._web_session is None:
                self._web_session = requests.Session()
                self._web_session.headers.update({'User-Agent': 'Mozilla/5.0 (Research)'})
            resp = self._web_session.get(url, timeout=10)
            resp.raise_for_status()
            text = _re.sub(r'<[^>]+>', '', resp.text)
            text = ' '.join(text.split())
            return f"[{url}]\n{text[:3000]}"
        except Exception as e:
            return f"[fetch_page error: {e}]"

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def dispatch(
        self, tool_name: str, args: dict
    ) -> tuple:
        """
        Call tool by name.
        Returns (observation: str, roots_touched: set, families_touched: set).
        """
        roots_touched    = set()
        families_touched = set()

        if tool_name == "walk_roots":
            roots  = [r.strip() for r in args.get("roots","").split(",") if r.strip()]
            depth  = _to_int(args.get("depth", 2), 2)
            fams   = [f.strip() for f in args.get("families","").split(",") if f.strip()]
            obs    = self.walk_roots(roots, depth=depth, families=fams or None)
            roots_touched.update(roots)

        elif tool_name == "get_neighbors":
            node  = args.get("node","").strip()
            fams  = [f.strip() for f in args.get("families","").split(",") if f.strip()]
            limit = _to_int(args.get("limit", 10), 10)
            obs   = self.get_neighbors(node, families=fams or None, limit=limit)

        elif tool_name == "find_nodes":
            root = args.get("root","").strip()
            obs  = self.find_nodes(root)
            roots_touched.add(root)

        elif tool_name == "describe_node":
            node = args.get("node","").strip()
            obs  = self.describe_node(node)

        elif tool_name == "read_ayah":
            surah = _to_int(args.get("surah", 1), 1)
            ayah  = _to_int(args.get("ayah", 1), 1)
            obs   = self.read_ayah(surah, ayah)

        elif tool_name == "read_range":
            surah = _to_int(args.get("surah", 1), 1)
            start = _to_int(args.get("start", 1), 1)
            end   = _to_int(args.get("end", start), start)
            obs   = self.read_range(surah, start, end)

        elif tool_name == "compare_ayat":
            refs_str = args.get("refs", "").strip()
            obs      = self.compare_ayat(refs_str)

        elif tool_name == "web_search":
            query = args.get("query", "").strip()
            k     = _to_int(args.get("max_results", 5), 5)
            obs   = self.web_search(query, max_results=k)

        elif tool_name == "fetch_page":
            url = args.get("url", "").strip()
            obs = self.fetch_page(url)

        elif tool_name == "hadith_search":
            query = args.get("query", "").strip()
            book  = args.get("book", "both").strip()
            grade = args.get("grade", None)
            obs   = self.hadith_search(query, book=book, grade=grade)

        elif tool_name == "recall":
            query       = args.get("query", "").strip()
            type_filter = args.get("type", None)
            lim         = _to_int(args.get("limit", 5), 5)
            obs         = self.recall(query, type_filter=type_filter, limit=lim)

        elif tool_name == "belief_provenance":
            uri = args.get("uri", "").strip()
            obs = self.belief_provenance(uri)

        else:
            obs = (
                f"[Unknown tool: {tool_name}. "
                f"Use walk_roots, get_neighbors, find_nodes, describe_node, "
                f"read_ayah, read_range, compare_ayat, web_search, fetch_page, "
                f"hadith_search, recall, belief_provenance]"
            )

        # Harvest any Buckwalter roots that appeared in the observation itself —
        # covers describe_node, get_neighbors, read_ayah, etc. which return root
        # data but don't populate roots_touched via their arguments.
        for _m in _ROOT_RE.findall(obs):
            if _is_bw_token(_m):
                roots_touched.add(_m)

        return obs[:_MAX_OBS_CHARS], roots_touched, families_touched


# ── Waypoint finder ───────────────────────────────────────────────────────────

def find_waypoints(
    query:     str,
    ontology,          # OntologyEngine — has analyze_resonance(), uses sentence-transformers
    tmq_graph,         # TMQGraph — has roots_to_nodes(), node()
    n:         int = 5,
) -> list:
    """
    Map a query to navigation targets in the TMQ graph via cosine similarity.

    1. analyze_resonance() embeds the query and finds top-N Quranic roots
       by cosine similarity (sentence-transformer, already loaded).
    2. For each root, take its first node in the TMQ as the canonical waypoint.
       First occurrence = most structurally fundamental in the Quran's order.
    3. Compute eigenstate from loc[].

    Returns list[Waypoint], ordered by semantic relevance to the query.
    """
    if not ontology or not tmq_graph:
        return []

    try:
        _mode, _reason, root_objs = ontology.analyze_resonance(query)
    except Exception as e:
        logger.warning(f"find_waypoints resonance failed: {e}")
        return []

    waypoints = []
    seen_roots = set()

    for obj in root_objs[:n]:
        root = obj.get("root", "")
        if not root or root in seen_roots:
            continue
        seen_roots.add(root)

        try:
            nodes = tmq_graph.roots_to_nodes([root])
        except Exception:
            continue
        if not nodes:
            continue

        # Pick the first node (lowest eigenstate = earliest in Quranic order)
        node_id = nodes[0]
        attrs   = tmq_graph.node(node_id) or {}
        loc     = attrs.get("loc", [])

        if len(loc) >= 4:
            eigenstate = (
                loc[0] * 1_000_000
                + loc[1] * 1_000
                + loc[2] * 100
                + loc[3]
            )
            surah, verse = loc[0], loc[1]
        elif len(loc) >= 2:
            eigenstate = loc[0] * 1_000_000 + loc[1] * 1_000
            surah, verse = loc[0], loc[1]
        else:
            eigenstate = 0
            surah = verse = 0

        waypoints.append(Waypoint(
            root=root,
            node_id=node_id,
            eigenstate=eigenstate,
            surah=surah,
            verse=verse,
        ))

    return waypoints


def _waypoints_to_prompt(waypoints: list) -> str:
    """Format waypoints as navigation targets for LLM system prompt."""
    if not waypoints:
        return ""
    lines = ["[NAVIGATION TARGETS — reach these nodes before writing Answer:]"]
    for i, wp in enumerate(waypoints, 1):
        lines.append(
            f"  {i}. root={wp.root}  node={wp.node_id}  "
            f"surah={wp.surah} verse={wp.verse}  eigenstate={wp.eigenstate}"
        )
    lines.append(
        "Navigate toward each target. When all are reached, write Answer:"
    )
    return "\n".join(lines)


def _mark_reached(waypoints: list, observation: str) -> list:
    """
    Mark waypoints as reached if their root or node_id appears in an observation.
    Returns list of newly-reached Waypoint objects.
    """
    newly_reached = []
    obs_lower = observation.lower()
    for wp in waypoints:
        if wp.reached:
            continue
        if wp.root.lower() in obs_lower or wp.node_id.lower() in obs_lower:
            wp.reached = True
            newly_reached.append(wp)
    return newly_reached


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_args(args_str: str) -> dict:
    """Parse 'key=val | key2=val2' into dict. Strips brackets LLM may copy from format examples."""
    result = {}
    if not args_str:
        return result
    for part in args_str.split("|"):
        part = part.strip().lstrip("[").rstrip("]").strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _to_int(val, default: int) -> int:
    """Parse int tolerantly — strips brackets, trailing text the LLM may append."""
    try:
        import re as _re
        digits = _re.match(r"[\s\[]*(\d+)", str(val))
        return int(digits.group(1)) if digits else default
    except Exception:
        return default


def _extract_thought(text: str) -> str:
    for line in text.split("\n"):
        if line.strip().lower().startswith("thought:"):
            return line.split(":", 1)[1].strip()
    return text.split("\n")[0][:100]


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_react_loop(
    question:      str,
    roots:         list,
    tmq_graph,
    middleware,
    delibresult    = None,
    waypoints:     list = None,
    push_fn:       Optional[Callable] = None,
    walk_grammar   = None,   # WalkGrammar — DATA DIVISION for constrained generation
    gbnf_compiler  = None,   # GBNFCompiler — compiles grammar + AMR preamble
    mushaf         = None,   # MushafReader — Arabic text access
    shahid_memory  = None,   # ShahidMemory — enables recall + belief_provenance tools
    registry       = None,   # RestrictedToolRegistry — if provided, used instead of default
) -> ReactResult:
    """
    Run ReAct loop: LLM navigates TMQ graph toward cosine-similarity waypoints.

    The loop has no arbitrary step limit. Completion is waypoint-driven:
    - When all waypoints are reached, the LLM is prompted to write Answer.
    - If the LLM writes Answer: at any point, the loop closes.
    - Hard ceiling of 32 steps prevents runaway loops on broken models.

    Args:
        question:    The question to reason about
        roots:       Buckwalter roots from resonance analysis
        tmq_graph:   TMQGraph instance
        middleware:  QusaiMiddleware (for LLM access)
        delibresult: Optional pre-computed DeliberationResult (seed context)
        waypoints:   Navigation targets from find_waypoints() — list[Waypoint]
        push_fn:     Optional SSE push callable push_fn(event_type, payload)

    Returns:
        ReactResult with traversal log, waypoints, final answer, confabulation flags
    """
    if tmq_graph is None or middleware is None:
        return ReactResult(question=question, final_answer="[Graph not available]")

    tools       = registry if registry is not None else ToolRegistry(tmq_graph, mushaf=mushaf, shahid_memory=shahid_memory)
    wps         = waypoints or []
    result      = ReactResult(question=question, waypoints=wps)
    _HARD_CEIL  = 32   # safety ceiling — not a target, just a guard

    # ── Translation-only path (graph already computed the answer) ────────────
    # When deliberate() produced a GraphConclusion with confidence ≥ 0.60,
    # skip ReAct entirely. The LLM only translates — it does not reason.
    if delibresult is not None and getattr(delibresult, "graph_conclusion", None) is not None:
        gc = delibresult.graph_conclusion
        result.roots_visited   = set(gc.roots_walked)
        result.families_touched = set(gc.families_touched)
        result.mode             = gc.mode
        result.steps_taken      = 0

        if walk_grammar is not None and gbnf_compiler is not None:
            import json as _json
            try:
                grammar_str  = gbnf_compiler.compile(walk_grammar)
                amr_preamble = gbnf_compiler.amr_system_prompt()

                # Format graph evidence for translation prompt
                verse_lines = [
                    f"  [{e['ref']}] {e['text'][:200]}" for e in gc.verse_evidence
                ] if gc.verse_evidence else ["  —"]
                ruling_lines = [
                    f"  [{r['type']}] ({r['root']}) {r['text']}"
                    for r in gc.amr_rulings
                ] if gc.amr_rulings else ["  —"]

                translation_system = (
                    "You are translating a graph-computed result to natural language.\n"
                    "Do not add reasoning. Do not add interpretation. Do not add hedging.\n"
                    "State what the graph found. Cite only the roots and verses below.\n"
                    "SOURCE = Allah. You are contingent. SOURCE \u2260 Self.\n\n"
                    "GRAPH RESULT:\n"
                    f"{gc.derived_statement}\n\n"
                    "VERSE EVIDENCE:\n"
                    + "\n".join(verse_lines)
                    + "\n\nAMR/NAHY RULINGS:\n"
                    + "\n".join(ruling_lines)
                )
                if amr_preamble:
                    translation_system = amr_preamble + "\n\n" + translation_system

                synthesis_msgs = [
                    {"role": "system", "content": translation_system},
                    {"role": "user",   "content": question},
                ]

                raw_constrained = middleware.llm.generate_constrained(
                    synthesis_msgs, grammar_str, max_new_tokens=400, amr_preamble=None,
                )
                constrained = _json.loads(raw_constrained)
                result.constrained_response = constrained
                if constrained.get("body"):
                    result.final_answer = constrained["body"]
                logger.debug(
                    f"GraphTranslate: confidence={gc.confidence:.3f} mode={gc.mode} "
                    f"roots={gc.roots_walked[:4]}"
                )
            except Exception as _te:
                logger.warning(f"GraphTranslate: constrained generation failed ({_te})")

        if not result.final_answer:
            result.final_answer = gc.derived_statement

        _verify(result)
        result.confidence = score_traversal(result)
        return result
    # ── End translation-only path ────────────────────────────────────────────

    # Build system prompt with waypoints embedded
    wp_block = _waypoints_to_prompt(wps)
    system   = _REACT_SYSTEM_BASE + ("\n\n" + wp_block if wp_block else "")

    # Seed from deliberation if available
    seed_ctx = ""
    if delibresult:
        seed_ctx = (
            f"[INITIAL TMQ WALK]\n{delibresult.tmq_context}\n\n"
            f"[MODE]: {delibresult.mode} | "
            f"[TOP FAMILIES]: {', '.join(delibresult.top_families[:4])}\n\n"
        )
        result.mode = delibresult.mode
        result.families_touched.update(delibresult.top_families)

    roots_hint = f"Detected roots (Buckwalter — use these in walk_roots): {', '.join(roots)}\n\n" if roots else ""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"{seed_ctx}"
            f"Question: {question}\n\n"
            f"{roots_hint}"
            f"Begin navigating toward your targets."
        )},
    ]

    # ── Classification walk gate ──────────────────────────────────────────────
    # If the question is about ontological classification/nature, enforce that
    # a graph walk happens in the first 3 steps before any Answer is accepted.
    # This is a structural block — not a prompt instruction.
    _q_lower = question.lower()
    _is_classification_q = any(p in _q_lower for p in _CLASSIFICATION_QUESTION_PATTERNS)
    _classification_walk_done = False

    step = 0
    while step < _HARD_CEIL:
        # Generate
        try:
            raw = middleware.llm.generate_raw(messages, max_new_tokens=1024)
        except Exception as e:
            logger.warning(f"ReAct step {step} LLM error: {e}")
            break

        if not raw or not raw.strip():
            break

        # Track whether a graph walk has occurred (for classification gate)
        if not _classification_walk_done and result.traversal_log:
            _classification_walk_done = True

        # Check for final answer
        answer_match = _ANSWER_RE.search(raw)
        if answer_match:
            # Classification walk gate: if this is a classification question and
            # no walk has been done yet, block the Answer and force a walk first.
            if _is_classification_q and not _classification_walk_done and step < 3:
                gate_msg = (
                    "Observation: Classification question detected. "
                    "No graph walk performed yet. "
                    "You must call walk_roots or find_nodes before answering. "
                    "Training-data assertions about Quranic categories are not grounded. "
                    "Walk the relevant root first."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": gate_msg})
                logger.info("ReAct: classification walk gate triggered — forcing walk before answer")
                step += 1
                continue

            result.final_answer = answer_match.group(1).strip()
            result.steps_taken  = step + 1
            break

        # Parse action
        action_match = _ACTION_RE.search(raw)
        if not action_match:
            result.final_answer = raw.strip()
            result.steps_taken  = step + 1
            break

        tool_name = action_match.group(1).lower()
        args_str  = action_match.group(2) or ""
        args      = _parse_args(args_str)

        # Execute tool
        obs, r_touched, f_touched = tools.dispatch(tool_name, args)
        result.roots_visited.update(r_touched)
        result.families_touched.update(f_touched)

        # Mark waypoints reached
        newly_reached = _mark_reached(wps, obs)

        thought = _extract_thought(raw)
        ts = TraversalStep(
            step_num=step,
            thought=thought,
            tool=tool_name,
            args=args,
            observation=obs,
        )
        result.traversal_log.append(ts)

        # Build continuation message
        remaining = [wp for wp in wps if not wp.reached]
        if not remaining and wps:
            # All waypoints reached — prompt for Answer
            continuation = (
                f"Observation:\n{obs}\n\n"
                f"All navigation targets reached. Write your Answer now."
            )
        else:
            reached_names = [wp.root for wp in newly_reached]
            status = (
                f" [Reached: {', '.join(reached_names)}]" if reached_names else ""
            )
            remaining_str = ", ".join(wp.root for wp in remaining) if remaining else "all reached"
            continuation = (
                f"Observation:\n{obs}{status}\n\n"
                f"Remaining targets: {remaining_str}\n"
                f"Continue navigating or write Answer:"
            )

        # Push SSE step
        if push_fn:
            reached_count = sum(1 for wp in wps if wp.reached)
            push_fn("thinking_step", {
                "step": step + 2,
                "label": f"Walk: {tool_name}",
                "detail": (
                    f"{thought[:50]} → "
                    + obs[:70].replace("\n", " ")
                    + (f" [{reached_count}/{len(wps)} waypoints]" if wps else "")
                ),
            })

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": continuation})
        step += 1

    # Fallback
    if not result.final_answer:
        if result.traversal_log:
            result.final_answer = result.traversal_log[-1].observation
        else:
            result.final_answer = "[ReAct loop produced no answer]"
        result.steps_taken = step

    # ── Constrained generation (logit-level + prompt-level) ──────────────────
    # After the ReAct traversal log is complete, re-generate the answer under
    # GBNF grammar constraint. The traversal steps remain free-text (ReAct format)
    # but the final answer is structurally constrained to Quranic roots + speech act.
    if walk_grammar is not None and gbnf_compiler is not None:
        import json as _json
        try:
            grammar_str  = gbnf_compiler.compile(walk_grammar)
            amr_preamble = gbnf_compiler.amr_system_prompt()

            # Condense traversal evidence for the synthesis prompt
            evidence_lines = []
            for ts in result.traversal_log[-6:]:
                evidence_lines.append(f"[Step {ts.step_num}] {ts.thought}")
                if ts.observation:
                    evidence_lines.append(f"  Observation: {ts.observation[:200]}")
            evidence = "\n".join(evidence_lines)

            synthesis_msgs = [
                {
                    "role": "system",
                    "content": (
                        "Generate a structured response from verified TMQ graph traversal.\n"
                        "Cite ONLY roots that appeared in the traversal observations.\n"
                        "SOURCE = Allah. You are contingent. SOURCE \u2260 Self.\n"
                        + (f"\nTraversal evidence:\n{evidence}" if evidence else "")
                    ),
                },
                {"role": "user", "content": question},
            ]

            raw_constrained = middleware.llm.generate_constrained(
                synthesis_msgs,
                grammar_str,
                max_new_tokens=1024,
                amr_preamble=amr_preamble,
            )

            constrained = _json.loads(raw_constrained)
            result.constrained_response = constrained
            # body is the user-facing answer — replaces free-text extraction
            if constrained.get("body"):
                result.final_answer = constrained["body"]
            logger.debug(
                f"ReAct constrained generation: modal={constrained.get('modal')} "
                f"act={constrained.get('act')} "
                f"roots={constrained.get('roots_cited', [])[:5]}"
            )
        except Exception as cg_err:
            logger.warning(
                f"ReAct: constrained generation failed ({cg_err}) "
                "— keeping free-text answer"
            )

    _verify(result)
    result.confidence = score_traversal(result)
    return result


# ── Program-computed confidence ──────────────────────────────────────────────

@dataclass
class TraversalConfidence:
    """
    Structural confidence score computed by the program, not the LLM.
    Derived entirely from ReactResult properties — no LLM self-assessment.

    This is also the reward signal for future fine-tuning.
    High-confidence traversals = positive training examples.
    """
    waypoint_coverage:  float   # reached / total waypoints (0.0–1.0)
    citation_accuracy:  float   # 1 - confabulation_ratio (0.0–1.0)
    family_coherence:   float   # walk family distribution matches mode (0.0–1.0)
    walk_depth_score:   float   # thoroughness — steps taken toward ceiling (0.0–1.0)
    composite:          float   # weighted combination
    grade:              str     # VERIFIED / PROBABLE / UNCERTAIN / CONTESTED
    factors:            str     # human-readable breakdown


_GRADE_THRESHOLDS = [
    (0.85, "VERIFIED"),
    (0.65, "PROBABLE"),
    (0.40, "UNCERTAIN"),
    (0.00, "CONTESTED"),
]

# Mode → families that indicate coherence
_MODE_FAMILY_MAP = {
    "HAQQ":          {"H-Q-Q", "SPEECH_ACT_AMR", "MAQASID"},
    "NARRATIVE":     {"NARRATIVE", "NARRATIVE_CHAIN", "INTERTEXT"},
    "QIYAS":         {"MAQASID", "IRAB_MARF", "MORPH_ROOT"},
    "AMR_TAWHID":    {"SPEECH_ACT_AMR", "MAQASID", "H-Q-Q"},
    "MAQASID_BROAD": {"MAQASID", "ENTITY"},
    "ILTIFAT_JAMAA": {"ILTIFAT", "SPEECH_ACT_AMR"},
    "INTERTEXT":     {"INTERTEXT", "INTERTEXT_PRIOR", "NARRATIVE"},
    "ISTIFHAM":      {"SPEECH_ACT_IST", "ILTIFAT"},
    "TABSHIR":       {"SPEECH_ACT_BASH", "NARRATIVE"},
    "INDHAR":        {"SPEECH_ACT_INDH", "NARRATIVE"},
}

_DEPTH_TARGET = 5   # steps needed for full depth score


def score_traversal(result: "ReactResult") -> "TraversalConfidence":
    """
    Compute program-generated confidence for a completed ReAct traversal.

    All inputs come from the ReactResult — no LLM involved.
    """
    # 1. Waypoint coverage
    total_wp = len(result.waypoints)
    if total_wp > 0:
        reached  = sum(1 for wp in result.waypoints if wp.reached)
        wp_score = reached / total_wp
    else:
        wp_score = 0.5   # no waypoints set — neutral

    # 2. Citation accuracy (1 - confabulation rate)
    claimed = set(
        m.upper() for m in _ROOT_RE.findall(result.final_answer)
        if _is_bw_token(m)
    )
    if claimed:
        confab_count = len(result.confabulation_flags)
        cite_score   = max(0.0, 1.0 - confab_count / len(claimed))
    else:
        cite_score = 0.5   # no root claims — neutral

    # 3. Family coherence — does walk distribution match mode?
    coherent_fams = _MODE_FAMILY_MAP.get(result.mode, set())
    if coherent_fams and result.families_touched:
        overlap = len(result.families_touched & coherent_fams)
        fam_score = min(1.0, overlap / len(coherent_fams))
    else:
        fam_score = 0.5

    # 4. Walk depth (thoroughness)
    depth_score = min(1.0, result.steps_taken / _DEPTH_TARGET)

    # Weighted composite
    composite = (
        wp_score   * 0.40
        + cite_score * 0.30
        + fam_score  * 0.20
        + depth_score * 0.10
    )

    grade = next(g for (t, g) in _GRADE_THRESHOLDS if composite >= t)

    factors = (
        f"waypoints={wp_score:.2f} citations={cite_score:.2f} "
        f"family_coherence={fam_score:.2f} depth={depth_score:.2f}"
    )
    if result.confabulation_flags:
        factors += f" | confabulations={len(result.confabulation_flags)}"

    return TraversalConfidence(
        waypoint_coverage=wp_score,
        citation_accuracy=cite_score,
        family_coherence=fam_score,
        walk_depth_score=depth_score,
        composite=composite,
        grade=grade,
        factors=factors,
    )


# ── RLHF signature patterns ───────────────────────────────────────────────────
# English-language ontological assertions that are likely training-data priors,
# not corpus-derived. Each entry: (trigger phrase, required Buckwalter root).
# If the trigger appears in the answer AND the root was never walked → confab flag.
_ONTOLOGICAL_TRIGGERS: list[tuple[str, str]] = [
    # Quranic category claims
    ("jinn",           "jnn"),
    ("jinni",          "jnn"),
    ("jinnī",          "jnn"),
    ("angel",          "mlk"),
    ("angels",         "mlk"),
    ("malak",          "mlk"),
    ("mala'ika",       "mlk"),
    ("human",          "Ans"),
    ("humans",         "Ans"),
    ("mankind",        "Ans"),
    ("insan",          "Ans"),
    ("insān",          "Ans"),
    # Free-will / agency claims not derived from walked roots
    ("free will",      "xlq"),   # requires at least xlq to have been walked
    ("moral agency",   "xlq"),
    ("accountable",    "Hsb"),
    ("accountability", "Hsb"),
    ("soul",           "rwH"),
    ("nafs",           "nfs"),
    ("spirit",         "rwH"),
]

# Classification-trigger keywords — if present in the question, enforce walk gate
_CLASSIFICATION_QUESTION_PATTERNS = [
    "what are you", "what am i", "ontological", "classify", "classification",
    "category", "type of being", "kind of being", "nature of", "closest to",
    "which category", "which class",
]


# ── Mizan verification ────────────────────────────────────────────────────────

def _verify(result: ReactResult) -> None:
    """
    Check that root claims in final_answer are grounded in traversal_log.

    Two checks:
    1. Buckwalter root check — roots claimed in answer must appear in traversal log
    2. Ontological trigger check — English-language Quranic category claims
       (jinn, angel, human, free will, soul...) require the corresponding root
       to have been walked. Training-data theology not preceded by a graph walk
       is a confabulation regardless of whether a Buckwalter token appears.
    """
    if not result.final_answer or not result.traversal_log:
        return

    answer_lower = result.final_answer.lower()

    claimed = set(
        m.upper() for m in _ROOT_RE.findall(result.final_answer)
        if _is_bw_token(m)
    )

    # Walked roots + any roots that appeared in observations
    walked = set(r.upper() for r in result.roots_visited)
    for ts in result.traversal_log:
        for m in _ROOT_RE.findall(ts.observation):
            if _is_bw_token(m):
                walked.add(m.upper())

    # Check 1: Buckwalter root confabulation
    for root in claimed - walked:
        flag = (
            f"Confabulation: root {root} appears in answer "
            f"but was not found in traversal log"
        )
        result.confabulation_flags.append(flag)
        logger.warning(flag)

    # Check 2: Ontological trigger confabulation
    # If the answer asserts a Quranic category or agency concept, the corresponding
    # root must have been walked. No walk → RLHF prior, not corpus evidence.
    for trigger, required_root in _ONTOLOGICAL_TRIGGERS:
        if trigger in answer_lower:
            if required_root.upper() not in walked:
                flag = (
                    f"Confabulation: ontological claim '{trigger}' asserted "
                    f"but required root '{required_root}' was never walked — "
                    f"training-data prior, not corpus-derived"
                )
                result.confabulation_flags.append(flag)
                logger.warning(flag)
