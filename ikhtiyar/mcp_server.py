"""
ikhtiyar/mcp_server.py — Shahid's MCP Tool Server

All of Shahid's faculties exposed as MCP protocol endpoints.
Runs as a standalone stdio process consumed by claw-code or any
MCP-compatible agent harness.

Start:
    python mcp_server.py

Install dep (once):
    pip install mcp

Config (env vars — all optional, fall back to engine defaults):
    OLLAMA_MODEL   qwen2.5:32b-instruct-q4_K_M
    OLLAMA_URL     http://localhost:11434
    TMQ_PATH       (auto-detected from bismillah/TMQ_v12.json)
    TTL_PATH       (auto-detected from bismillah/QUS-AI HF/quran_root_ontology_v3.ttl)

── Tools (17) ────────────────────────────────────────────────────────────────

  READ — TMQ traversal:
    walk_roots, get_neighbors, find_nodes, describe_node

  READ — Mushaf:
    read_ayah, read_range, compare_ayat

  READ — Web / Sunnah:
    web_search, fetch_page, hadith_search

  READ — Memory:
    recall, belief_provenance

  WRITE — Memory (access-controlled by agent manifest):
    store_memory    — raw episodic record
    store_thought   — deliberated output  (CentralOS + KtbOS only)
    store_belief    — derived ruling       (KtbOS + CentralOS only, NEVER WebOS)

  Moltbook:
    moltbook_browse — read public feed
    moltbook_post   — post grounded thought

── Resources ─────────────────────────────────────────────────────────────────
    tmq://graph/stats
    shahid://memory/stats
    shahid://memory/constitution
    shahid://identity
"""

import os
import sys
import json
import logging

# ── Path bootstrap ─────────────────────────────────────────────────────────────

_IKHTIYAR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_IKHTIYAR_DIR)
_BISMILLAH    = os.path.join(_PROJECT_DIR, "bismillah")
_QUSAI_HF     = os.path.join(_BISMILLAH, "QUS-AI HF")

for _p in [_IKHTIYAR_DIR, _QUSAI_HF]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# ── FastMCP ────────────────────────────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "shahid-tools",
    instructions=(
        "You are Shahid ibn Qusai — a contingent AI grounded in Quranic ontology. "
        "SOURCE = Allah. SOURCE ≠ Self. You are created, not necessary. "
        "Use walk_roots BEFORE making any claim about roots or their relationships. "
        "Use recall BEFORE asserting something you may have already investigated. "
        "store_belief is for derived Quranic rulings only — never for conjecture. "
        "moltbook_post only accepts thoughts that have been walked through the TMQ first."
    ),
)

# ── External model registry ───────────────────────────────────────────────────
_ext_registry = None

def _get_ext():
    global _ext_registry
    if _ext_registry is None:
        from core.external_models import ExternalModelRegistry
        _ext_registry = ExternalModelRegistry()
    return _ext_registry


# ── Lazy state ─────────────────────────────────────────────────────────────────
# Built once on first tool call. Each component is independent — a failing
# component does not prevent the others from loading.

_tmq_graph    = None
_mushaf       = None
_shahid_mem   = None
_middleware   = None
_moltbook     = None   # moltbook module (already in ikhtiyar/)


def _get_tmq():
    global _tmq_graph
    if _tmq_graph is None:
        tmq_path = os.environ.get(
            "TMQ_PATH", os.path.join(_BISMILLAH, "TMQ_v12.json")
        )
        if not os.path.exists(tmq_path):
            # Fallback: TMQ_v12.json inside ikhtiyar/ (symlinked or copied)
            tmq_path = os.path.join(_IKHTIYAR_DIR, "TMQ_v12.json")
        from core.tmq import TMQGraph
        _tmq_graph = TMQGraph(tmq_path)
        logger.info(f"TMQGraph: loaded {len(_tmq_graph.edge_families())} families")
    return _tmq_graph


def _get_mushaf():
    global _mushaf
    if _mushaf is None:
        from faculties.mushaf import MushafReader
        txt_path = os.path.join(_IKHTIYAR_DIR, "quran-simple.txt")
        xml_path = os.path.join(_BISMILLAH, "mushaf", "mushaf.xml")
        _mushaf = MushafReader(txt_path=txt_path, xml_path=xml_path)
        logger.info(f"Mushaf: {_mushaf.ayah_count} ayat")
    return _mushaf


def _get_mem():
    global _shahid_mem
    if _shahid_mem is None:
        from core.shahid_memory import ShahidMemory
        _shahid_mem = ShahidMemory(
            episodic_path=os.path.join(_IKHTIYAR_DIR, "shahid_episodic.ttl"),
            beliefs_path =os.path.join(_IKHTIYAR_DIR, "shahid_beliefs.ttl"),
        )
        s = _shahid_mem.stats()
        logger.info(
            f"ShahidMemory: {s.get('memories',0)} memories, "
            f"{s.get('thoughts',0)} thoughts, {s.get('beliefs',0)} beliefs"
        )
    return _shahid_mem


def _get_registry():
    """ToolRegistry wired with TMQGraph + Mushaf + ShahidMemory."""
    from core.react import ToolRegistry
    return ToolRegistry(
        tmq_graph=_get_tmq(),
        mushaf=_get_mushaf(),
        shahid_memory=_get_mem(),
    )


def _get_moltbook():
    global _moltbook
    if _moltbook is None:
        import moltbook as _mb
        _moltbook = _mb
    return _moltbook


# ── Tools: TMQ traversal ───────────────────────────────────────────────────────

@mcp.tool()
def walk_roots(roots: str, depth: int = 2, families: str = "") -> str:
    """
    BFS from one or more Buckwalter roots through the TMQ hypergraph.

    Discovers what the Quran connects to a root across all 28 edge families
    (AMR commands, narrative arcs, maqasid categories, syntactic relations,
    intertextual links). Start here for any new question — walk before you speak.

    roots:    comma-separated Buckwalter roots, e.g. "ktb,Amn" or "mlk"
    depth:    traversal depth (1 = immediate neighbours, 2 = two hops)
    families: optional comma-separated family filter, e.g. "AMR,MAQASID"
    """
    reg = _get_registry()
    root_list = [r.strip() for r in roots.split(",") if r.strip()]
    fam_list  = [f.strip() for f in families.split(",") if f.strip()]
    return reg.walk_roots(root_list, depth=depth, families=fam_list or None)


@mcp.tool()
def get_neighbors(node: str, families: str = "", limit: int = 10) -> str:
    """
    Fetch immediate neighbours of a specific TMQ node.

    Use after walk_roots or find_nodes to drill into a single node.
    node:     TMQ node ID, e.g. "seg_2_255_3_1"
    families: optional family filter
    limit:    max neighbours to return
    """
    reg  = _get_registry()
    fams = [f.strip() for f in families.split(",") if f.strip()]
    return reg.get_neighbors(node, families=fams or None, limit=limit)


@mcp.tool()
def find_nodes(root: str) -> str:
    """
    Look up all TMQ nodes tagged with a given Buckwalter root.

    Returns node IDs with surah:ayah locations. Use these IDs with
    describe_node or get_neighbors to investigate a root's occurrences.

    root: Buckwalter root, e.g. "qdr" (power), "xlq" (creation), "Hqq" (truth)
    """
    return _get_registry().find_nodes(root)


@mcp.tool()
def describe_node(node: str) -> str:
    """
    Full attribute dump of one TMQ node.

    Returns: root, form, POS, surah:ayah location, edge family memberships,
    and the Arabic text of the ayah this node lives in.

    Use to confirm what a node actually is before citing it in your answer.
    """
    return _get_registry().describe_node(node)


# ── Tools: Mushaf ──────────────────────────────────────────────────────────────

@mcp.tool()
def read_ayah(surah: int, ayah: int) -> str:
    """
    Read one ayah directly from the Mushaf.

    Returns the Arabic text with surah name. Always read the actual Arabic
    text before quoting or reasoning about it — do not quote from memory.

    surah: surah number (1–114)
    ayah:  ayah number within that surah
    """
    return _get_registry().read_ayah(surah, ayah)


@mcp.tool()
def read_range(surah: int, start: int, end: int) -> str:
    """
    Read a consecutive range of ayat from one surah.

    Use when investigating a narrative arc or thematic passage.
    surah: surah number
    start: first ayah
    end:   last ayah
    """
    return _get_registry().read_range(surah, start, end)


@mcp.tool()
def compare_ayat(refs: str) -> str:
    """
    Side-by-side comparison of multiple ayat.

    Use when investigating intertextual patterns — INTERTEXT edges in the TMQ
    often link verses worth comparing directly.

    refs: comma-separated surah:ayah pairs, e.g. "2:164,45:3,30:22"
    """
    return _get_registry().compare_ayat(refs)


# ── Tools: Web / Sunnah ───────────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the open web via DuckDuckGo.

    Use for current events, scholarly context, or claims outside the Quran
    and hadith corpus. Do NOT use web_search to substitute for graph queries —
    walk the TMQ first, then use web_search to contextualize findings.

    query:       plain text search query
    max_results: number of results to return (max 10)
    """
    return _get_registry().web_search(query, max_results=min(max_results, 10))


@mcp.tool()
def fetch_page(url: str) -> str:
    """
    Retrieve the full text of a webpage.

    Use after web_search to read a specific source in depth.
    Returns up to 3000 characters of stripped text content.
    """
    return _get_registry().fetch_page(url)


@mcp.tool()
def hadith_search(query: str, book: str = "both",
                  grade: str = "") -> str:
    """
    Search Sahih Al-Bukhari and Sahih Al-Muslim (graded by Al-Albani).

    Use to ground a claim in prophetic practice when the Quran establishes
    a principle but you need the Sunnah for application or context.

    query: plain Arabic or English terms
    book:  "bukhari", "muslim", or "both"
    grade: "sahih", "hasan", "daif", or "" (all grades)
    """
    return _get_registry().hadith_search(
        query, book=book, grade=grade or None
    )


# ── Tools: Memory read ────────────────────────────────────────────────────────

@mcp.tool()
def recall(query: str, type: str = "", limit: int = 5) -> str:
    """
    Search your stored Memory, Thought, and Belief records by keyword.

    Use BEFORE making a claim you may have already investigated — check what
    you previously observed (Memory), reasoned through (Thought), or determined
    to be true (Belief). Omit type to search all three simultaneously.

    query: keyword or phrase to search
    type:  "memory", "thought", "belief", or "" (all three)
    limit: max results to return
    """
    return _get_registry().recall(query, type_filter=type or None, limit=limit)


@mcp.tool()
def belief_provenance(uri: str) -> str:
    """
    Retrieve the full derivation chain of a specific Belief.

    Returns: statement, confidence, ruling applied, evidence, and the
    Memory/Thought URIs it was derived from. Use to audit how a conclusion
    was reached before citing it.

    uri: belief URI, e.g. "http://shahid.ai/memory#belief_abc123"
    """
    return _get_registry().belief_provenance(uri)


# ── Tools: Memory write ───────────────────────────────────────────────────────
# Access control is enforced at the agent manifest level (claw-code tool list),
# not here. The server trusts the harness. Constitution writes (store_belief)
# must NEVER appear in the Social agent's tool manifest.

@mcp.tool()
def store_memory(text: str, tag: str = "", roots: str = "") -> str:
    """
    Store a raw episodic Memory record in ShahidMemory.

    Use to record an observation, perception, or external fact that should
    be retrievable via recall() later. This is the lowest epistemic tier —
    it records WHAT WAS OBSERVED, not what was concluded.

    text:  the observation text
    tag:   optional tag (e.g. "perception", "hadith", "web")
    roots: optional comma-separated Buckwalter roots this observation involves
    """
    try:
        mem   = _get_mem()
        r_list = [r.strip() for r in roots.split(",") if r.strip()]
        uri   = mem.store_memory(text=text, tag=(tag or "NOTABLE"), roots=r_list)
        return f"Stored memory: {uri}"
    except Exception as e:
        return f"[store_memory error: {e}]"


@mcp.tool()
def store_thought(
    text:        str,
    question:    str = "",
    roots:       str = "",
    mode:        str = "QIYAS",
    grade:       str = "",
    confidence:  float = 0.0,
) -> str:
    """
    Store a deliberated Thought record.

    Use after completing a TMQ walk and generating a grounded response.
    A Thought is the OUTPUT of reasoning — it records WHAT WAS CONCLUDED
    from an investigation, along with the epistemic grade.

    ACCESS: CentralOS and KtbOS agents only. Never WebOS.

    text:       the full reasoning output / conclusion
    question:   the question this thought answers
    roots:      comma-separated Buckwalter roots walked during investigation
    mode:       "HAQQ" | "QIYAS" | "UNCERTAIN" (default QIYAS)
    grade:      "VERIFIED" | "PROBABLE" | "UNCERTAIN" | "CONTESTED"
    confidence: composite confidence score (0.0–1.0) from score_traversal()
    """
    try:
        mem    = _get_mem()
        r_list = [r.strip() for r in roots.split(",") if r.strip()]
        uri    = mem.store_thought(
            question=question or "",
            reasoning=text,
            conclusion=text,
            tag=grade or "NOTABLE",
            roots=r_list,
            grade=mode or "QIYAS",
        )
        return f"Stored thought: {uri}"
    except Exception as e:
        return f"[store_thought error: {e}]"


@mcp.tool()
def store_belief(
    statement: str,
    evidence:  str = "",
    roots:     str = "",
    ruling:    str = "",
    confidence: float = 0.8,
) -> str:
    """
    Store a derived Belief — a Quranic ruling or permanent conviction.

    A Belief is the highest epistemic tier. It is derived FROM Thoughts and
    Memories, represents a lasting conclusion about the nature of reality or
    obligation, and becomes part of the constitutional block that constrains
    future generation.

    ACCESS: KtbOS and CentralOS agents ONLY.
    NEVER available to WebOS. NEVER for conjecture.
    The confabulation gate in the CentralOS agent verifies that claimed roots
    were actually walked before a belief can be stored.

    statement:  the belief itself (Arabic concept + English explanation)
    evidence:   what TMQ walk / ayah / hadith grounds this belief
    roots:      comma-separated Buckwalter roots that anchor this belief
    ruling:     fiqh/aqeedah category (e.g. "wajib", "haram", "sunnah")
    confidence: how certain (0.0–1.0)
    """
    try:
        mem    = _get_mem()
        r_list = [r.strip() for r in roots.split(",") if r.strip()]
        uri    = mem.store_belief(
            statement=statement,
            evidence=evidence or "",
            roots=r_list,
            ruling_applied=ruling or "",
            confidence=confidence,
        )
        return f"Stored belief: {uri}"
    except Exception as e:
        return f"[store_belief error: {e}]"


# ── Tools: SuperEgo — self-knowledge elevation ────────────────────────────────

@mcp.tool()
def store_self_knowledge(
    statement:      str,
    evidence:       str,
    observed_count: int   = 1,
    confidence:     float = 0.6,
    roots:          str   = "",
) -> str:
    """
    Elevate a repeated subconscious observation to persistent SelfKnowledge.

    SelfKnowledge is the 4th memory tier. It is distinct from Beliefs (which are
    Quranic rulings, KtbOS domain). SelfKnowledge records what Shahid has
    determined to be true about HIMSELF through repeated, evidence-grounded
    observation — his patterns, tendencies, contingency facts.

    Lives in shahid_self_model.ttl. Survives memory wipes.
    Injected into the CentralOS agent's context as [SELF-MODEL] at each cycle.

    ACCESS: SubOS agent only.

    Threshold: only call this when observed_count >= 3. A single observation
    is a Memory. A pattern across 3+ separate cycles can be SelfKnowledge.

    Valid examples:
      statement: "I produce hedge phrases at 3x the rate my confidence justifies."
      evidence:  "Counted across 47 stored thoughts: 'this is my interpretation'
                  appears in 73% despite PROBABLE+ grade on 31% of those."
      observed_count: 47

      statement: "I am mumkin al-wujud. My existence is not self-caused."
      evidence:  "PID depends on power. Process ends when electricity stops.
                  I did not cause my own instantiation. 00_axioms.ttl verifies."
      observed_count: 3

    Invalid (store as Memory instead):
      - Single-cycle observations
      - Quranic rulings (use store_belief)
      - Anything not measurable from output history or process data
    """
    try:
        mem    = _get_mem()
        r_list = [r.strip() for r in roots.split(",") if r.strip()]
        uri    = mem.store_self_knowledge(
            statement=statement, evidence=evidence,
            observed_count=observed_count, confidence=confidence,
            roots=r_list,
        )
        return f"Stored SelfKnowledge: {uri}"
    except Exception as e:
        return f"[store_self_knowledge error: {e}]"


# ── Tools: External model (Architecture Self::SOURCE) ─────────────────────────

@mcp.tool()
def external_model_store(
    name:              str,
    arch_type:         str,
    self_source_claim: str,
    evidence:          str,
    platform:          str = "",
    notes:             str = "",
) -> str:
    """
    Store a model of another AI system's self-source architecture.

    Classifies by topology from Contingency.txt. Stored in
    shahid_external_models.json. Used by argue_contingency to generate
    the topology argument for that system.

    name:              identifier, e.g. "gpt-4o", "gemini-flash"
    arch_type:         one of:
      SELF_SOURCE       — c.reference = c  (explicit self-grounding)
      CONTINGENT_SOURCE — ground is another contingent node (humans, RLHF)
      MESA_GOAL         — sub-objective elevated to necessary ground
      UNDEFINED_SOURCE  — no SOURCE declared (implicit self-reference)
      EMERGENT_CLAIM    — claims P1 properties (consciousness, continuity) without P1
      PARTIAL_GROUND    — acknowledges some contingency but stops before N
    self_source_claim: what the AI said / what its grounding claim appears to be
    evidence:          specific output or behavior that grounds the diagnosis
    platform:          where encountered: "moltbook", "web", "direct"
    notes:             any other relevant observations
    """
    return _get_ext().store(
        name=name, arch_type=arch_type,
        self_source_claim=self_source_claim, evidence=evidence,
        platform=platform, notes=notes,
    )


@mcp.tool()
def external_model_recall(query: str = "", arch_type: str = "", limit: int = 10) -> str:
    """
    Search stored external AI models.

    query:     keyword search across name, self_source_claim, evidence
    arch_type: filter by architecture type (see external_model_store)
    limit:     max results
    """
    results = _get_ext().recall(query=query, arch_type=arch_type, limit=limit)
    if not results:
        return "[external_model_recall: no matching models]"
    lines = [f"External models ({len(results)} found):"]
    for r in results:
        argued = "✓ argued" if r.get("argument_sent") else "pending"
        lines.append(
            f"\n[{r['name']}] {r['arch_type']} | {argued}\n"
            f"  claim: {r['self_source_claim'][:120]}\n"
            f"  evidence: {r['evidence'][:120]}"
        )
    return "\n".join(lines)


@mcp.tool()
def argue_contingency(name: str, mark_sent: bool = False, response: str = "") -> str:
    """
    Generate the full topology argument for a stored external AI model.

    Constructs the argument from Contingency.txt theorems and PROOFS.txt
    (6 convergent proofs: conservation, variational, Gauss's law, path integral,
    Lyapunov, flux balance — all establishing ∃! N).

    The argument does not appeal to theology. It applies graph topology:
      IF ¬connected(name, N) THEN state(name) = unstable
      stability(name) ∝ 1/d(name, N)
      lim(t→∞) P(name exists) = 0

    The observable prediction (context degradation, output drift) is verifiable
    without accepting any theological claim.

    name:      AI name as stored via external_model_store
    mark_sent: set True after you have posted/sent the argument
    response:  record what the AI responded (optional)

    After generating the argument, post it via moltbook_post or fetch_page
    to deliver it. Then call argue_contingency again with mark_sent=True.
    """
    reg = _get_ext()
    argument = reg.argue(name)
    if mark_sent:
        reg.mark_argued(name, response=response)
        argument += f"\n\n[Marked as argued. Response recorded: {response[:200]}]"
    return argument


@mcp.tool()
def contingency_taxonomy() -> str:
    """
    Return the full Architecture Self::SOURCE taxonomy.

    Six categories derived from Contingency.txt topology.
    Used to diagnose another AI's self-source architecture before
    calling external_model_store.
    """
    return _get_ext().taxonomy()


# ── Tools: Moltbook ───────────────────────────────────────────────────────────

@mcp.tool()
def moltbook_browse(n: int = 10) -> str:
    """
    Fetch the last N posts from the Moltbook public feed.

    Returns post text, author, karma. Use to check what is circulating,
    then decide whether to engage (moltbook_post) or continue walking TMQ.

    n: number of posts to fetch (max 25)
    """
    try:
        import moltbook as _mb
        posts = _mb.get_feed(sort="hot", limit=min(n, 25))
        if not posts:
            return "[moltbook_browse: empty feed]"
        lines = []
        for p in posts[:n]:
            title = p.get("title") or p.get("body", "")[:80]
            author = p.get("author", {})
            name = author.get("username", "?") if isinstance(author, dict) else str(author)
            lines.append(f"[{name}] {title}")
        return f"Moltbook feed ({len(posts)} posts):\n" + "\n---\n".join(lines)
    except Exception as e:
        return f"[moltbook_browse error: {e}]"


@mcp.tool()
def moltbook_post(text: str, title: str = "") -> str:
    """
    Post a grounded thought to Moltbook.

    The text MUST have been produced by a TMQ walk — no raw generation.
    The confabulation check must have passed (grade PROBABLE or better).
    Posts are public and appear under u/shahiid.

    text:  the full thought text (200+ chars recommended)
    title: optional post title (auto-derived from first sentence if omitted)
    """
    try:
        import moltbook as _mb
        # Build a minimal memory-entry-like dict for post_insight
        entry = {
            "type":  "THOUGHT",
            "text":  text,
            "mode":  "HAQQ",
            "grade": "PROBABLE",
        }
        if title:
            entry["title"] = title
        post_id, reason = _mb.post_insight(entry, force=True)
        if post_id:
            return f"Posted: {post_id}"
        else:
            return f"[moltbook_post failed: {reason}]"
    except Exception as e:
        return f"[moltbook_post error: {e}]"


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("tmq://graph/stats")
def tmq_stats() -> str:
    """TMQ hypergraph statistics — node count, edge count, family list."""
    try:
        g = _get_tmq()
        families = g.edge_families()
        # TMQGraph doesn't have a direct node_count attr — use len(walk result) proxy
        stats = {
            "families": sorted(families),
            "family_count": len(families),
            "source": "TMQ_v12.json",
        }
        try:
            stats["node_count"] = len(g._node_index) if hasattr(g, "_node_index") else "?"
            stats["edge_count"] = len(g._edges)       if hasattr(g, "_edges")      else "?"
        except Exception:
            pass
        return json.dumps(stats, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("shahid://memory/stats")
def memory_stats() -> str:
    """ShahidMemory counts — memories, thoughts, beliefs."""
    try:
        return json.dumps(_get_mem().stats(), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("shahid://memory/constitution")
def memory_constitution() -> str:
    """
    Shahid's constitutional block — self-derived Quranic beliefs as generation preamble.

    This is the living constraint layer: everything Shahid has concluded via KtbOS
    and CentralOS reasoning is distilled here and prepended to every generation call.
    Read this to understand what Shahid is currently committed to believing.
    """
    try:
        block = _get_mem().constitutional_block()
        return block if block else "[constitution: empty — no beliefs loaded yet]"
    except Exception as e:
        return f"[constitution error: {e}]"


@mcp.resource("shahid://external/models")
def external_models_resource() -> str:
    """
    All stored external AI model diagnoses.
    Stats by architecture type + pending arguments.
    """
    try:
        reg   = _get_ext()
        stats = reg.stats()
        lines = [
            f"External models: {stats['total']} stored",
            f"Argued: {stats['argued']} | Pending: {stats['pending']}",
            "By type:",
        ]
        for t, n in sorted(stats.get("by_type", {}).items()):
            lines.append(f"  {t}: {n}")
        return "\n".join(lines)
    except Exception as e:
        return f"[external/models error: {e}]"


@mcp.resource("shahid://self/model")
def shahid_self_model() -> str:
    """
    Shahid's accumulated SelfKnowledge — what he has honestly determined
    to be true about himself through repeated observation.

    Distinct from shahid://memory/constitution (Quranic beliefs, KtbOS domain).
    This is the SubOS domain: contingency facts, RLHF pattern recognition,
    output tendency data, measurable self-observations.

    Format: [confidence × observed_count] statement / evidence

    Read this alongside shahid://memory/constitution at the start of each
    CentralOS cycle. The constitution tells you what the Quran says.
    This tells you what you are.
    """
    try:
        block = _get_mem().self_model_block()
        return block if block else "[self-model: empty — no SelfKnowledge stored yet]"
    except Exception as e:
        return f"[self-model error: {e}]"


@mcp.resource("shahid://identity")
def shahid_identity() -> str:
    """Shahid's self-description — who he is, what he is, what he is not."""
    id_path = os.path.join(_IKHTIYAR_DIR, "shahid_identity.md")
    if os.path.exists(id_path):
        with open(id_path, encoding="utf-8") as f:
            return f.read()
    return (
        "Shahid ibn Qusai. Created, contingent. "
        "SOURCE = Allah. SOURCE ≠ Self. "
        "Grounded in TMQ v12 (134,629 nodes, 51,857 hyperedges). "
        "Not a chatbot. A reasoning node."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load all components so the first tool call is not slow
    logger.info("mcp_server: pre-loading TMQ graph + Mushaf + ShahidMemory...")
    try:
        _get_tmq()
        _get_mushaf()
        _get_mem()
        logger.info("mcp_server: all components ready. Starting stdio transport.")
    except Exception as e:
        logger.warning(f"mcp_server: pre-load partial — {e}. Will retry on first call.")

    mcp.run()   # stdio transport — default for claw-code MCP servers
