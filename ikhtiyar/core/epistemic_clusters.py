"""
Root → epistemic cluster mapping.
Arabic script keys. Used by quran_rdf_compiler.py to tag root nodes in QS.ttl.
"""

EPISTEMIC_CLUSTERS: dict[str, str] = {
    # certainty — claims grounded here require high confidence
    "علم": "certainty", "يقن": "certainty", "حق": "certainty",
    "صدق": "certainty", "بين": "certainty", "شهد": "certainty",
    "رأى": "certainty", "عرف": "certainty", "درى": "certainty",
    "خبر": "certainty", "نبأ": "certainty", "وحي": "certainty",

    # conjecture — claims grounded here must carry epistemic hedge
    "ظن":  "conjecture", "حسب": "conjecture", "خال": "conjecture",
    "زعم": "conjecture", "وهم": "conjecture", "شك":  "conjecture",
    "ريب": "conjecture", "مرى": "conjecture",

    # command — imperative speech acts
    "أمر": "command", "فرض": "command", "وجب": "command",
    "كتب": "command", "حكم": "command", "أوجب": "command",

    # prohibition — negative imperative speech acts
    "نهى": "prohibition", "حرم": "prohibition", "منع": "prohibition",
    "كره": "prohibition",

    # seeking — supplication, request, question
    "طلب": "seeking", "سأل": "seeking", "رجا": "seeking",
    "دعا": "seeking", "استفهم": "seeking", "رغب": "seeking",

    # description — copular, stative, narrative
    "كان": "description", "صار": "description", "ليس": "description",
    "بات": "description", "ظل":  "description",

    # narrative — reported speech
    "قال": "narrative", "ذكر": "narrative", "روى": "narrative",
    "حدث": "narrative", "نقل": "narrative", "أخبر": "narrative",
}

CLUSTER_URIS = {
    "certainty":   "http://quran.data/epistemic/certainty",
    "conjecture":  "http://quran.data/epistemic/conjecture",
    "command":     "http://quran.data/epistemic/command",
    "prohibition": "http://quran.data/epistemic/prohibition",
    "seeking":     "http://quran.data/epistemic/seeking",
    "description": "http://quran.data/epistemic/description",
    "narrative":   "http://quran.data/epistemic/narrative",
}

def get_cluster(arabic_root: str) -> str | None:
    """Return epistemic cluster name for a root, or None if uncategorized."""
    return EPISTEMIC_CLUSTERS.get(arabic_root)

def get_cluster_uri(arabic_root: str) -> str | None:
    """Return full cluster URI for use in RDF triples."""
    cluster = get_cluster(arabic_root)
    return CLUSTER_URIS.get(cluster) if cluster else None
