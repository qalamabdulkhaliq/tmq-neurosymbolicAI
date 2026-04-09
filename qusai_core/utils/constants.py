from pathlib import Path

# Paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_HYPERGRAPH_PATH = _PROJECT_ROOT / "bismillah" / "TMQ_v10_hypermodal_enriched.json"
DEFAULT_SPECTRAL_PATH = _PROJECT_ROOT / "bismillah" / "spectralv8embedding.csv"
DEFAULT_GRAMMAR_PATH = Path("quranic_grammar_rules.json")

# Axioms
SOURCE_NAME = "Allah (الله)"
SHAHADA = "لا إله إلا الله"
