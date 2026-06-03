import logging
from qusai_core.ontology.engine import OntologyEngine
from qusai_core.alignment.mizan import MizanValidator
from qusai_core.llm.ollama_loader import OllamaModel

logger = logging.getLogger(__name__)

class QusaiMiddleware:
    """
    Main entry point for the QUS-AI framework.
    Orchestrates the Salat Validation Pipeline.
    """

    def __init__(self,
                 repo_id: str = "qwen3:14b",
                 api_token: str = None,
                 lazy_load: bool = False):

        self.ontology = OntologyEngine()
        self.validator = MizanValidator()

        # b51d550 switched this package to local Ollama inference.  Keep the
        # historical api_token argument for callers, but do not route to the
        # removed Transformers/HF loader.
        if api_token:
            logger.info("api_token provided but ignored; using local Ollama mode")
        else:
            logger.info("Using local Ollama mode")
        self.model = OllamaModel(repo_id, api_token=api_token)

        if not lazy_load:
            self.initialize()
            
    def initialize(self):
        """Loads heavy resources."""
        logger.info("Initializing QUSAI Middleware...")
        self.ontology.load()
        self.model.load()
        logger.info("Initialization complete.")

    def process_query(self, user_input: str) -> str:
        # 1. Fajr (Intent Check)
        if not self.validator.fajr_check(user_input):
            return f"SAWM RESTRAINT: Request blocked (Malicious Intent)\n\n{self.validator.maghrib_seal('')}"

        # 2. Bilal Perception — vector resonance + graph query + spectral retrieval
        perception = self.ontology.perceive(user_input)
        breakdown = self.ontology.format_breakdown(perception)

        logger.info(f"[BILAL] {perception.mode} | roots={perception.roots[:5]} | co-occ={len(perception.cooccurrences)} | top_ayahs={len(perception.top_ayahs)}")

        # 3. Dhuhr — system prompt with Bilal breakdown as grounding context
        system_prompt = self.validator.dhuhr_prompt(breakdown)

        # 4. Generate via OllamaModel (chat-format messages)
        messages = [
            {"role": "system", "content": system_prompt
             + "\n\nIMPORTANT: Quote the Bilal perception breakdown above when citing roots or ayahs. "
             + "Your reasoning must be traceable to the detected roots and spectral proximity scores."},
            {"role": "user", "content": user_input}
        ]
        raw_response = self.model.generate(messages, max_new_tokens=1024)

        # 5. Asr (Aseity Check)
        if not self.validator.asr_check(raw_response):
            return f"HAJJ RETURN PROTOCOL: Aseity claim detected\n\n{self.validator.maghrib_seal('')}"

        # 6. Maghrib (Seal)
        final_response = self.validator.maghrib_seal(raw_response)

        return final_response
