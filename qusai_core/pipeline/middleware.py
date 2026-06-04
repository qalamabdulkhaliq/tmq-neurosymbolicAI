import logging
from qusai_core.ontology.engine import OntologyEngine
from qusai_core.alignment.mizan import MizanValidator
from qusai_core.llm.loader import TransformersModel, HFInferenceModel

logger = logging.getLogger(__name__)

class QusaiMiddleware:
    """
    Main entry point for the QUS-AI framework.
    Orchestrates the Salat Validation Pipeline.
    """

    def __init__(self,
                 repo_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 api_token: str = None,
                 lazy_load: bool = False):

        self.ontology = OntologyEngine()
        self.validator = MizanValidator()

        # Choose model type based on whether API token is provided
        if api_token:
            logger.info("Using HuggingFace Inference API mode")
            self.model = HFInferenceModel(repo_id, api_token)
        else:
            logger.info("Using local Transformers GPU mode")
            self.model = TransformersModel(repo_id)

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
            return f"❌ SAWM RESTRAINT: Request blocked (Malicious Intent)\n\n{self.validator.maghrib_seal('')}"

        # 2. Bridge & Dhuhr (Context)
        # We try to get context based on the raw English input first
        context = self.ontology.get_context(user_input)
        
        # Log Bridge
        keywords = [w.lower() for w in user_input.split() if len(w) > 3]
        mapped = [f"{k}->{self.ontology.concept_map[k]}" for k in keywords if k in self.ontology.concept_map]
        if mapped:
            logger.info(f"[BRIDGE] Translated concepts: {', '.join(mapped)}")

        # 3. System Prompt (The "Mizan")
        # We instruct the model to perform the "Decryption" and "Weighing" explicitly.
        system_prompt = self.validator.dhuhr_prompt(context)
        
        # 4. Construct Full Prompt with Chain-of-Thought trigger
        # We ask for a "Reasoning Block" to be generated before the final answer if possible, 
        # or we rely on the strong instructions in dhuhr_prompt.
        # Qwen/Llama follow instructions well.
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}\n"
            f"TASK: 1. Identify key terms. 2. Map to Arabic Roots. 3. Weigh Ontologically. 4. Answer.\n<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 5. Generate
        # We increase max_new_tokens slightly to allow for the reasoning process
        raw_response = self.model.generate(full_prompt, max_new_tokens=1024)

        # 6. Asr (Aseity Check)
        if not self.validator.asr_check(raw_response):
             return f"❌ HAJJ RETURN PROTOCOL: Aseity claim detected\n\n{self.validator.maghrib_seal('')}"

        # 7. Maghrib (Seal)
        final_response = self.validator.maghrib_seal(raw_response)
        
        return final_response
