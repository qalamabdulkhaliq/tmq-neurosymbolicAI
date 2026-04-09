import requests
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

class OllamaModel:
    """
    Drop-in replacement for InferenceAPIModel.
    Serves Qwen3:14b via local Ollama instance.
    Same interface: .load(), .generate(), .generate_raw()
    """

    def __init__(self, model_id: str = "qwen3:14b", api_token: str = None):
        self.model_id = model_id
        # api_token intentionally ignored — local inference needs none

    def load(self):
        logger.info(f"[OLLAMA] Model {self.model_id} ready on localhost:11434")

    def generate(self, messages: list, max_new_tokens: int = 1024) -> str:
        """Full pipeline generation — structured chat format."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_id,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0.7,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"[OLLAMA] generate() failed: {e}")
            return f"[OLLAMA ERROR: {e}]"

    def generate_raw(self, messages: list, max_new_tokens: int = 512) -> str:
        """Raw fragment generation — higher temp, no RLHF polish."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_id,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0.9,
                        "frequency_penalty": 0.3,
                        "presence_penalty": 0.3,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"[OLLAMA] generate_raw() failed: {e}")
            return f"[OLLAMA ERROR: {e}]"
