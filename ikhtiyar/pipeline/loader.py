import os
import logging
import requests
from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        pass
    
    @abstractmethod
    def load(self):
        pass

class InferenceAPIModel(ModelInterface):
    """
    Uses the Hugging Face Serverless Inference API.
    Accesses 70B+ models using the Pro Subscription benefits.
    """
    def __init__(self, model_id: str, api_token: str = None):
        self.model_id = model_id
        # Use provided token or fallback to environment variable
        self.token = api_token or os.environ.get("HF_TOKEN")
        self.client = None

    def load(self):
        if not self.token:
            logger.warning("⚠️ No HF_TOKEN found! Rate limits will be low (Free Tier). Add HF_TOKEN to Space secrets for Pro speeds.")
        
        logger.info(f"Connecting to Serverless Inference API: {self.model_id}")
        self.client = InferenceClient(model=self.model_id, token=self.token)
        logger.info("✓ API Client Ready")

    def generate(self, prompt: str | list, max_new_tokens: int = 512) -> str:
        """Standard generation for user-facing chat (full RLHF behavior)."""
        if not self.client:
            self.load()

        try:
            messages = prompt
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stream=False
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"API Generation Error: {e}")
            return f"Error: {e} (Check HF_TOKEN or Model Status)"

    def generate_raw(self, prompt: str | list, max_new_tokens: int = 512) -> str:
        """
        Raw generation for autonomous reasoning.

        Higher temperature + frequency/presence penalties = less formulaic,
        more diverse, breaks repetition loops.
        """
        if not self.client:
            self.load()

        try:
            messages = prompt
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.9,           # Higher = more diverse token selection
                top_p=0.95,                 # Wider nucleus = more variety
                frequency_penalty=0.7,      # Penalize repeated phrases heavily
                presence_penalty=0.5,       # Encourage new vocabulary/concepts
                stream=False
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Raw Generation Error: {e}")
            return f"Error: {e} (Check HF_TOKEN or Model Status)"


class OllamaModel(ModelInterface):
    """
    Uses a locally running Ollama instance.
    Requires: ollama serve + ollama pull <model_name>
    """
    def __init__(self, model_name: str = "qwen2.5:14b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def load(self):
        logger.info(f"OllamaModel ready: {self.model_name} @ {self.base_url}")

    def generate(self, prompt: str | list, max_new_tokens: int = 512) -> str:
        """Standard generation for user-facing chat."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "options": {"num_predict": max_new_tokens, "temperature": 0.7, "top_p": 0.9},
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama Generation Error: {e}")
            return f"Error: {e} (Is Ollama running? Try: ollama serve)"

    def generate_raw(self, prompt: str | list, max_new_tokens: int = 512) -> str:
        """Raw generation for autonomous reasoning — higher temperature, penalizes repetition."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "repeat_penalty": 1.3,
                    },
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama Raw Generation Error: {e}")
            return f"Error: {e} (Is Ollama running? Try: ollama serve)"