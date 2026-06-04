import os
import logging
from abc import ABC, abstractmethod
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from threading import Thread
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        pass
    
    @abstractmethod
    def load(self):
        pass

class TransformersModel(ModelInterface):
    """
    GPU-Accelerated Loader using Hugging Face Transformers.
    Designed for HF Spaces with ZeroGPU (A100).
    """
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.model = None
        self.tokenizer = None
        self.is_ready = False

    def load(self):
        try:
            logger.info(f"Loading {self.repo_id} on GPU...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            self.is_ready = True
            logger.info("✓ Model Loaded Successfully (GPU/Transformers)")
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not self.is_ready:
            return "[Model Not Loaded]"
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation Error: {e}")
            return f"Error: {e}"


class HFInferenceModel(ModelInterface):
    """
    API-based inference using HuggingFace Inference API.
    Designed for serverless deployment (Render, Railway, etc.)
    """
    def __init__(self, repo_id: str, api_token: str):
        if not HF_API_AVAILABLE:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

        self.repo_id = repo_id
        self.api_token = api_token
        self.client = None
        self.is_ready = False

    def load(self):
        try:
            logger.info(f"Initializing HF Inference API client for {self.repo_id}...")
            self.client = InferenceClient(model=self.repo_id, token=self.api_token)
            self.is_ready = True
            logger.info("✓ HF Inference API client ready")
        except Exception as e:
            logger.error(f"Failed to initialize HF API client: {e}")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not self.is_ready:
            return "[Model Not Loaded]"

        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            return response.strip()
        except Exception as e:
            logger.error(f"API Generation Error: {e}")
            return f"Error: {e}"
