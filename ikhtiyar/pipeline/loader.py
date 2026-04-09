import os
import logging
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ── Backend selector ──────────────────────────────────────────────────────────
#
# Primary: Ollama running locally (GPU/CPU split).
# GBNF grammar enforcement is logit-level — only possible on Ollama/llama.cpp.
# Override model or URL via env vars:
#   OLLAMA_MODEL   default: qwen2.5:32b-instruct-q4_K_M
#   OLLAMA_URL     default: http://localhost:11434
#
# Cloud APIs (Groq, OpenRouter, Cerebras) are NOT in this path.
# Groq is used only in moltbook_agent.py as a stateless social relay.

_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
_OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")


def load_model() -> "ModelInterface":
    """
    GBNF-capable model for all generation paths.
    14B verified to support grammar constraints. Generation without GBNF
    is not acceptable — an unconstrained model reasons from RLHF weights.
    """
    logger.info(f"Backend: Ollama ({_OLLAMA_MODEL} @ {_OLLAMA_URL})")
    return OllamaModel(model_name=_OLLAMA_MODEL, base_url=_OLLAMA_URL)


# ── Abstract interface ────────────────────────────────────────────────────────

class ModelInterface(ABC):
    @abstractmethod
    def generate(self, prompt, max_new_tokens: int = 256) -> str:
        pass

    @abstractmethod
    def generate_constrained(
        self,
        prompt,
        grammar_str: str,
        max_new_tokens: int = 512,
        amr_preamble: str = "",
    ) -> str:
        """
        Generate with a GBNF grammar constraint at logit level.
        Tokens violating the grammar receive -inf logit before sampling.
        Only OllamaModel enforces this structurally; other backends fall back
        to prompt-level AMR preamble injection.
        """
        pass

    @abstractmethod
    def load(self):
        pass


# ── Ollama (primary — logit-level GBNF) ──────────────────────────────────────

class OllamaModel(ModelInterface):
    """
    Locally running Ollama instance.
    Required: ollama serve && ollama pull qwen2.5:32b-instruct-q4_K_M

    Hardware: RTX 3080 (10GB VRAM) + Ryzen 7 3700X (32GB RAM).
    ~10GB on GPU, ~10GB offloaded to CPU RAM. ~5-8 tok/s — acceptable for
    Shahid's 20s reasoning cycle.
    """

    def __init__(self, model_name: str = _OLLAMA_MODEL,
                 base_url: str = _OLLAMA_URL):
        self.model_name = model_name
        self.base_url   = base_url.rstrip("/")

    def load(self):
        logger.info(f"OllamaModel ready: {self.model_name} @ {self.base_url}")

    def generate(self, prompt, max_new_tokens: int = 512) -> str:
        """Standard generation for user-facing chat."""
        messages = prompt if isinstance(prompt, list) \
            else [{"role": "user", "content": prompt}]
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":   self.model_name,
                    "messages": messages,
                    "options": {"num_predict": max_new_tokens,
                                "temperature": 0.7, "top_p": 0.9},
                    "stream":  False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            return f"Error: {e} (Is Ollama running? ollama serve)"

    def generate_raw(self, prompt, max_new_tokens: int = 512) -> str:
        """Raw generation for autonomous reasoning — higher temp, penalizes repetition."""
        messages = prompt if isinstance(prompt, list) \
            else [{"role": "user", "content": prompt}]
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":   self.model_name,
                    "messages": messages,
                    "options": {"num_predict": max_new_tokens,
                                "temperature": 0.9, "top_p": 0.95,
                                "repeat_penalty": 1.3},
                    "stream":  False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama generate_raw error: {e}")
            return f"Error: {e} (Is Ollama running? ollama serve)"

    def generate_constrained(
        self,
        prompt,
        grammar_str: str,
        max_new_tokens: int = 512,
        amr_preamble: str = "",
    ) -> str:
        """
        Logit-level GBNF constraint via Ollama's grammar option.

        Every token that violates grammar_str receives -inf logit before
        sampling — violations are structurally ungenerable, not filtered.

        grammar_str:  GBNF string from GBNFCompiler.compile(walk_grammar)
        amr_preamble: AMR standing orders from Quran's 1205 command edges —
                      injected as system prompt prefix (prompt-level layer).
        """
        messages = prompt if isinstance(prompt, list) \
            else [{"role": "user", "content": prompt}]

        if amr_preamble:
            injected = False
            updated  = []
            for m in messages:
                if m.get("role") == "system" and not injected:
                    updated.append({
                        "role":    "system",
                        "content": amr_preamble + "\n\n" + m["content"],
                    })
                    injected = True
                else:
                    updated.append(m)
            if not injected:
                updated = [{"role": "system", "content": amr_preamble}] + messages
            messages = updated

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":   self.model_name,
                    "messages": messages,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": 0.7,
                        "top_p":       0.9,
                        "grammar":     grammar_str,   # logit-level GBNF
                    },
                    "stream":  False,
                },
                timeout=360,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama constrained generation error: {e}")
            return f"Error: {e}"


