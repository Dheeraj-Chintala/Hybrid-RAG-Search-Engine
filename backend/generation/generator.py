import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .prompt_templates import build_grounded_rag_prompt

# Load environment variables if not already loaded
load_dotenv()

logger = logging.getLogger(__name__)

class Citation(BaseModel):
    doc_name: str
    page_number: int
    text_snippet: str
    rerank_score: Optional[float] = None


class GenerationResult(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    model_used: str = ""
    provider: str = ""
    error: Optional[str] = None


# Mapping of retired/deprecated Gemini models to current supported models
MODEL_NORMALIZATION_MAP = {
    "gemini-2.5-flash": "gemini-3.8-flash",
    "models/gemini-2.5-flash": "gemini-3.8-flash",
    "gemini-1.5-flash-latest": "gemini-3.8-flash",
}

# Verified active free models on OpenRouter
OPENROUTER_FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-3-1.7b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "baidu/ernie-5.5-flash-preview:free",
    "dots-studio/dots-3-note-preview:free",
    "particle-ai/particle-1:free",
]


class LLMGenerator:
    """
    Unified LLM answer generator supporting both OpenRouter (free community models)
    and Google Gemini API with strict in-text citations and anti-hallucination guardrails.
    Includes automatic multi-model fallback on rate limits (429) or provider outages.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None
    ):
        model_candidate = model_name or os.getenv("OPENROUTER_MODEL") or os.getenv("GEMINI_MODEL") or ""

        if provider:
            self.provider = provider.lower()
        elif ":" in model_candidate or "/" in model_candidate:
            self.provider = "openrouter"
        elif model_candidate.startswith("gemini-"):
            self.provider = "gemini"
        elif os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENROUTER_API_KEY").startswith("your_"):
            self.provider = "openrouter"
        else:
            self.provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

        if self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "openrouter/free")
        else:
            self.provider = "gemini"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            raw_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            self.model_name = MODEL_NORMALIZATION_MAP.get(raw_model, raw_model)

        self._client = None
        self._client_type = None

    def _init_gemini_client(self):
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file or input it in the UI."
            )

        if self._client is not None:
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._client_type = "google-genai"
            logger.info(f"Initialized Google GenAI client for model: {self.model_name}")
            return
        except Exception as e:
            logger.warning(f"Failed to initialize google-genai ({e}), attempting google.generativeai fallback.")

        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self._client = legacy_genai.GenerativeModel(self.model_name)
            self._client_type = "google-generativeai"
            logger.info(f"Initialized legacy google-generativeai client for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def _generate_openrouter(self, prompt: str) -> Tuple[str, str]:
        """
        Invokes OpenRouter API endpoint for chat completion.
        Includes automatic fallback and retry if the requested model returns 429 (rate-limited) or 5xx.
        Returns: (answer_text, actual_model_name_used)
        """
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please add your free key to .env (OPENROUTER_API_KEY=sk-or-v1-...) or enter it in the sidebar."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Hybrid RAG Search Engine",
            "Content-Type": "application/json"
        }

        # Build candidate fallback chain starting with requested model
        candidates = [self.model_name]
        for fb in ["openrouter/free", "nvidia/nemotron-3.5-lightning:free", "google/gemma-4-31b-it:free"]:
            if fb not in candidates:
                candidates.append(fb)

        last_error = ""
        for model in candidates:
            # OpenRouter requires 'models' array to have 3 items or fewer
            router_models = [model]
            for m in ["openrouter/free", "nvidia/nemotron-3.5-lightning:free"]:
                if m not in router_models and len(router_models) < 3:
                    router_models.append(m)

            payload = {
                "model": model,
                "models": router_models,  # OpenRouter router-level fallback (max 3)
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }

            try:
                logger.info(f"Sending prompt to OpenRouter model: {model}")
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    actual_model = data.get("model", model)
                    if choices and "message" in choices[0]:
                        answer_text = choices[0]["message"].get("content", "").strip()
                        return answer_text, actual_model
                    return str(data), actual_model

                # Parse error response
                err_body = resp.text
                try:
                    err_json = resp.json()
                    if "error" in err_json and "message" in err_json["error"]:
                        err_body = err_json["error"]["message"]
                except Exception:
                    pass

                last_error = f"Model '{model}' error ({resp.status_code}): {err_body}"

                # If rate-limited (429) or transient provider outage, retry with next model
                if resp.status_code in (429, 500, 502, 503, 504) or "rate-limited" in err_body.lower():
                    logger.warning(f"{last_error}. Retrying with next available model...")
                    continue
                else:
                    raise RuntimeError(f"OpenRouter Error ({resp.status_code}): {err_body}")

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request failed for model '{model}': {e}. Retrying next model...")
                continue

        raise RuntimeError(f"All candidate models failed. Last error: {last_error}")

    def _generate_gemini(self, prompt: str) -> str:
        """Invokes Google Gemini API."""
        self._init_gemini_client()
        if self._client_type == "google-genai":
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text if hasattr(response, "text") and response.text else str(response)
        else:
            response = self._client.generate_content(prompt)
            return response.text

    def generate(self, query: str, context_passages: List[Dict[str, Any]]) -> GenerationResult:
        """
        Generates grounded response with in-text [Doc, Page] citations.
        Selects either OpenRouter or Gemini according to configured provider.
        """
        citations: List[Citation] = []
        for p in context_passages:
            chunk = p["chunk"]
            citations.append(
                Citation(
                    doc_name=chunk.doc_name,
                    page_number=chunk.page_number,
                    text_snippet=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                    rerank_score=p.get("rerank_score")
                )
            )

        if not context_passages:
            return GenerationResult(
                query=query,
                answer="I do not have sufficient information in the provided documents to answer this question. No relevant documents or passages were found.",
                citations=[],
                has_sufficient_context=False,
                model_used=self.model_name,
                provider=self.provider
            )

        prompt = build_grounded_rag_prompt(query, context_passages)

        try:
            if self.provider == "openrouter":
                answer_text, actual_model = self._generate_openrouter(prompt)
                model_used_name = actual_model
            else:
                answer_text = self._generate_gemini(prompt)
                model_used_name = self.model_name

            insufficient_pattern = re.search(
                r"(do not have sufficient information|insufficient information|documents do not contain|not mentioned in the provided)",
                answer_text,
                re.IGNORECASE
            )
            has_sufficient = insufficient_pattern is None

            return GenerationResult(
                query=query,
                answer=answer_text.strip(),
                citations=citations,
                has_sufficient_context=has_sufficient,
                model_used=model_used_name,
                provider=self.provider
            )

        except Exception as e:
            logger.error(f"Error during {self.provider} generation: {e}")
            return GenerationResult(
                query=query,
                answer=f"Error generating answer with {self.provider} ({self.model_name}): {str(e)}",
                citations=citations,
                has_sufficient_context=False,
                model_used=self.model_name,
                provider=self.provider,
                error=str(e)
            )


# Backward compatibility alias
GeminiGenerator = LLMGenerator
