"""
Unified LLM / VLM access.

The paper uses two families of models:

* **Vision-language models** (Gemini 2.5 Pro by default) for *failure reasoning*
  -- turning a sequence of frames into a trajectory + failure-reason description.
* **Reasoning LLMs** (OpenAI ``o4-mini`` by default) for *taxonomy discovery*
  and *trajectory assignment* -- clustering free-text failure reasons and
  assigning trajectories to clusters, including structured (function-calling)
  outputs.
* **Embedding models** (OpenAI ``text-embedding-3-large`` / SBERT) for the
  similarity-based baselines and evaluation metrics.

All of these are wrapped here behind a small, stable interface so that the
pipeline scripts never touch a raw SDK call and never see an API key. Keys are
read from the environment:

    GOOGLE_API_KEY      -- Gemini      (google-generativeai)
    OPENAI_API_KEY      -- OpenAI      (gpt-*, o*-series, embeddings)
    ANTHROPIC_API_KEY   -- Anthropic   (optional, LLM-judge evaluation)
    TOGETHER_API_KEY    -- Together AI (optional, Llama / DeepSeek baselines)

Set them once in ``~/.bashrc`` (see README) -- e.g. ``export OPENAI_API_KEY=...``.
"""

from __future__ import annotations

import functools
import logging
import os
import random
import time
from typing import Any, Callable, Iterable, Optional, Sequence, Type, TypeVar

logger = logging.getLogger("failure_taxonomy.llm")

# Default models used throughout the paper. Override per-call where needed.
DEFAULT_VLM_MODEL = "gemini-2.5-pro"
DEFAULT_LLM_MODEL = "o4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #
def retry_with_backoff(
    max_retries: int = 6,
    initial_delay: float = 4.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """Decorator: retry a function on *any* exception with exponential backoff.

    LLM/VLM endpoints fail transiently (rate limits, timeouts, 5xx). We keep the
    caught-exception set broad on purpose so this works identically across the
    OpenAI, Gemini and Anthropic SDKs without importing each one's error types.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 -- intentional broad retry
                    if attempt >= max_retries:
                        logger.error("Giving up after %d attempts: %s", attempt, exc)
                        raise
                    wait = delay * (0.5 + random.random() if jitter else 1.0)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                        func.__name__, attempt, max_retries, exc, wait,
                    )
                    time.sleep(wait)
                    delay *= exponential_base
        return wrapper

    return decorator


def _require_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise ValueError(
            f"No API key found in environment variable {env_var}. "
            f"Add `export {env_var}=...` to your ~/.bashrc (see README)."
        )
    return key


# --------------------------------------------------------------------------- #
# Gemini (vision-language model) -- Stage 1 failure reasoning
# --------------------------------------------------------------------------- #
_GENAI_CONFIGURED = False


def _gemini_key() -> str:
    """Gemini key, accepting either GOOGLE_API_KEY or GEMINI_API_KEY."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in "
            "your ~/.bashrc (see README)."
        )
    return key


def _configure_gemini() -> None:
    global _GENAI_CONFIGURED
    if _GENAI_CONFIGURED:
        return
    import google.generativeai as genai  # lazy import

    genai.configure(api_key=_gemini_key())
    _GENAI_CONFIGURED = True


def _is_openai_vlm(model: str) -> bool:
    """Route GPT / o-series ids to the OpenAI vision backend, Gemini otherwise."""
    m = model.lower()
    return m.startswith(("gpt", "o1", "o3", "o4", "chatgpt"))


@retry_with_backoff()
def vlm_generate(
    prompt: str,
    images: Optional[Sequence[Any]] = None,
    model: str = DEFAULT_VLM_MODEL,
) -> str:
    """Generate text from a vision-language model given a prompt and images.

    Supports Gemini (default) and OpenAI vision models -- the backend is chosen
    from the model id, so the paper's VLM comparison (Gemini vs. GPT / o-series)
    runs through one interface. Local open-weight VLMs (LLaVA, Qwen, Cosmos,
    RoboFAC) are run through their own HuggingFace pipelines and are out of scope
    for this hosted-API wrapper.

    Args:
        prompt: The instruction / chain-of-thought prompt.
        images: An ordered sequence of ``PIL.Image`` frames (may be empty/None
            for text-only calls).
        model: VLM model id (``gemini-*`` -> Gemini, ``gpt-*``/``o*`` -> OpenAI).

    Returns:
        The model's text response.
    """
    if _is_openai_vlm(model):
        from .io_utils import pil_to_base64  # local import to avoid cycles

        base64_images = [pil_to_base64(img) for img in (images or [])]
        messages = build_image_message(prompt, base64_images, detail="high")
        return llm_chat(messages, model=model)

    _configure_gemini()
    import google.generativeai as genai

    gen_model = genai.GenerativeModel(model)
    payload: list = [prompt]
    if images:
        payload.extend(images)

    response = gen_model.generate_content(payload)
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        logger.debug(
            "Gemini tokens -- in: %s out: %s total: %s",
            getattr(usage, "prompt_token_count", "?"),
            getattr(usage, "candidates_token_count", "?"),
            getattr(usage, "total_token_count", "?"),
        )
    return response.text


# --------------------------------------------------------------------------- #
# OpenAI (reasoning LLM) -- Stage 2 clustering, Stage 3 assignment, monitoring
# --------------------------------------------------------------------------- #
_OPENAI_CLIENT = None


def _openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI  # lazy import

        _OPENAI_CLIENT = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    return _OPENAI_CLIENT


@retry_with_backoff()
def llm_chat(
    messages: list[dict],
    model: str = DEFAULT_LLM_MODEL,
    **kwargs: Any,
) -> str:
    """Plain chat completion returning the assistant text.

    ``messages`` follow the OpenAI chat format. ``kwargs`` are forwarded to the
    API (e.g. ``temperature`` -- omit for o-series reasoning models).
    """
    client = _openai_client()
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def llm_prompt(prompt: str, model: str = DEFAULT_LLM_MODEL, **kwargs: Any) -> str:
    """Convenience wrapper for a single user-turn text prompt."""
    return llm_chat([{"role": "user", "content": prompt}], model=model, **kwargs)


def build_image_message(prompt: str, base64_images: Iterable[str], detail: str = "high") -> list[dict]:
    """Build a multimodal OpenAI ``messages`` list from base64-encoded images.

    Used by the runtime monitor, which sends a short window of frames to a
    vision-capable OpenAI model together with the monitoring prompt.
    """
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": detail}}
        for img in base64_images
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


T = TypeVar("T")


@retry_with_backoff()
def llm_structured(
    messages: list[dict],
    schema: Type[T],
    model: str = DEFAULT_LLM_MODEL,
) -> T:
    """Structured output: parse the model response into a pydantic model.

    Uses the OpenAI structured-outputs API (``responses``/``parse``) so the
    model is constrained to the given schema -- this replaces the older
    function-calling boilerplate scattered across the original scripts.

    Args:
        messages: Chat messages.
        schema: A ``pydantic.BaseModel`` subclass describing the desired output.
        model: Model id (must support structured outputs, e.g. o4-mini, gpt-4o).

    Returns:
        A validated instance of ``schema``.
    """
    client = _openai_client()
    completion = client.beta.chat.completions.parse(
        model=model, messages=messages, response_format=schema
    )
    return completion.choices[0].message.parsed


@retry_with_backoff()
def llm_embed(
    texts: Sequence[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100,
) -> list[list[float]]:
    """Return embedding vectors for a list of texts (batched)."""
    client = _openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        response = client.embeddings.create(model=model, input=batch)
        out.extend(item.embedding for item in response.data)
    return out
