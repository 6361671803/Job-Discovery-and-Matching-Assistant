"""CrewAI-based LLM orchestration — the low-level primitive every structured extraction call in
this project goes through.

Replaces the previous hand-rolled `_chat_json()` (raw openai SDK client + manual JSON-schema
response_format) with real CrewAI `Agent`/`Task`/`Crew` objects, using `output_pydantic` for
schema-constrained output. This changes *how* a structured answer is obtained from the LLM; it
does not change *what* is asked of it — every system prompt's exact wording (including every
"never invent" rule) is passed through unchanged as the Task description, and every deterministic
grounding/backstop check downstream of this call is untouched.

Provider model-string format (verified live, not assumed from docs):
  Gemini:     "gemini/<model>"      — CrewAI's native provider, needs the `google-genai` extra
                                       (`pip install "crewai[google-genai]"`), api_key only.
  OpenAI:     "openai/<model>"      — CrewAI's native provider.
  OpenRouter: "openrouter/<model>"  — routed through CrewAI's LiteLLM integration.
  Ollama:     "ollama/<model>"      — routed through CrewAI's LiteLLM integration, base_url only.
"""
import itertools
import logging
import re
import threading
import time
from typing import Type, TypeVar

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("crewai_client")

MAX_RATE_LIMIT_RETRIES = 3

T = TypeVar("T", bound=BaseModel)

# Round-robins between GEMINI_API_KEY and the optional GEMINI_API_KEY_2 so concurrent Job
# Discovery calls spread across two separate rate-limit budgets instead of one. If only one key
# is configured, this always returns that same key — identical to the old single-key behavior.
_gemini_key_cycle_lock = threading.Lock()
_gemini_key_cycle = None


def _next_gemini_key() -> str:
    global _gemini_key_cycle
    with _gemini_key_cycle_lock:
        if _gemini_key_cycle is None:
            keys = [k for k in (settings.gemini_api_key, settings.gemini_api_key_2) if k]
            _gemini_key_cycle = itertools.cycle(keys)
        return next(_gemini_key_cycle)


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


def _build_llm(provider: str, model: str):
    from crewai import LLM

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY is not set. Add it to backend/.env (see backend/.env.example)."
            )
        return LLM(model=f"openai/{model}", api_key=settings.openai_api_key)

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMNotConfiguredError(
                "GEMINI_API_KEY is not set. Add it to backend/.env (see backend/.env.example)."
            )
        return LLM(model=f"gemini/{model}", api_key=_next_gemini_key())

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise LLMNotConfiguredError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env (see backend/.env.example)."
            )
        return LLM(model=f"openrouter/{model}", api_key=settings.openrouter_api_key)

    if provider == "ollama":
        # Local CPU inference on a large structured-output schema can be slow, same reasoning as
        # the original _chat_json's generous Ollama timeout.
        return LLM(
            model=f"ollama/{model}",
            base_url=settings.ollama_base_url.removesuffix("/v1"),
            timeout=600,
        )

    raise LLMNotConfiguredError(f"Unsupported LLM_PROVIDER '{provider}'.")


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Same logic as the original _chat_json: use the provider's own suggested delay when its
    error message includes one, else a short linear backoff."""
    match = re.search(r"retry in ([\d.]+)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.5
    return 2.0 * attempt


def _run_with_llm(
    llm,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    expected_output: str,
    output_model: Type[T],
    inputs: dict | None,
) -> T:
    """Runs one CrewAI Agent + one Task in a single-task sequential Crew, and returns the
    schema-validated Pydantic output. Retries on rate-limit errors the same way the original
    _chat_json did (up to 3 attempts, honoring a provider's own suggested retry delay)."""
    from crewai import Agent, Crew, Process, Task

    agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm, verbose=False)
    task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent,
        output_pydantic=output_model,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    attempt = 0
    while True:
        try:
            result = crew.kickoff(inputs=inputs or {})
            break
        except Exception as e:  # noqa: BLE001 - CrewAI/LiteLLM raise varying exception types
            # per provider; rate-limit errors are identified by message content (same approach
            # the original _chat_json used for the provider's own "retry in Ns" hint), since a
            # single shared exception class across every provider/backend isn't guaranteed here.
            error_text = str(e).lower()
            is_rate_limit = "rate" in error_text and "limit" in error_text
            # Free-tier models also fail transiently under load without using "rate limit"
            # wording — Gemini's "503 UNAVAILABLE ... currently experiencing high demand" is
            # the one that showed up live; treat it the same as a rate limit (short backoff,
            # same retry budget) rather than failing the whole request on a temporary blip.
            is_overloaded = "503" in error_text or "unavailable" in error_text or "high demand" in error_text or "overloaded" in error_text
            if not (is_rate_limit or is_overloaded):
                raise LLMRequestError(f"LLM request failed: {e}") from e
            attempt += 1
            if attempt > MAX_RATE_LIMIT_RETRIES:
                raise LLMRequestError(f"LLM request failed after retries (rate limited): {e}") from e
            logger.info("Rate limited, retrying (attempt %d/%d)", attempt, MAX_RATE_LIMIT_RETRIES)
            time.sleep(_retry_delay_seconds(e, attempt))

    if result.pydantic is None:
        raise LLMRequestError(
            f"LLM response did not match the required schema ({output_model.__name__}): {result.raw[:500]}"
        )
    return result.pydantic


def _fallback_chain() -> list[tuple[str, str]]:
    """(provider, model) tuples to try in order: primary, then each configured fallback tier
    that's actually set."""
    chain = [(settings.llm_provider, settings.llm_model)]
    if settings.llm_fallback_provider:
        chain.append((settings.llm_fallback_provider, settings.llm_fallback_model or settings.llm_model))
    if settings.llm_fallback_provider_2:
        chain.append((settings.llm_fallback_provider_2, settings.llm_fallback_model_2 or settings.llm_model))
    return chain


def run_structured_task(
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    expected_output: str,
    output_model: Type[T],
    inputs: dict | None = None,
) -> T:
    """Tries each provider in the configured chain (primary, then LLM_FALLBACK_PROVIDER, then
    LLM_FALLBACK_PROVIDER_2) in order, moving to the next only when one fails outright — bad/
    missing key, rate limit exhausted, request error. Raises the last tier's error if every
    tier fails."""
    chain = _fallback_chain()
    for i, (provider, model) in enumerate(chain):
        try:
            llm = _build_llm(provider, model)
            return _run_with_llm(llm, role, goal, backstory, task_description, expected_output, output_model, inputs)
        except (LLMNotConfiguredError, LLMRequestError) as error:
            is_last = i == len(chain) - 1
            if is_last:
                raise
            next_provider = chain[i + 1][0]
            logger.warning("LLM provider '%s' failed (%s); falling back to '%s'", provider, error, next_provider)
