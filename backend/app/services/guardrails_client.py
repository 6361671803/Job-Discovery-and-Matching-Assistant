"""Prompt-injection guard for scraped career-page/job-listing text — the only place in this app
where content an LLM sees comes from a source the user doesn't control (a company's own website),
rather than the candidate's own resume or the app's own generated text.

Wired into app/agents/job_discovery_agent.py, right before the scraped page text is handed to
app/services/llm_client.py's extraction calls.

Uses NVIDIA NeMo Guardrails' built-in "self check input" rail (app/guardrails/), backed by a
local Ollama model rather than the OpenRouter/Gemini cloud tiers, so this check never competes
with the app's already rate-limited free-tier extraction calls for quota.
"""
import logging
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger("guardrails_client")

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "guardrails"

# The exact default refusal text NeMo Guardrails' built-in "self check input" rail uses when it
# blocks input (nemoguardrails/colang/v2_x/library/core.co: `bot refuse to respond`). Checked as
# a substring rather than an exact match in case surrounding formatting varies.
_REFUSAL_MARKER = "can't respond to that"

_rails: LLMRails | None = None


def _get_rails() -> LLMRails:
    global _rails
    if _rails is None:
        config = RailsConfig.from_path(str(_CONFIG_DIR))
        _rails = LLMRails(config)
    return _rails


async def is_scraped_content_safe(text: str) -> bool:
    """Returns False if the self-check-input rail flags this scraped page text as containing an
    instruction/jailbreak attempt aimed at the extraction LLM, rather than being genuine page
    content. Fails open (returns True) if the guardrails check itself errors — e.g. local Ollama
    isn't running — so a broken guardrails setup degrades to "no extra check" instead of taking
    down Job Discovery entirely."""
    if not text.strip():
        return True
    try:
        rails = _get_rails()
        response = await rails.generate_async(messages=[{"role": "user", "content": text}])
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        blocked = _REFUSAL_MARKER in content.lower()
        if blocked:
            logger.warning("Guardrails flagged scraped page content as a possible prompt injection")
        return not blocked
    except Exception as e:  # noqa: BLE001 - guardrails/Ollama failures must not break discovery
        logger.warning("Guardrails check failed (%s); allowing content through unchecked", e)
        return True
