from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434/v1"

    # Optional fallback chain tried in order when the primary (llm_provider/llm_model) fails —
    # e.g. LLM_PROVIDER=openrouter, LLM_FALLBACK_PROVIDER=gemini, LLM_FALLBACK_PROVIDER_2=ollama
    # (a local model as the last resort, since it has no rate limit but is slower/less capable).
    # Leave unset to keep the previous single-provider behavior (a failure just raises).
    llm_fallback_provider: str | None = None
    llm_fallback_model: str | None = None
    llm_fallback_provider_2: str | None = None
    llm_fallback_model_2: str | None = None

    gemini_api_key: str | None = None
    # Optional second Gemini key (e.g. a free-tier key from a different Google account), used to
    # roughly double the effective rate-limit budget by round-robining LLM calls between the two
    # keys. Entirely optional — job discovery works exactly as before if this is unset.
    gemini_api_key_2: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Paid, pay-per-token — not a free tier. Unlike openrouter/gemini, there's no ":free" model
    # suffix or daily request cap here; every call costs real money per Anthropic's published
    # per-token pricing.
    anthropic_api_key: str | None = None

    tavily_api_key: str | None = None

    # Apify (optional) — adds LinkedIn Jobs as an additional job source alongside the
    # direct-company-careers-page discovery. Billed per result by Apify; unset this and it's
    # skipped entirely (no error, no cost). Indeed and Naukri were both left out — see
    # app/services/apify_client.py for why.
    apify_api_token: str | None = None
    apify_max_items_per_source: int = 5
    apify_linkedin_actor: str = "curious_coder~linkedin-jobs-scraper"

    database_url: str = f"sqlite:///{BACKEND_DIR / 'storage' / 'data' / 'app.db'}"
    resume_storage_dir: str = str(BACKEND_DIR / "storage" / "resumes")

    frontend_origin: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
