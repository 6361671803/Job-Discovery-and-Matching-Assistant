# Job Discovery & Matching Assistant

A multi-agent AI system that automates the most time-consuming parts of a job search: discovering real companies and job openings, and matching them against a candidate's resume with an explainable score. That match screen — score, job description, job details, and skill match — is the end of the pipeline; the project does not select jobs, prepare applications, or touch a real application form in any way.

Built as a full-stack application: a Python/FastAPI backend orchestrating five single-responsibility agents through **CrewAI** (real `Agent`/`Task`/`Crew` objects, not a hand-rolled prompt wrapper), and a React single-page frontend guiding the user through the workflow end-to-end.

## Why This Exists

Manually checking dozens of individual company career pages and judging how well each opening matches your own background does not scale. Most company career sites are JavaScript-rendered SPAs, so a plain HTTP request cannot even read the listings. This project discovers directly from primary sources (the employer's own career page, rendered with a real browser) and explains every point of its match score, rather than handing back an opaque percentage.

## Core Design Principle

**The system never invents information.**

- Every job listing traces back to a link that was actually found on a real, rendered page.
- Every skill match is a literal string comparison against the resume's own extracted text — no LLM guessing.
- The pipeline stops at showing the candidate their ranked, explained matches. It does not select jobs, does not open or interact with any real application page, and does not touch a form field anywhere.

## How It Works

```
Resume Upload → Preferences → Company Discovery → Job Discovery → Matching & Ranking
```

1. **Resume Upload & Parsing** — extracts structured data (skills, education, experience, projects) from a PDF/DOCX resume without inventing anything not in the source document.
2. **Preferences** — collects work-mode, city, and experience preferences, or infers likely target roles from the resume via LLM.
3. **Company Discovery** — finds real companies and their official career pages via live web search (Tavily), with a deterministic filter that rejects place names (e.g. a city mis-extracted as a "company") and government/administrative bodies.
4. **Job Discovery** — renders each company's career page with a real headless browser (Playwright), extracts individual job listings (never navigation/category links), follows through to the real ATS board when the landing page is just a search widget, then visits each job's own detail page for its actual requirements. Optionally augmented with LinkedIn listings via Apify.
5. **Matching & Ranking** — scores every job with a six-factor weighted formula: skills (deterministic whole-word matching), semantic similarity (real Gemini embeddings + cosine similarity), education, experience, and role fit (LLM judgment, grounded in the job's own text), and location (deterministic rule-based comparison). The results screen — match score, job description, job details, and skill match — is the final thing the project does.

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, FastAPI, Uvicorn, SQLAlchemy + SQLite |
| Agent Framework | **CrewAI** — every structured LLM call runs as a real `Agent` + `Task` inside a `Crew`, with `output_pydantic` enforcing the response shape |
| Resume Parsing | pypdf, python-docx |
| LLM (chat) | Configurable primary provider (Anthropic Claude, Google Gemini, OpenRouter, OpenAI, or local Ollama) with an optional two-tier fallback chain — e.g. Anthropic → Gemini → local Ollama — so a rate-limited or unavailable provider automatically falls through to the next rather than failing the request; routed through CrewAI's own provider layer (native for Anthropic/Gemini/OpenAI, LiteLLM-backed for OpenRouter/Ollama) |
| LLM (embeddings) | Google Gemini `gemini-embedding-001`, always used for semantic matching regardless of the chat provider |
| Prompt-Injection Guard | Fast free heuristic pre-filter + LLM safety check (via the app's own LLM_PROVIDER chain), applied only to scraped career-page text before it reaches the extraction LLM — the one place in this app where LLM input comes from a source the user doesn't control. Most pages never reach the LLM check at all |
| Web Search | Tavily Search API |
| Additional Job Source | Apify (LinkedIn actor), optional and opt-in |
| Browser Automation | Playwright (Chromium) |
| Frontend | React 19, Vite — no router library, no state-management library, no UI kit; hand-written CSS design system |

## Key Engineering Decisions

- **CrewAI orchestrates every LLM call, but never decides anything on its own** — each of the 7 structured-extraction calls in the app is one narrow, pre-defined `Agent` + `Task`, invoked at one specific point in a linear pipeline. No autonomous planning, no dynamic tool-calling loop.
- **JSON-schema-constrained LLM extraction** — every LLM call forces strict, schema-valid JSON output, never free-form prose.
- **Deterministic anti-hallucination backstops layered under every LLM call** — whole-word skill matching, URL grounding (a link must literally appear on the rendered page), date grounding (a date must be a verbatim substring of the page text), a hard-coded third-party job-board domain blocklist, and a place-name filter.
- **Two-stage job extraction** — listing-page fields (title/location/link) are extracted separately from detail-page fields (requirements/skills), since asking one call to guess fields that aren't actually on that page caused fabricated data during development.
- **Index-based link selection** — the model picks a link by its numeric position in a real, numbered list rather than typing a URL from memory, which was found to prevent fabricated-but-plausible-looking URLs.
- **Provider-agnostic LLM client with automatic rate-limit retry/backoff** — swap providers via one environment variable with zero code changes elsewhere.

## Project Structure

```
backend/
  app/
    main.py                    # All FastAPI REST endpoints
    config.py                  # Environment-based settings
    agents/                    # One agent per workflow phase
      resume_analyzer.py
      preference_agent.py
      company_discovery_agent.py
      job_discovery_agent.py
      matching_agent.py
    services/                  # Reusable, LLM-agnostic logic
      llm_client.py            # Every prompt/schema — calls into crewai_client.py
      crewai_client.py         # Real CrewAI Agent/Task/Crew orchestration + retry logic
      browser_client.py        # Playwright page rendering for discovery
      semantic_matcher.py      # Embeddings + cosine similarity
      apify_client.py          # Optional LinkedIn job source
      skill_matcher.py         # Deterministic skill matching
      ats_detector.py, date_utils.py, job_filter.py, ...
    db/models.py                # Candidate, Company, Job ORM models
    models/schemas.py           # Every Pydantic API schema
frontend/
  src/
    App.jsx                    # Top-level view/state machine
    api/client.js               # Every backend call as a typed fetch wrapper
    components/                 # One component per screen + shared UI pieces
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) [Ollama](https://ollama.com) if running a local LLM as your primary/fallback provider

### 1. Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python -m playwright install chromium

copy .env.example .env        # then fill in your own API keys
```

`requirements.txt` already includes `crewai[google-genai]`, since CrewAI's native Gemini provider (the default LLM here) needs that extra to work.

### 2. Configure `backend/.env`

Set `LLM_PROVIDER` to one of `openai` / `ollama` / `gemini` / `openrouter` / `anthropic` and fill in the matching API key. Note that `anthropic` is paid, pay-per-token with no free tier — unlike the other providers, every call costs real money (current cheapest option: `claude-haiku-4-5` at $1/$5 per 1M input/output tokens). Optionally set `LLM_FALLBACK_PROVIDER` (and `LLM_FALLBACK_PROVIDER_2`) to a different provider — if the primary fails (bad/missing key, rate limit exhausted, request error), the app automatically retries against the fallback chain in order before giving up. A Tavily key is required for company/job discovery. A Gemini key is required for semantic matching specifically (used independently of whichever chat provider you choose). Apify is optional (adds LinkedIn listings). See `backend/.env.example` for the full list — never commit real values.

The prompt-injection guard on scraped career-page text (`backend/app/services/guardrails_client.py`) first runs a free, instant heuristic filter — only text that trips it (phrasing like "ignore previous instructions") escalates to an LLM safety check through the same LLM_PROVIDER chain as everything else. Most real pages never reach the LLM check at all. It fails open (allows content through unchecked) if every configured provider is down, so this is a hardening layer, not a hard requirement to run the app.

### 3. Frontend setup

```bash
cd frontend
npm install
```

### 4. Run both servers together

```bash
npm run dev
```

This starts the backend (`uvicorn app.main:app --port 8000`) and frontend (`vite`, port 5173) together. Open **http://localhost:5173**.

## Honest Limitations

- No automated test suite — verification during development was done manually against real, live data throughout.
- Some sites (e.g. TCS, EPAM, Naukri) actively block automated browser access; those sources are either excluded or shipped with a documented caveat rather than a fake result.
- Job Discovery renders companies concurrently (bounded, via Playwright's async API against one shared browser) — an earlier attempt using Playwright's sync API from a thread pool caused a real 30+ minute hang, since that API isn't thread-safe; the async rewrite avoids that. LLM extraction calls stay capped to a small concurrency (tied to how many Gemini API keys are configured — one key can be round-robined with a second `GEMINI_API_KEY_2` to roughly double the rate-limit budget), since that shared rate limit, not the browser, is the real bottleneck. A full run against ~25 companies still takes a few minutes.
- Semantic matching requires a Gemini API key specifically; it's unavailable if only a non-Gemini provider is configured, with a documented, non-silent fallback to a 5-factor score.
- The prompt-injection guard fails open: if the LLM safety check itself errors (e.g. every configured provider is down), scraped page content is passed to the extraction LLM unchecked rather than blocking Job Discovery outright. It's a hardening layer, not a guarantee.
- Single-user, local application — no authentication/multi-user support, no production deployment configuration.
- Does not select jobs, prepare applications, open any real application page, or interact with a form field in any way — the project intentionally ends at the matched-jobs results screen. Applying for a job is entirely up to the user, outside this app, using the job/application links shown on each result.

## License

Personal project — no license specified.
