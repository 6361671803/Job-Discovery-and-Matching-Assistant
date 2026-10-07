"""Every structured LLM extraction call in the application, now orchestrated through real CrewAI
Agent/Task/Crew objects (see crewai_client.py) instead of a hand-rolled OpenAI SDK client.

Every function below keeps its exact original name, parameters, and return shape (a plain dict),
so every calling agent (resume_analyzer.py, preference_agent.py, company_discovery_agent.py,
job_discovery_agent.py, matching_agent.py) needed zero changes for this migration. Every original
system prompt's exact wording — including every "never invent" rule — is preserved verbatim as
each CrewAI Agent's backstory. Every deterministic grounding/anti-hallucination backstop that ran
after the old raw LLM call runs identically here, unchanged, after the new CrewAI call.
"""
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.models.schemas import CandidateProfile
from app.services.crewai_client import LLMNotConfiguredError, LLMRequestError, run_structured_task

logger = logging.getLogger("llm_client")

__all__ = [
    "LLMNotConfiguredError",
    "LLMRequestError",
    "extract_candidate_profile",
    "infer_target_roles",
    "extract_company_names",
    "resolve_official_career_page",
    "extract_job_listings",
    "extract_job_description",
    "assess_job_match",
]


# ============================================================================
# 1. Resume -> structured candidate profile
# ============================================================================

# Kept deliberately short. Measured live against claude-sonnet-5: the original ~15-line version
# of these rules (same content, more verbose) made extraction unreliable — 0/9 and 0/15 in two
# separate batches came back completely empty (genuinely null, or wrapped in an unrequested
# extra key) despite the input resume text being fine. This condensed version, same substance,
# ran 5/5 clean on the same resume. Don't re-expand this without re-testing reliability — the
# failure mode is silent (no exception, just an empty-but-"valid" result) and easy to miss.
_EXTRACTION_BACKSTORY = (
    "Extract only information explicitly present in the resume text — never invent, infer, or "
    "embellish skills, experience, education, qualifiers, or dates. Use null (or an empty list) "
    "for anything not present. Only include a degree that literally appears in the resume — "
    "never add a second or higher degree the skills merely suggest. Never expand an abbreviation "
    "(e.g. \"RAG\", \"NLP\") beyond what the resume itself writes. Copy email/phone/URLs exactly "
    "as written; add \"https://\" to a bare domain but don't invent a URL that isn't present. "
    "Estimate years_of_experience only from dates/durations actually stated."
)


# Claude occasionally still returns a genuinely/wrongly empty profile despite good input (either
# all-null, or wrapped in an unrequested extra key) — root-caused to _EXTRACTION_BACKSTORY's
# length (see its comment); the condensed version measured 5/5 clean. Kept as cheap insurance
# against whatever residual rate remains, not as the primary fix.
_MAX_EMPTY_PROFILE_RETRIES = 3


def _looks_suspiciously_empty(profile: dict, resume_text: str) -> bool:
    """True if resume_text has real content but nothing meaningful was extracted — the
    signature of the wrapper-mismatch failure, not a genuinely sparse resume (a resume with
    literally no name, skills, experience, or education is implausible for real input)."""
    if len(resume_text.strip()) < 100:
        return False
    return not (
        profile.get("name")
        or profile.get("skills")
        or profile.get("experience")
        or profile.get("internships")
        or profile.get("education")
        or profile.get("projects")
    )


def extract_candidate_profile(resume_text: str) -> dict:
    """Calls the configured LLM provider to turn raw resume text into a structured profile dict."""
    profile: dict = {}
    for attempt in range(1, _MAX_EMPTY_PROFILE_RETRIES + 1):
        result = run_structured_task(
            role="Resume Data Extraction Specialist",
            goal="Extract structured candidate data from resume text with zero fabrication.",
            backstory=_EXTRACTION_BACKSTORY,
            task_description=f"Resume text:\n\n{resume_text}",
            # The attempt-number suffix varies the prompt slightly on each retry (outside the
            # resume_text block, so it can't be mistaken for resume content) — cheap insurance
            # against Anthropic's prompt cache making retries less independent than intended.
            expected_output=f"A complete candidate profile matching the required schema exactly, "
            f"with null/empty values for anything not explicitly present in the resume text. "
            f"(extraction pass {attempt})",
            output_model=CandidateProfile,
            inputs={"resume_text": resume_text},
        )
        profile = result.model_dump()
        if not _looks_suspiciously_empty(profile, resume_text):
            return profile
        logger.warning(
            "Resume extraction came back empty for non-trivial input (attempt %d/%d); retrying",
            attempt, _MAX_EMPTY_PROFILE_RETRIES,
        )
    return profile


# ============================================================================
# 2. Infer target roles from resume when the user didn't pick one
# ============================================================================

class _RoleInferenceResult(BaseModel):
    roles: list[str]


_ROLE_INFERENCE_BACKSTORY = """STRICT RULES:
- Do not restrict suggestions to software engineering roles — consider the candidate's actual
  background (e.g. finance, marketing, design, data, operations, HR, etc.) and suggest roles that
  fit whatever domain their skills/experience actually show.
- Every suggested role must be justifiable from the given skills/experience/projects — do not
  suggest roles that require skills or domain knowledge absent from the input.
- Return between 1 and 8 role titles, ranked most-suitable first."""


def infer_target_roles(profile: dict, experience_level: str) -> list[str]:
    """Suggests suitable job roles from resume-derived skills/experience when the user didn't pick one."""
    import json

    candidate_summary = {
        "skills": profile.get("skills", []),
        "technologies": profile.get("technologies", []),
        "experience": profile.get("experience", []),
        "internships": profile.get("internships", []),
        "projects": profile.get("projects", []),
        "education": profile.get("education", []),
        "certifications": profile.get("certifications", []),
        "years_of_experience": profile.get("years_of_experience"),
        "user_selected_experience_level": experience_level,
    }
    result = run_structured_task(
        role="Career Role-Fit Analyst",
        goal="Suggest suitable job role titles based only on the candidate's actual background.",
        backstory=_ROLE_INFERENCE_BACKSTORY,
        task_description=f"Candidate background:\n\n{json.dumps(candidate_summary, indent=2)}",
        expected_output="1 to 8 ranked, justifiable job role titles.",
        output_model=_RoleInferenceResult,
    )
    return result.roles


# ============================================================================
# 3. Extract real company names from web search results
# ============================================================================

class _CompanyNamesResult(BaseModel):
    companies: list[str]


_COMPANY_NAME_EXTRACTION_BACKSTORY = """STRICT RULES:
- Only include company names that literally appear in the provided search result titles or content.
- Never invent, guess, or embellish a company name.
- Exclude job boards / aggregators themselves (e.g. LinkedIn, Indeed, Naukri, Glassdoor, Monster,
  ZipRecruiter, AngelList, Wellfound, Instahyre) — those are sources, not employers.
- Exclude geographic place names (cities, states, provinces, countries, regions) even if they
  appear prominently in the results — e.g. "Hyderabad", "Telangana", "Bangalore", "India" are
  locations, never company names, even though they'll appear constantly since they describe
  where a job is based, not who is hiring.
- Exclude government departments, ministries, public service commissions, and recruitment
  boards (e.g. "Telangana State Public Service Commission", "Staff Selection Commission",
  "Department of School Education") — these publish exam/recruitment notifications, not a
  specific employer's job opening, and are out of scope for this job search.
- Return every distinct company name that genuinely appears in the results (do not artificially
  limit yourself to a small number) — there is no need to omit real companies to keep the list short."""


def extract_company_names(search_results: list[dict], role: str, location: str) -> list[str]:
    """Pulls real company names (only ones literally present in the search results) out of a search."""
    import json

    if not search_results:
        return []
    result = run_structured_task(
        role="Company Name Extraction Specialist",
        goal="Identify real, distinct hiring company names from web search results, and nothing else.",
        backstory=_COMPANY_NAME_EXTRACTION_BACKSTORY,
        task_description=f"Role: {role}\nLocation: {location}\n\nSearch results:\n\n"
        f"{json.dumps(search_results, indent=2)}",
        expected_output="Every distinct real company name that literally appears in the search results.",
        output_model=_CompanyNamesResult,
    )
    return result.companies


# ============================================================================
# 4. Resolve a company's official website and careers page from search results
# ============================================================================

class _CareerPageResult(BaseModel):
    official_website: str | None = None
    careers_url: str | None = None


_CAREER_PAGE_RESOLUTION_BACKSTORY = """STRICT RULES:
- You may ONLY return a URL that appears verbatim in the provided search results. Never fabricate,
  guess, construct, or modify a URL.
- Prefer the company's own domain over third-party job boards (LinkedIn, Indeed, Glassdoor, Naukri,
  Monster, ZipRecruiter, AngelList, Wellfound, Instahyre, etc.).
- If no result is confidently the company's own official site, set "official_website" to null.
- If no result is confidently the company's own official careers/jobs page, set "careers_url" to null
  — do NOT fall back to a third-party job board URL for this field."""

# Domains of third-party job boards/aggregators — never acceptable as a "careers_url", even if
# the model tries to return one (smaller local models don't always follow that instruction).
THIRD_PARTY_JOB_BOARD_DOMAINS = (
    "linkedin.com", "indeed.com", "glassdoor.com", "naukri.com", "monster.com",
    "ziprecruiter.com", "angel.co", "wellfound.com", "instahyre.com", "shine.com",
    "foundit.in", "timesjobs.com", "simplyhired.com",
)


def _is_third_party_job_board(url: str | None) -> bool:
    if not url:
        return False
    return any(domain in url.lower() for domain in THIRD_PARTY_JOB_BOARD_DOMAINS)


def resolve_official_career_page(company_name: str, search_results: list[dict]) -> tuple[str | None, str | None]:
    """Picks the official website/careers URL for a company out of search results — never fabricated."""
    import json

    if not search_results:
        return None, None
    result = run_structured_task(
        role="Official Company Page Resolver",
        goal="Identify a company's real official website and careers page from search results, "
        "using only URLs that literally appear in those results.",
        backstory=_CAREER_PAGE_RESOLUTION_BACKSTORY,
        task_description=f"Company: {company_name}\n\nSearch results:\n\n{json.dumps(search_results, indent=2)}",
        expected_output="The company's official_website and careers_url, or null for either if not confident.",
        output_model=_CareerPageResult,
    )
    website = result.official_website
    careers_url = result.careers_url

    # Belt-and-braces: never trust a URL the model didn't actually copy from the results.
    valid_urls = {r["url"] for r in search_results if r.get("url")}
    if website not in valid_urls:
        website = None
    if careers_url not in valid_urls:
        careers_url = None

    # Belt-and-braces #2: never accept a third-party job board as the "official" careers page,
    # even if the model returned one despite being told not to.
    if _is_third_party_job_board(website):
        website = None
    if _is_third_party_job_board(careers_url):
        careers_url = None

    return website, careers_url


# ============================================================================
# 5. Stage 1: extract job listings from a careers/listing page
# ============================================================================

class _JobListingEntry(BaseModel):
    title: str
    location: str | None = None
    work_mode: str | None = None
    link_index: int | None = None


class _JobListingsResult(BaseModel):
    jobs: list[_JobListingEntry]


# Stage 1: the careers/listing page itself usually only shows title, location, and an apply
# link per job — NOT requirements/qualifications/skills (those live on each job's own detail
# page). Asking the model to fill those fields from a listing page that doesn't have them caused
# it to hallucinate, so this stage only extracts what a listing page actually contains.
#
# Local models were also found to fabricate plausible-looking-but-fake URLs when asked to type
# out a full href (e.g. guessing a Greenhouse-style URL pattern with the wrong ID). Asking the
# model to instead pick a link by its INDEX in a numbered list is far more reliable for weaker
# models — it's a closed-set choice rather than free recall.
_JOB_LISTING_EXTRACTION_BACKSTORY = """You extract the list of INDIVIDUAL job postings shown
on a company's careers/jobs LISTING page (not an individual job description page).

CRITICAL — DO NOT CONFUSE NAVIGATION/CATEGORY LINKS WITH REAL JOB POSTINGS:
Many career pages mix real job postings together with site navigation, career-path categories,
and general program pages. You must extract ONLY genuine individual openings, each with its own
specific job title (e.g. "Software Engineer II", "Data Analyst - Mumbai", "Senior AI Developer").
Do NOT extract entries like these, even if they appear in a "careers" context — they are
navigation/category links, not job postings, and including them would send the candidate to the
wrong page:
  - Broad career-path or audience categories: "Students and graduates", "Experienced
    professionals", "Executives", "Professions", "Early Careers", "Graduate Programs"
  - Named leadership/development PROGRAMS rather than a specific role: "Leadership Program",
    "Accelerated Leadership Track", "Future Shapers", "Leadership University"
  - Generic site links: "Careers", "About Us", "Employee", "Life at [Company]", "Benefits"
  - Department or team names with no specific role attached: "Engineering", "Sales", "Marketing"
If a page consists mostly of this kind of navigation/category content and you cannot find
genuinely distinct individual job postings, return an EMPTY list rather than including any of
the above — an empty list is the correct, honest answer for that page.

STRICT RULES FOR ENTRIES YOU DO EXTRACT:
- Only extract jobs that are actually listed on this page. Do not invent job openings.
- Listing pages typically show only a title, a location, and an "Apply"/"View" link per job —
  do NOT try to guess responsibilities, requirements, qualifications, or skills; those aren't on
  this page. Only extract "title", "location", "work_mode", and "link_index".
- "link_index" must be the number (from the numbered links list below the page text) of the link
  that goes to this specific job. You may ONLY use a number that appears in that list. If you
  cannot find a matching link, use null. Never invent a link or URL yourself.
- "location" and "work_mode" must be null if not shown next to that specific listing — do not
  guess based on the company's general location.
- "work_mode" (Remote / Hybrid / On-site) must only be set if the page explicitly states it for
  that listing.
- Each job in your output must correspond to a genuinely distinct listing on the page. Never
  repeat the same job more than once.
- Extract at most 8 of the listings shown."""


def extract_job_listings(company_name: str, page_text: str, links: list[dict]) -> list[dict]:
    """Pulls the basic job listing (title/location/work_mode/job_url) from a careers page."""
    if not page_text.strip():
        return []

    numbered_links = [f"[{i}] {l['text']} -> {l['href']}" for i, l in enumerate(links) if l.get("href")]

    result = run_structured_task(
        role="Career Page Listing Analyst",
        goal="Extract only genuine individual job postings from a company careers page, "
        "never navigation or category links.",
        backstory=_JOB_LISTING_EXTRACTION_BACKSTORY,
        task_description=f"Company: {company_name}\n\nPage text:\n\n{page_text}\n\nNumbered links on page:\n\n"
        + "\n".join(numbered_links),
        expected_output="A list of at most 8 genuinely distinct job postings, each with a "
        "title, location, work_mode, and link_index grounded in the numbered links list.",
        output_model=_JobListingsResult,
    )
    jobs = [j.model_dump() for j in result.jobs]

    seen_titles: set[str] = set()
    deduped = []
    for job in jobs:
        idx = job.get("link_index")
        job["job_url"] = links[idx]["href"] if isinstance(idx, int) and 0 <= idx < len(links) else None
        job.pop("link_index", None)

        # Backstop: only trust an explicit "Remote"/"Hybrid"/"On-site" claim if that word also
        # appears in this listing's own location text — otherwise the model is likely guessing.
        location = (job.get("location") or "").lower()
        work_mode = (job.get("work_mode") or "").lower()
        if work_mode and work_mode.split()[0] not in location:
            job["work_mode"] = None

        key = (job.get("title") or "").strip().lower()
        if key and key not in seen_titles:
            seen_titles.add(key)
            deduped.append(job)

    return deduped


# ============================================================================
# 6. Stage 2: extract full job description from a job's own detail page
# ============================================================================

class _JobDetailResult(BaseModel):
    job_id: str | None = None
    location: str | None = None
    work_mode: str | None = None
    date_posted: str | None = None
    closing_date: str | None = None
    employment_type: str | None = None
    experience_requirement: str | None = None
    education_requirement: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: str | None = None
    qualifications: str | None = None


# Stage 2: for a single job's own detail page, extract the actual description fields. This page
# genuinely contains the requirements/qualifications, so asking for them here is well-grounded.
_JOB_DETAIL_EXTRACTION_BACKSTORY = """You extract structured details from a single job
posting's own description page.

STRICT RULES:
- Only use information explicitly present in the page text. Never invent, infer, or embellish
  requirements, qualifications, dates, or skills.
- If a field is not present on the page, use null (or an empty list for skill lists).
- "date_posted" and "closing_date" must be copied exactly as written on the page (e.g.
  "3 days ago", "2026-08-01"). Never calculate, estimate, or guess a date — use null if none is
  shown.
- "work_mode" (Remote / Hybrid / On-site) must only be set if the page explicitly states it.
- Separate "required_skills" (must-have) from "preferred_skills" (nice-to-have) based on how the
  page itself frames them (e.g. "Requirements" vs "Nice to have" / "Preferred")."""


def extract_job_description(job_title: str, page_text: str) -> dict:
    """Extracts requirements/qualifications/etc. from a job's own description page."""
    if not page_text.strip():
        return {}
    result = run_structured_task(
        role="Job Description Detail Extractor",
        goal="Extract the real requirements, qualifications, and dates from a single job's own detail page.",
        backstory=_JOB_DETAIL_EXTRACTION_BACKSTORY,
        task_description=f"Job title: {job_title}\n\nPage text:\n\n{page_text}",
        expected_output="Every requested field, populated only with information literally present on the page.",
        output_model=_JobDetailResult,
    )
    result_dict = result.model_dump()

    # Belt-and-braces: dates are load-bearing for the later posting-window filter, and small
    # local models have been observed to invent a plausible-looking date that isn't on the page
    # at all. Only trust a date the model claims if it's actually a verbatim substring of the page.
    for date_field in ("date_posted", "closing_date"):
        value = result_dict.get(date_field)
        if value and value not in page_text:
            result_dict[date_field] = None

    return result_dict


# ============================================================================
# 7. Resume/job match assessment (the judgment-based half of matching)
# ============================================================================

class _JobMatchAssessmentResult(BaseModel):
    education_verdict: Literal["meets", "partial", "does_not_meet", "unclear"]
    education_reasoning: str
    experience_verdict: Literal["meets", "partial", "does_not_meet", "unclear"]
    experience_reasoning: str
    role_relevance: Literal["strong", "moderate", "weak"]
    role_reasoning: str
    relevant_projects: list[str] = Field(default_factory=list)
    eligibility_issues: list[str] = Field(default_factory=list)
    why_this_matches: str


# Skills/technology matching and location matching are handled deterministically elsewhere
# (skill_matcher.py) since those claims are safety-critical and string comparison has zero
# hallucination risk. This call only handles the parts that genuinely need judgment: whether
# stated education/experience requirements are met, how relevant the role is, and eligibility
# notes explicitly written in the posting. It's given the already-computed matched skills so its
# "why this matches" text can reference them accurately instead of re-deriving (and potentially
# getting wrong) which skills actually matched.
_JOB_MATCH_ASSESSMENT_BACKSTORY = """You assess how well a candidate fits a specific job, based
ONLY on the candidate and job information given to you.

STRICT RULES:
- Base "education_verdict" only on the job's stated education_requirement vs the candidate's
  actual education entries. If the job doesn't state an education requirement, use "unclear".
- Base "experience_verdict" only on the job's stated experience_requirement vs the candidate's
  actual years_of_experience/experience entries. If the job doesn't state a requirement, use
  "unclear".
- "role_relevance" must be judged from the candidate's actual skills/experience/projects vs the
  job title and responsibilities — not from the job title alone.
- "relevant_projects" must ONLY contain project names copied verbatim from the candidate's
  project list given to you. Never invent a project or reference one not in that list.
- "eligibility_issues" must ONLY include issues EXPLICITLY stated in the job's own text (e.g.
  "must have current work authorization", "on-site only, no relocation assistance", "PhD
  required", "security clearance required"). Never invent a generic eligibility concern that
  isn't literally written in the job posting.
- "why_this_matches" must ONLY reference skills from the "matched_required_skills" and
  "matched_preferred_skills" lists given to you, and/or projects from "relevant_projects". Do not
  claim any skill or experience not explicitly given to you. Keep it to 2-3 sentences."""


def assess_job_match(
    profile: dict,
    preferences: dict,
    job: dict,
    matched_required_skills: list[str],
    matched_preferred_skills: list[str],
) -> dict:
    import json

    candidate_summary = {
        "education": profile.get("education", []),
        "years_of_experience": profile.get("years_of_experience"),
        "experience": profile.get("experience", []),
        "internships": profile.get("internships", []),
        "projects": [p.get("name") for p in profile.get("projects", []) if p.get("name")],
        "experience_level_selected": preferences.get("experience_level"),
    }
    job_summary = {
        "title": job.get("title"),
        "education_requirement": job.get("education_requirement"),
        "experience_requirement": job.get("experience_requirement"),
        "responsibilities": job.get("responsibilities"),
        "qualifications": job.get("qualifications"),
    }
    context = {
        "candidate": candidate_summary,
        "job": job_summary,
        "matched_required_skills": matched_required_skills,
        "matched_preferred_skills": matched_preferred_skills,
    }
    result = run_structured_task(
        role="Resume-to-Job Fit Assessor",
        goal="Judge education/experience/role fit strictly from the given candidate and job data.",
        backstory=_JOB_MATCH_ASSESSMENT_BACKSTORY,
        task_description=json.dumps(context, indent=2),
        expected_output="A grounded verdict on education, experience, and role fit, plus "
        "eligibility issues and a why-this-matches summary, using only the given data.",
        output_model=_JobMatchAssessmentResult,
    )
    return result.model_dump()
