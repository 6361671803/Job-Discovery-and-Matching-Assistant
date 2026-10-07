import json
import logging

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.agents.company_discovery_agent import CompanyDiscoveryAgent
from app.agents.job_discovery_agent import JobDiscoveryAgent
from app.agents.matching_agent import MatchingAgent
from app.agents.preference_agent import PreferenceAgent
from app.agents.resume_analyzer import ResumeAnalyzerAgent
from app.config import settings
from app.db.database import get_db, init_db
from app.db.models import Candidate, Company, Job
from app.models.schemas import (
    CandidateResponse,
    CompanyDiscoveryResponse,
    JobDiscoveryResponse,
    PreferenceInput,
    PreferenceResponse,
)
from app.services.job_filter import ALLOWED_WINDOW_DAYS, DEFAULT_WINDOW_DAYS, filter_jobs_by_window
from app.services.llm_client import LLMNotConfiguredError, LLMRequestError
from app.services.resume_parser import SUPPORTED_EXTENSIONS, UnsupportedResumeFormat
from app.services.search_client import SearchNotConfiguredError, SearchRequestError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

app = FastAPI(title="Job Discovery & Matching Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/resume/upload", response_model=CandidateResponse)
async def upload_resume(file: UploadFile, db: Session = Depends(get_db)) -> CandidateResponse:
    suffix = "." + file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Please upload a PDF or DOCX resume.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    agent = ResumeAnalyzerAgent()
    try:
        # analyze() is a blocking, synchronous call (LLM request + DB write) — running it
        # directly on this async route's event loop would freeze the *entire* server for every
        # client, not just this request (confirmed live: a stuck resume upload made even a
        # plain GET /docs health check stop responding). Every other route in this file is a
        # plain `def`, which FastAPI already offloads to its thread pool automatically; this is
        # the one route that's `async def` (needed for `await file.read()`), so the blocking
        # part is explicitly offloaded here instead.
        candidate = await run_in_threadpool(agent.analyze, db, file.filename, content)
    except UnsupportedResumeFormat as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return CandidateResponse(
        id=candidate.id,
        resume_filename=candidate.resume_filename,
        state=candidate.state,
        profile=json.loads(candidate.profile_json),
    )


@app.get("/api/candidate/{candidate_id}", response_model=CandidateResponse)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)) -> CandidateResponse:
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return CandidateResponse(
        id=candidate.id,
        resume_filename=candidate.resume_filename,
        state=candidate.state,
        profile=json.loads(candidate.profile_json),
    )


@app.post("/api/candidate/{candidate_id}/preferences", response_model=PreferenceResponse)
def submit_preferences(
    candidate_id: str, payload: PreferenceInput, db: Session = Depends(get_db)
) -> PreferenceResponse:
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    agent = PreferenceAgent()
    try:
        candidate = agent.collect(db, candidate, payload)
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return PreferenceResponse(
        candidate_id=candidate.id,
        state=candidate.state,
        preferences=json.loads(candidate.preferences_json),
    )


@app.get("/api/candidate/{candidate_id}/preferences", response_model=PreferenceResponse)
def get_preferences(candidate_id: str, db: Session = Depends(get_db)) -> PreferenceResponse:
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    if candidate.preferences_json is None:
        raise HTTPException(status_code=404, detail="Preferences not yet collected for this candidate.")
    return PreferenceResponse(
        candidate_id=candidate.id,
        state=candidate.state,
        preferences=json.loads(candidate.preferences_json),
    )


def _company_discovery_response(candidate: Candidate, companies: list[Company]) -> CompanyDiscoveryResponse:
    return CompanyDiscoveryResponse(
        candidate_id=candidate.id,
        state=candidate.state,
        companies=[
            {
                "id": c.id,
                "name": c.name,
                "website": c.website,
                "careers_url": c.careers_url,
                "discovery_role": c.discovery_role,
                "discovery_location": c.discovery_location,
            }
            for c in companies
        ],
    )


@app.post("/api/candidate/{candidate_id}/discover-companies", response_model=CompanyDiscoveryResponse)
def discover_companies(candidate_id: str, db: Session = Depends(get_db)) -> CompanyDiscoveryResponse:
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    if candidate.preferences_json is None:
        raise HTTPException(status_code=400, detail="Preferences must be collected before discovering companies.")

    agent = CompanyDiscoveryAgent()
    try:
        companies = agent.discover(db, candidate)
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except SearchNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SearchRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return _company_discovery_response(candidate, companies)


@app.get("/api/candidate/{candidate_id}/companies", response_model=CompanyDiscoveryResponse)
def get_companies(candidate_id: str, db: Session = Depends(get_db)) -> CompanyDiscoveryResponse:
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    companies = db.query(Company).filter(Company.candidate_id == candidate_id).all()
    return _company_discovery_response(candidate, companies)


def _serialize_job(j: Job) -> dict:
    # A job only has a well-formed match once MatchingAgent has written every field together;
    # treat it as unmatched rather than erroring if any core piece is unexpectedly missing.
    breakdown_fields = (
        j.skills_score, j.semantic_score, j.education_score, j.experience_score,
        j.role_score, j.location_score,
    )
    match = None
    if j.overall_score is not None and all(f is not None for f in breakdown_fields):
        match = {
            "overall_score": j.overall_score,
            "breakdown": {
                "skills_score": j.skills_score,
                "semantic_score": j.semantic_score,
                "education_score": j.education_score,
                "experience_score": j.experience_score,
                "role_score": j.role_score,
                "location_score": j.location_score,
            },
            "matched_required_skills": json.loads(j.matched_required_skills_json or "[]"),
            "missing_required_skills": json.loads(j.missing_required_skills_json or "[]"),
            "matched_preferred_skills": json.loads(j.matched_preferred_skills_json or "[]"),
            "missing_preferred_skills": json.loads(j.missing_preferred_skills_json or "[]"),
            "relevant_projects": json.loads(j.relevant_projects_json or "[]"),
            "eligibility_issues": json.loads(j.eligibility_issues_json or "[]"),
            "location_match_reasoning": j.location_match_reasoning or "",
            "education_reasoning": j.education_reasoning or "",
            "experience_reasoning": j.experience_reasoning or "",
            "role_reasoning": j.role_reasoning or "",
            "why_this_matches": j.why_this_matches or "",
        }
    return {
        "id": j.id,
        "company_id": j.company_id,
        "company_name": j.company_name,
        "title": j.title,
        "job_id_external": j.job_id_external,
        "location": j.location,
        "work_mode": j.work_mode,
        "date_posted": j.date_posted,
        "posted_date_parsed": j.posted_date_parsed.isoformat() if j.posted_date_parsed else None,
        "closing_date": j.closing_date,
        "employment_type": j.employment_type,
        "experience_requirement": j.experience_requirement,
        "education_requirement": j.education_requirement,
        "required_skills": json.loads(j.required_skills_json),
        "preferred_skills": json.loads(j.preferred_skills_json),
        "responsibilities": j.responsibilities,
        "qualifications": j.qualifications,
        "job_url": j.job_url,
        "application_url": j.application_url,
        "status": j.status,
        "match": match,
    }


def _job_discovery_response(
    candidate: Candidate, jobs: list[Job], window_days: int, companies_skipped: list[str]
) -> JobDiscoveryResponse:
    # Rank by match score when available so the 30-day filter's "recent" bucket comes out ranked.
    jobs = sorted(jobs, key=lambda j: (j.overall_score is None, -(j.overall_score or 0)))
    grouped = filter_jobs_by_window(jobs, window_days)
    return JobDiscoveryResponse(
        candidate_id=candidate.id,
        state=candidate.state,
        window_days=window_days,
        companies_skipped=companies_skipped,
        recent_jobs=[_serialize_job(j) for j in grouped["recent"]],
        unknown_date_jobs=[_serialize_job(j) for j in grouped["unknown_date"]],
        older_jobs_count=len(grouped["older"]),
    )


def _validate_window_days(window_days: int) -> int:
    if window_days not in ALLOWED_WINDOW_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"window_days must be one of {ALLOWED_WINDOW_DAYS}, got {window_days}.",
        )
    return window_days


@app.post("/api/candidate/{candidate_id}/discover-jobs", response_model=JobDiscoveryResponse)
def discover_jobs(
    candidate_id: str,
    window_days: int = Query(DEFAULT_WINDOW_DAYS),
    db: Session = Depends(get_db),
) -> JobDiscoveryResponse:
    _validate_window_days(window_days)
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    agent = JobDiscoveryAgent()
    try:
        jobs, skipped = agent.discover(db, candidate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _job_discovery_response(candidate, jobs, window_days, skipped)


@app.get("/api/candidate/{candidate_id}/jobs", response_model=JobDiscoveryResponse)
def get_jobs(
    candidate_id: str,
    window_days: int = Query(DEFAULT_WINDOW_DAYS),
    db: Session = Depends(get_db),
) -> JobDiscoveryResponse:
    _validate_window_days(window_days)
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    jobs = db.query(Job).filter(Job.candidate_id == candidate_id).all()
    return _job_discovery_response(candidate, jobs, window_days, [])


@app.post("/api/candidate/{candidate_id}/match-jobs", response_model=JobDiscoveryResponse)
def match_jobs(
    candidate_id: str,
    window_days: int = Query(DEFAULT_WINDOW_DAYS),
    db: Session = Depends(get_db),
) -> JobDiscoveryResponse:
    _validate_window_days(window_days)
    candidate = db.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    agent = MatchingAgent()
    try:
        jobs = agent.match(db, candidate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _job_discovery_response(candidate, jobs, window_days, [])
