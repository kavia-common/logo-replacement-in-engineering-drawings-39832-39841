from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.models.schemas import JobCreated, JobState, JobStatus
from src.services.job_store import JobStore
from src.services.processing import process_job_pipeline

# App metadata and tags for OpenAPI
app = FastAPI(
    title="Logo Replacement Backend",
    description=(
        "REST API for processing engineering drawing images by detecting and replacing logos. "
        "Create a job, upload a ZIP of drawings and a logo image, start processing, "
        "poll status, and download the processed ZIP.\n\n"
        "WebSocket is not used in this demo; poll GET /jobs/{job_id}/status for updates."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Service health and info"},
        {"name": "jobs", "description": "Job lifecycle operations"},
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo, allow all. In production, restrict origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global thread pool to avoid blocking request handlers
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "4")))

# Base directory for job storage; configurable via env
JOBS_DIR = Path(os.getenv("JOBS_DIR", "jobs")).resolve()
JOB_STORE = JobStore(base_dir=JOBS_DIR)


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")


def _ensure_job_exists(job_id: str) -> JobStatus:
    try:
        return JOB_STORE.get_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Simple health check endpoint returning service status."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.post(
    "/jobs",
    response_model=JobCreated,
    summary="Create a new processing job",
    description="Create a new job and return its identifier. Next, upload files to /jobs/{job_id}/upload.",
    tags=["jobs"],
    responses={
        201: {"description": "Job created"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    status_code=status.HTTP_201_CREATED,
)
def create_job() -> JobCreated:
    """Create a new job with initial PENDING status."""
    try:
        created = JOB_STORE.create_job(initial_status=JobState.PENDING, message="Job created")
        return created
    except Exception as exc:  # pragma: no cover - safety
        raise HTTPException(status_code=500, detail=f"Failed to create job: {exc}") from exc


# PUBLIC_INTERFACE
@app.post(
    "/jobs/{job_id}/upload",
    response_model=JobStatus,
    summary="Upload drawings ZIP and logo image",
    description=(
        "Upload the ZIP of drawings as 'drawings' form field and the new logo image as 'logo'. "
        "Files are stored and drawings ZIP extracted. Status moves to READY on success."
    ),
    tags=["jobs"],
    responses={
        200: {"description": "Uploads saved and extracted"},
        400: {"model": ErrorResponse, "description": "Bad input"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def upload_files(
    job_id: str,
    drawings: UploadFile = File(..., description="ZIP file containing drawing images"),
    logo: UploadFile = File(..., description="Logo image to overlay"),
) -> JobStatus:
    """Receive drawings ZIP and logo image, save into job, and extract ZIP safely."""
    _ensure_job_exists(job_id)
    # Validate content types minimally
    if not (drawings.filename and drawings.filename.lower().endswith(".zip")):
        raise HTTPException(status_code=400, detail="drawings must be a .zip file")
    if not logo.filename:
        raise HTTPException(status_code=400, detail="logo file is required")

    try:
        # Save incoming files to temp paths under the job
        uploads_dir = JOB_STORE.get_uploads_dir(job_id)
        tmp_dir = uploads_dir / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        drawings_path = tmp_dir / "drawings.zip"
        logo_path = tmp_dir / f"logo{Path(logo.filename).suffix or '.png'}"

        with drawings_path.open("wb") as f:
            f.write(await drawings.read())
        with logo_path.open("wb") as f:
            f.write(await logo.read())

        # Persist into job store (copies to canonical locations and extracts)
        JOB_STORE.update_status(job_id, status=JobState.UPLOADING, message="Saving uploads")
        JOB_STORE.save_uploads(job_id, drawings_path, logo_path)
        return JOB_STORE.get_status(job_id)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - safety
        JOB_STORE.save_error(job_id, f"Upload failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


# PUBLIC_INTERFACE
@app.post(
    "/jobs/{job_id}/start",
    response_model=JobStatus,
    summary="Start background processing",
    description=(
        "Start processing of the uploaded drawings in the background without blocking the request. "
        "Poll GET /jobs/{job_id}/status to monitor progress."
    ),
    tags=["jobs"],
    responses={
        200: {"description": "Processing started or already running"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job not ready to start"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
def start_processing(job_id: str) -> JobStatus:
    """Queue the processing job in a thread pool. Safe to call multiple times."""
    status_obj = _ensure_job_exists(job_id)
    if status_obj.status not in {JobState.READY, JobState.PENDING, JobState.ERROR}:
        # If RUNNING/COMPLETED, return current status; if UPLOADING, can't start yet.
        if status_obj.status in {JobState.RUNNING, JobState.COMPLETED}:
            return status_obj
        if status_obj.status == JobState.UPLOADING:
            raise HTTPException(status_code=409, detail="Job is still UPLOADING; try later")

    # Set to RUNNING and submit task
    JOB_STORE.update_status(job_id, status=JobState.RUNNING, progress=0, message="Queued for processing")
    EXECUTOR.submit(process_job_pipeline, JOB_STORE, job_id)
    return JOB_STORE.get_status(job_id)


# PUBLIC_INTERFACE
@app.get(
    "/jobs/{job_id}/status",
    response_model=JobStatus,
    summary="Get job status",
    description="Return current status, progress, message, and result URL when available.",
    tags=["jobs"],
    responses={
        200: {"description": "Status returned"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
def get_status(job_id: str) -> JobStatus:
    """Return current job status information."""
    return _ensure_job_exists(job_id)


# PUBLIC_INTERFACE
@app.get(
    "/jobs/{job_id}/download",
    summary="Download processed ZIP",
    description="Stream the processed ZIP file with Content-Disposition header for download.",
    tags=["jobs"],
    responses={
        200: {"description": "ZIP stream returned"},
        404: {"model": ErrorResponse, "description": "Job not found or result not ready"},
        409: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
def download_result(job_id: str):
    """Stream the result ZIP to the client if the job is completed."""
    status_obj = _ensure_job_exists(job_id)
    if status_obj.status != JobState.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    result_dir = JOB_STORE.get_result_dir(job_id)
    zip_path = result_dir / "processed.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Result ZIP not found")

    # Stream with Content-Disposition
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{job_id}-processed.zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}-processed.zip"'},
    )


# PUBLIC_INTERFACE
@app.delete(
    "/jobs/{job_id}",
    summary="Delete a job and its files",
    description="Remove job directory and all contents.",
    tags=["jobs"],
    responses={
        204: {"description": "Deleted"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_job(job_id: str):
    """Delete the job directory and artifacts."""
    # Ensure exists first
    _ensure_job_exists(job_id)
    JOB_STORE.cleanup(job_id)
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
