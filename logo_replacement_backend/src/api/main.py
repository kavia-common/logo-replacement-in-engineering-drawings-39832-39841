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
    BackgroundTasks,
)
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.models.schemas import JobCreated, JobState, JobStatus, ProcessedFile, ProcessedFileList
from src.services.job_store import JobStore
from src.services.processing import process_job_pipeline

# App metadata and tags for OpenAPI
app = FastAPI(
    title="Logo Replacement Backend",
    description=(
        "REST API for processing engineering drawing images by detecting and replacing logos. "
        "Create a job, upload a ZIP of drawings and a logo image, start processing, "
        "poll status, and download the processed ZIP.\n\n"
        "The drawings ZIP may contain raster images (PNG/JPG/TIFF/BMP/GIF) and PDFs. "
        "PDFs are rasterized per page internally before logo detection.\n\n"
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
    expose_headers=["Content-Disposition"],
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
    summary="Upload drawings (ZIP and/or files) and logo image",
    description=(
        "Upload one or both of the following:\n"
        "- drawings_zip: a .zip containing drawing images and/or PDFs\n"
        "- drawings_files[]: individual files (PNG, JPG/JPEG, TIFF, BMP, GIF, PDF)\n"
        "and the new logo image as 'logo_image' (required).\n\n"
        "Files are stored and drawings ZIP extracted. Individual files are saved as-is. "
        "Status moves to READY on success.\n\n"
        "PDFs will be rasterized into images before processing."
    ),
    tags=["jobs"],
    responses={
        200: {"description": "Uploads saved"},
        400: {"model": ErrorResponse, "description": "Bad input"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def upload_files(
    job_id: str,
    logo_image: UploadFile = File(..., description="Logo image to overlay"),
    drawings_zip: Optional[UploadFile] = File(None, description="Optional ZIP containing drawings (images/PDFs)"),
    drawings_files: Optional[List[UploadFile]] = File(
        None, description="Optional individual files (PNG, JPG/JPEG, TIFF, BMP, GIF, PDF). Can be multiple."
    ),
    old_logo_image: Optional[UploadFile] = File(None, description="Optional reference image of the old logo for template matching fallback"),
) -> JobStatus:
    """
    Receive logo (required) and drawings via ZIP and/or individual files.
    Save under uploads/:
      - logo.ext
      - drawings/ (extracted from ZIP if provided)
      - files/ (individual files preserved with original names)
    """
    _ensure_job_exists(job_id)

    # Validate logo is present
    if not (logo_image and logo_image.filename):
        raise HTTPException(status_code=400, detail="logo_image file is required")

    # Validate at least one drawings source
    has_zip = bool(drawings_zip and drawings_zip.filename)
    has_files = bool(drawings_files and len(drawings_files) > 0)
    if not (has_zip or has_files):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one drawings source: drawings_zip or drawings_files[]",
        )

    # Validate zip extension if provided
    if has_zip and not drawings_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="drawings_zip must be a .zip file")

    # Validate individual files extensions
    allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".pdf"}
    if has_files:
        bad = [f.filename for f in drawings_files if not (f.filename and Path(f.filename).suffix.lower() in allowed_exts)]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file types in drawings_files: {', '.join(bad)}. "
                       f"Allowed: {', '.join(sorted(allowed_exts))}",
            )

    try:
        uploads_dir = JOB_STORE.get_uploads_dir(job_id)
        tmp_dir = uploads_dir / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Save logo
        logo_suffix = Path(logo_image.filename).suffix or ".png"
        logo_tmp = tmp_dir / f"logo{logo_suffix}"
        with logo_tmp.open("wb") as f:
            f.write(await logo_image.read())

        # Prepare paths
        drawings_zip_tmp: Optional[Path] = None
        if has_zip:
            drawings_zip_tmp = tmp_dir / "drawings.zip"
            with drawings_zip_tmp.open("wb") as f:
                f.write(await drawings_zip.read())

        files_tmp_dir: Optional[Path] = None
        if has_files:
            files_tmp_dir = tmp_dir / "files"
            files_tmp_dir.mkdir(parents=True, exist_ok=True)
            # Save all individual files preserving file names
            for uf in drawings_files or []:
                if not uf.filename:
                    continue
                out_path = files_tmp_dir / Path(uf.filename).name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as f:
                    f.write(await uf.read())

        # If optional old_logo_image provided, save it
        if old_logo_image and old_logo_image.filename:
            old_suffix = Path(old_logo_image.filename).suffix or ".png"
            old_tmp = tmp_dir / f"old_logo{old_suffix}"
            with old_tmp.open("wb") as f:
                f.write(await old_logo_image.read())

        # Persist into job store
        JOB_STORE.update_status(job_id, status=JobState.UPLOADING, message="Saving uploads")
        JOB_STORE.save_uploads_flexible(job_id, logo_tmp, drawings_zip_tmp, files_tmp_dir)
        # If we saved old_logo in _incoming, move it along to uploads root
        for p in tmp_dir.glob("old_logo.*"):
            dest = JOB_STORE.get_uploads_dir(job_id) / p.name
            dest.write_bytes(p.read_bytes())
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
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Get job status (alias)",
    description="Alias for /jobs/{job_id}/status to align with simplified architecture.",
    tags=["jobs"],
)
def get_status_alias(job_id: str) -> JobStatus:
    """Alias endpoint returning job status."""
    return get_status(job_id)


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
@app.get(
    "/download/{job_id}",
    summary="Download processed ZIP (alias)",
    description="Alias for /jobs/{job_id}/download for the simplified architecture.",
    tags=["jobs"],
)
def download_result_alias(job_id: str):
    """Alias endpoint streaming the result ZIP."""
    return download_result(job_id)


def _guess_mime_type(filename: str) -> str:
    """Basic mime guessing for common image/pdf types."""
    ext = Path(filename).suffix.lower()
    if ext in {".png"}:
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext in {".tif", ".tiff"}:
        return "image/tiff"
    if ext in {".bmp"}:
        return "image/bmp"
    if ext in {".gif"}:
        return "image/gif"
    if ext == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


# PUBLIC_INTERFACE
@app.get(
    "/jobs/{job_id}/files",
    response_model=ProcessedFileList,
    summary="List processed files",
    description="List per-file processed outputs preserved under result/out/.",
    tags=["jobs"],
    responses={
        200: {"description": "List of processed files"},
        404: {"model": ErrorResponse, "description": "Job not found or results unavailable"},
        409: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
def list_processed_files(job_id: str) -> ProcessedFileList:
    """Return the relative paths of individually processed outputs."""
    status_obj = _ensure_job_exists(job_id)
    if status_obj.status != JobState.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    # Files are preserved under result/out/
    out_dir = JOB_STORE.get_result_dir(job_id) / "out"
    if not out_dir.exists():
        # Backward compatibility: if no out dir, try listing from work dir
        work_dir = JOB_STORE.get_work_dir(job_id)
        if not work_dir.exists():
            raise HTTPException(status_code=404, detail="No processed files available")
        base = work_dir
    else:
        base = out_dir

    files: list[ProcessedFile] = []
    for p in base.rglob("*"):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            files.append(ProcessedFile(filename=rel, size=p.stat().st_size, content_type=_guess_mime_type(rel)))
    return ProcessedFileList(items=files)


# PUBLIC_INTERFACE
@app.get(
    "/jobs/{job_id}/files/{file_path:path}",
    summary="Download an individual processed file",
    description="Stream a single processed file with proper Content-Type and Content-Disposition.",
    tags=["jobs"],
    responses={
        200: {"description": "File stream"},
        404: {"model": ErrorResponse, "description": "Job or file not found"},
        409: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
def download_processed_file(job_id: str, file_path: str):
    """Stream a single processed output file by relative path."""
    status_obj = _ensure_job_exists(job_id)
    if status_obj.status != JobState.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    base_out = JOB_STORE.get_result_dir(job_id) / "out"
    base_alt = JOB_STORE.get_work_dir(job_id)

    # Choose base that exists
    base = base_out if base_out.exists() else base_alt
    if not base.exists():
        raise HTTPException(status_code=404, detail="Processed files not found")

    # Normalize and prevent path traversal
    requested = (base / file_path).resolve()
    try:
        requested.relative_to(base.resolve())
    except Exception:
        raise HTTPException(status_code=404, detail="Invalid path")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    mime = _guess_mime_type(requested.name)
    return FileResponse(
        path=str(requested),
        media_type=mime,
        filename=Path(requested.name).name,
        headers={"Content-Disposition": f'attachment; filename="{Path(requested.name).name}"'},
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


# PUBLIC_INTERFACE
@app.post(
    "/process",
    summary="Submit drawings and logo in a single request",
    description=(
        "Accept a single multipart/form-data with 'logo_image' and either 'drawings_zip' "
        "or 'drawings_files[]'. Responds immediately with job_id while processing runs "
        "in the background. Poll GET /status/{job_id} and download via GET /download/{job_id}."
    ),
    tags=["jobs"],
    responses={
        200: {"description": "Job accepted and started"},
        400: {"model": ErrorResponse, "description": "Bad input"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def process_single_step(
    background_tasks: BackgroundTasks,
    logo_image: UploadFile = File(..., description="Logo image to overlay"),
    drawings_zip: Optional[UploadFile] = File(None, description="Optional ZIP containing drawings (images/PDFs)"),
    drawings_files: Optional[List[UploadFile]] = File(
        None, description="Optional individual files (PNG, JPG/JPEG, TIFF, BMP, GIF, PDF). Can be multiple."
    ),
    old_logo_image: Optional[UploadFile] = File(None, description="Optional reference image of the old logo for template matching fallback"),
) -> dict:
    """Create a job, save uploads, queue processing, and return job_id immediately."""
    try:
        created = JOB_STORE.create_job(initial_status=JobState.PENDING, message="Job created")
        job_id = created.job_id

        # Reuse validation from upload_files
        if not (logo_image and logo_image.filename):
            raise HTTPException(status_code=400, detail="logo_image file is required")
        has_zip = bool(drawings_zip and drawings_zip.filename)
        has_files = bool(drawings_files and len(drawings_files or []) > 0)
        if not (has_zip or has_files):
            raise HTTPException(status_code=400, detail="Provide drawings_zip or drawings_files[]")

        if has_zip and not drawings_zip.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="drawings_zip must be a .zip file")

        allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".pdf"}
        if has_files:
            bad = [f.filename for f in drawings_files or [] if not (f.filename and Path(f.filename).suffix.lower() in allowed_exts)]
            if bad:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file types in drawings_files: {', '.join(bad)}. Allowed: {', '.join(sorted(allowed_exts))}",
                )

        # Temporarily store inputs under uploads/_incoming then persist via JobStore
        uploads_dir = JOB_STORE.get_uploads_dir(job_id)
        tmp_dir = uploads_dir / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        logo_suffix = Path(logo_image.filename).suffix or ".png"
        logo_tmp = tmp_dir / f"logo{logo_suffix}"
        with logo_tmp.open("wb") as f:
            f.write(await logo_image.read())

        drawings_zip_tmp: Optional[Path] = None
        if has_zip:
            drawings_zip_tmp = tmp_dir / "drawings.zip"
            with drawings_zip_tmp.open("wb") as f:
                f.write(await drawings_zip.read())

        files_tmp_dir: Optional[Path] = None
        if has_files:
            files_tmp_dir = tmp_dir / "files"
            files_tmp_dir.mkdir(parents=True, exist_ok=True)
            for uf in drawings_files or []:
                if not uf.filename:
                    continue
                out_path = files_tmp_dir / Path(uf.filename).name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as f:
                    f.write(await uf.read())

        # Optional old logo
        if old_logo_image and old_logo_image.filename:
            old_suffix = Path(old_logo_image.filename).suffix or ".png"
            with (tmp_dir / f"old_logo{old_suffix}").open("wb") as f:
                f.write(await old_logo_image.read())

        JOB_STORE.update_status(job_id, status=JobState.UPLOADING, message="Saving uploads")
        JOB_STORE.save_uploads_flexible(job_id, logo_tmp, drawings_zip_tmp, files_tmp_dir)
        # Move old logo to uploads root if exists
        for p in tmp_dir.glob("old_logo.*"):
            (JOB_STORE.get_uploads_dir(job_id) / p.name).write_bytes(p.read_bytes())

        # Queue background processing
        JOB_STORE.update_status(job_id, status=JobState.RUNNING, progress=0, message="Queued for processing")
        EXECUTOR.submit(process_job_pipeline, JOB_STORE, job_id)

        # Return only job_id to match requirement
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {exc}") from exc
