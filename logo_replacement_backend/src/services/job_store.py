import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models.schemas import JobCreated, JobState, JobStatus
from src.utils.zip_utils import safe_extract_zip


DEFAULT_BASE_DIR = Path("jobs")


@dataclass
class JobPaths:
    """Container of paths for a specific job on the filesystem."""
    root: Path
    meta: Path
    uploads: Path
    work: Path
    result: Path


def _job_paths(base_dir: Path, job_id: str) -> JobPaths:
    root = base_dir / job_id
    return JobPaths(
        root=root,
        meta=root / "meta.json",
        uploads=root / "uploads",
        work=root / "work",
        result=root / "result",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# PUBLIC_INTERFACE
class JobStore:
    """Filesystem-backed JobStore managing job lifecycle, persistence, and file operations.

    Directory structure:
        jobs/{job_id}/
            meta.json
            uploads/
            work/
            result/

    The meta.json file persists job state, progress, messages, error info, and timestamps.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize a JobStore.

        Args:
            base_dir: Base directory for storing jobs. Defaults to 'jobs' under CWD.
        """
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # PUBLIC_INTERFACE
    def create_job(self, initial_status: JobState = JobState.PENDING, message: Optional[str] = None) -> JobCreated:
        """Create a new job directory with meta.json and return JobCreated."""
        job_id = str(uuid.uuid4())
        paths = _job_paths(self.base_dir, job_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.uploads.mkdir(parents=True, exist_ok=True)
        paths.work.mkdir(parents=True, exist_ok=True)
        paths.result.mkdir(parents=True, exist_ok=True)

        meta = {
            "job_id": job_id,
            "status": initial_status.value,
            "progress": 0,
            "message": message or "Job created",
            "error": None,
            "result_url": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        with paths.meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return JobCreated(job_id=job_id, status=initial_status, message=message or "Job created")

    def _read_meta(self, job_id: str) -> dict:
        paths = _job_paths(self.base_dir, job_id)
        if not paths.meta.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        with paths.meta.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_meta(self, job_id: str, meta: dict) -> None:
        paths = _job_paths(self.base_dir, job_id)
        meta["updated_at"] = _now_iso()
        with paths.meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # PUBLIC_INTERFACE
    def update_status(
        self,
        job_id: str,
        *,
        status: Optional[JobState] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result_url: Optional[str] = None,
    ) -> JobStatus:
        """Update job status/progress/message/result_url and persist.

        Args:
            job_id: The job identifier.
            status: Optional new status.
            progress: Optional new progress 0-100.
            message: Optional new message.
            result_url: Optional URL for downloading results.

        Returns:
            JobStatus instance with the updated metadata.
        """
        meta = self._read_meta(job_id)
        if status is not None:
            meta["status"] = status.value
        if progress is not None:
            meta["progress"] = max(0, min(100, int(progress)))
        if message is not None:
            meta["message"] = message
        if result_url is not None:
            meta["result_url"] = result_url
        self._write_meta(job_id, meta)
        return self.get_status(job_id)

    # PUBLIC_INTERFACE
    def save_error(self, job_id: str, error_message: str) -> JobStatus:
        """Record an error message and set job state to ERROR."""
        meta = self._read_meta(job_id)
        meta["status"] = JobState.ERROR.value
        meta["error"] = error_message
        self._write_meta(job_id, meta)
        return self.get_status(job_id)

    # PUBLIC_INTERFACE
    def get_status(self, job_id: str) -> JobStatus:
        """Load and return current job status."""
        meta = self._read_meta(job_id)
        return JobStatus(
            job_id=meta["job_id"],
            status=JobState(meta["status"]),
            progress=int(meta.get("progress", 0)),
            message=meta.get("message"),
            error=meta.get("error"),
            result_url=meta.get("result_url"),
        )

    # PUBLIC_INTERFACE
    def save_uploads(self, job_id: str, drawings_zip_path: Path, logo_image_path: Path) -> None:
        """Save uploads into job uploads dir and extract drawings zip safely.

        The uploads directory will contain:
            - drawings.zip (original)
            - drawings/ (extracted)
            - logo.ext (original logo file)
        """
        paths = _job_paths(self.base_dir, job_id)
        if not paths.uploads.exists():
            raise FileNotFoundError(f"Job uploads directory missing for job: {job_id}")
        paths.uploads.mkdir(parents=True, exist_ok=True)

        # Copy original files
        drawings_zip_dest = paths.uploads / "drawings.zip"
        logo_dest = paths.uploads / f"logo{Path(logo_image_path).suffix or '.png'}"

        shutil.copy2(drawings_zip_path, drawings_zip_dest)
        shutil.copy2(logo_image_path, logo_dest)

        # Extract safely
        extracted_dir = paths.uploads / "drawings"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(drawings_zip_dest, extracted_dir)

        # Update meta
        meta = self._read_meta(job_id)
        meta["status"] = JobState.READY.value
        meta["message"] = "Uploads saved and drawings extracted"
        self._write_meta(job_id, meta)

    # PUBLIC_INTERFACE
    def get_uploads_dir(self, job_id: str) -> Path:
        """Return path to job uploads directory."""
        return _job_paths(self.base_dir, job_id).uploads

    # PUBLIC_INTERFACE
    def get_work_dir(self, job_id: str) -> Path:
        """Return path to job work directory."""
        return _job_paths(self.base_dir, job_id).work

    # PUBLIC_INTERFACE
    def get_result_dir(self, job_id: str) -> Path:
        """Return path to job result directory."""
        return _job_paths(self.base_dir, job_id).result

    # PUBLIC_INTERFACE
    def set_result_url(self, job_id: str, result_url: str) -> JobStatus:
        """Persist result URL to meta and return updated status."""
        return self.update_status(job_id, result_url=result_url)

    # PUBLIC_INTERFACE
    def cleanup(self, job_id: str) -> None:
        """Remove the job directory and all of its contents."""
        paths = _job_paths(self.base_dir, job_id)
        if paths.root.exists():
            shutil.rmtree(paths.root, ignore_errors=True)
