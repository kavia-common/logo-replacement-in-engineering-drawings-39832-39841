import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from src.models.schemas import JobCreated, JobState, JobStatus, PerFileDetectionSummary, DetectionBox
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
        detections: Optional[List[PerFileDetectionSummary]] = None,
    ) -> JobStatus:
        """Update job status/progress/message/result_url and persist.

        Args:
            job_id: The job identifier.
            status: Optional new status.
            progress: Optional new progress 0-100.
            message: Optional new message.
            result_url: Optional URL for downloading results.
            detections: Optional complete list of per-file detection summaries.

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
        if detections is not None:
            # Serialize into simple dicts
            meta["detections"] = [
                {
                    "file": d.file,
                    "found": d.found,
                    "method": d.method,
                    "reason": d.reason,
                    "boxes": [
                        {
                            "x": b.x,
                            "y": b.y,
                            "width": b.width,
                            "height": b.height,
                            "confidence": b.confidence,
                            "method": b.method,
                            "page": b.page,
                            "dtype": b.dtype,
                        }
                        for b in d.boxes
                    ],
                    "replaced_count": getattr(d, "replaced_count", 0),
                    "replaced_logo_count": getattr(d, "replaced_logo_count", 0),
                    "replaced_text_count": getattr(d, "replaced_text_count", 0),
                    "placements": getattr(d, "placements", None),
                }
                for d in detections
            ]
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
        detections_serialized = meta.get("detections")
        detections = None
        if isinstance(detections_serialized, list):
            detections = []
            for d in detections_serialized:
                boxes = []
                for b in d.get("boxes", []):
                    boxes.append(
                        DetectionBox(
                            x=float(b.get("x", 0)),
                            y=float(b.get("y", 0)),
                            width=float(b.get("width", 0)),
                            height=float(b.get("height", 0)),
                            confidence=float(b.get("confidence", 0)),
                            method=b.get("method"),
                            page=b.get("page"),
                            dtype=b.get("dtype"),
                        )
                    )
                detections.append(
                    PerFileDetectionSummary(
                        file=d.get("file", ""),
                        found=bool(d.get("found", False)),
                        method=d.get("method"),
                        boxes=boxes,
                        reason=d.get("reason"),
                        replaced_count=int(d.get("replaced_count", 0)),
                        replaced_logo_count=int(d.get("replaced_logo_count", 0)),
                        replaced_text_count=int(d.get("replaced_text_count", 0)),
                        placements=d.get("placements"),
                    )
                )

        return JobStatus(
            job_id=meta["job_id"],
            status=JobState(meta["status"]),
            progress=int(meta.get("progress", 0)),
            message=meta.get("message"),
            error=meta.get("error"),
            result_url=meta.get("result_url"),
            detections=detections,
        )

    # PUBLIC_INTERFACE
    def save_uploads(self, job_id: str, drawings_zip_path: Path, logo_image_path: Path) -> None:
        """Backward-compatible helper to save only ZIP + logo (legacy path)."""
        self.save_uploads_flexible(job_id, logo_image_path, drawings_zip_path, None)

    # PUBLIC_INTERFACE
    def save_uploads_flexible(
        self,
        job_id: str,
        logo_image_path: Path,
        drawings_zip_path: Optional[Path] = None,
        individual_files_dir: Optional[Path] = None,
    ) -> None:
        """Save uploads into job uploads dir supporting ZIP and/or individual files.

        The uploads directory will contain:
            - logo.ext
            - drawings.zip (if zip provided)
            - drawings/ (extracted from zip if provided)
            - files/ (individual uploaded files, if provided)
        """
        paths = _job_paths(self.base_dir, job_id)
        if not paths.uploads.exists():
            raise FileNotFoundError(f"Job uploads directory missing for job: {job_id}")
        paths.uploads.mkdir(parents=True, exist_ok=True)

        # Save logo
        logo_dest = paths.uploads / f"logo{Path(logo_image_path).suffix or '.png'}"
        shutil.copy2(logo_image_path, logo_dest)

        # If ZIP provided: copy and extract
        extracted_dir = paths.uploads / "drawings"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        if drawings_zip_path is not None:
            drawings_zip_dest = paths.uploads / "drawings.zip"
            shutil.copy2(drawings_zip_path, drawings_zip_dest)
            safe_extract_zip(drawings_zip_dest, extracted_dir)

        # If individual files provided: copy into uploads/files preserving names
        if individual_files_dir is not None and individual_files_dir.exists():
            files_dest = paths.uploads / "files"
            files_dest.mkdir(parents=True, exist_ok=True)
            for p in individual_files_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(individual_files_dir)
                    dest = files_dest / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest)

        # Update meta
        meta = self._read_meta(job_id)
        meta["status"] = JobState.READY.value
        # Build informative message
        parts = []
        if drawings_zip_path is not None:
            parts.append("ZIP extracted")
        if individual_files_dir is not None and any((paths.uploads / "files").rglob("*")):
            parts.append("individual files saved")
        info = " and ".join(parts) if parts else "uploads saved"
        meta["message"] = f"Uploads saved: {info}".strip()
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
