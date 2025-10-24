from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.models.schemas import JobState
from src.services.job_store import JobStore
from src.services.vision import VisionClient, Detection
from src.utils.image_utils import overlay_logo
from src.utils.zip_utils import create_zip_from_directory


def _iter_image_files(root: Path) -> Iterable[Path]:
    """Yield image files under root that are likely drawings."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# PUBLIC_INTERFACE
def process_job_pipeline(job_store: JobStore, job_id: str) -> None:
    """Run the processing steps for a job and persist progress.

    Steps:
    1) Prepare work directory and locate uploads.
    2) Detect logo areas for each image (via VisionClient, stub/OpenAI).
    3) Overlay new logo onto images and write to work dir preserving structure.
    4) Pack processed files into result ZIP.
    5) Update meta with COMPLETED status and result_url.

    On any error, mark job ERROR with message.
    """
    try:
        job_store.update_status(job_id, status=JobState.RUNNING, progress=1, message="Initializing")
        uploads = job_store.get_uploads_dir(job_id)
        work = job_store.get_work_dir(job_id)
        result_dir = job_store.get_result_dir(job_id)
        work.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        drawings_dir = uploads / "drawings"
        logo_file = next((p for p in uploads.iterdir() if p.name.startswith("logo")), None)
        if not drawings_dir.exists() or logo_file is None:
            raise RuntimeError("Uploads missing: drawings or logo not found")

        images = list(_iter_image_files(drawings_dir))
        total = max(1, len(images))
        job_store.update_status(job_id, message=f"Found {len(images)} drawings to process")

        # Initialize vision client (will stub if no OpenAI key)
        vision = VisionClient.from_env()

        # Process each image
        for idx, img_path in enumerate(images, start=1):
            rel_path = img_path.relative_to(drawings_dir)
            out_path = work / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Detect logo region(s)
            detections: list[Detection] = []
            try:
                detections = vision.detect_logos(img_path)
            except Exception:
                # Non-fatal; proceed with default position
                detections = []

            # Determine overlay position; default top-left with padding
            if detections:
                # Use first detection; simple placement using top-left
                det = detections[0]
                position = (max(0, int(det.x)), max(0, int(det.y)))
            else:
                position = (10, 10)

            # Overlay logo with reasonable defaults
            overlay_logo(
                base_image_path=img_path,
                logo_path=logo_file,
                output_path=out_path,
                position=position,
                scale=0.2,
                opacity=0.95,
                max_logo_size_ratio=0.4,
            )

            progress = int(10 + (idx * 80) / total)
            job_store.update_status(job_id, progress=progress, message=f"Processed {idx}/{total} images")

        # Package into zip
        zip_path = result_dir / "processed.zip"
        create_zip_from_directory(work, zip_path)
        # Result URL is a hint for frontend; the download route will stream
        download_url = f"/jobs/{job_id}/download"
        job_store.set_result_url(job_id, result_url=download_url)
        job_store.update_status(job_id, status=JobState.COMPLETED, progress=100, message="Completed")
    except Exception as exc:  # pragma: no cover - safety
        job_store.save_error(job_id, f"Processing failed: {exc}")
