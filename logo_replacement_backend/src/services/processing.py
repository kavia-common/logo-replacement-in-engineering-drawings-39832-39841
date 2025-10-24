from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image

from src.config import CONFIG
from src.models.schemas import JobState, PerFileDetectionSummary, DetectionBox
from src.services.job_store import JobStore
from src.services.vision import VisionClient, Detection
from src.utils.image_utils import overlay_logo
from src.utils.zip_utils import create_zip_from_directory
from src.utils.pdf_utils import rasterize_pdf_to_images, PdfRasterizerUnavailable, PdfRasterizationConfig


def _iter_image_files(root: Path) -> Iterable[Path]:
    """Yield image files under root that are likely drawings."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _iter_pdf_files(root: Path) -> Iterable[Path]:
    """Yield PDF files under root."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf":
            yield p


def _prepare_rasterized_inputs(drawings_dir: Path, work: Path) -> List[Path]:
    """Create a unified list of images to process.
    - Include original raster images under drawings_dir.
    - For any PDFs, rasterize each page into work/_pdf_raster/<original_subpath>/.
    Returns list of image paths to feed into detection/overlay.
    """
    images: List[Path] = []

    # 1) Include existing raster images
    images.extend(list(_iter_image_files(drawings_dir)))

    # 2) Rasterize PDFs, preserving directory structure under work/_pdf_raster
    pdfs = list(_iter_pdf_files(drawings_dir))
    if pdfs:
        raster_root = work / "_pdf_raster"
        cfg = PdfRasterizationConfig()
        for pdf in pdfs:
            rel = pdf.relative_to(drawings_dir)
            out_dir = raster_root / rel.parent
            try:
                pages = rasterize_pdf_to_images(pdf, out_dir, cfg)
                images.extend(pages)
            except PdfRasterizerUnavailable as e:
                # If no rasterizer, we fail early with a clear message
                raise RuntimeError(
                    "PDF support is enabled but no PDF rasterizer is available. "
                    "Install PyMuPDF (fitz) or pdf2image with Poppler. "
                    f"File: {pdf} Error: {e}"
                ) from e
            except Exception as e:
                # Other rasterization errors
                raise RuntimeError(f"Failed to rasterize PDF '{pdf}': {e}") from e

    return images


# PUBLIC_INTERFACE
def process_job_pipeline(job_store: JobStore, job_id: str) -> None:
    """Run the processing steps for a job and persist progress.

    Steps:
    1) Prepare work directory and locate uploads.
    2) Detect logo areas per image using OpenAI Vision or template matching fallback.
    3) Overlay new logo precisely with alpha and target box fitting.
    4) Pack processed files into result ZIP and store per-file detection summaries.
    5) Update meta with COMPLETED status and result_url.

    Notes:
    - PDFs are rasterized first.
    - When no region is found, we log a per-file reason.
    """
    try:
        job_store.update_status(job_id, status=JobState.RUNNING, progress=1, message="Initializing")
        uploads = job_store.get_uploads_dir(job_id)
        work = job_store.get_work_dir(job_id)
        result_dir = job_store.get_result_dir(job_id)
        work.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        drawings_dir = uploads / "drawings"
        files_dir = uploads / "files"
        logo_file = next((p for p in uploads.iterdir() if p.name.startswith("logo")), None)
        if logo_file is None:
            raise RuntimeError("Uploads missing: logo not found")

        # Optional old logo template provided by user
        old_logo_path: Optional[Path] = None
        for p in uploads.iterdir():
            if p.name.startswith("old_logo"):
                old_logo_path = p
                break

        # Build list of input images from both sources
        job_store.update_status(job_id, progress=5, message="Scanning inputs (images and PDFs)")

        images: list[Path] = []
        if drawings_dir.exists():
            images.extend(_prepare_rasterized_inputs(drawings_dir, work))
        if files_dir.exists():
            images.extend(_prepare_rasterized_inputs(files_dir, work))
        total = max(1, len(images))
        job_store.update_status(job_id, message=f"Found {len(images)} pages/images to process")

        # Initialize vision client
        vision = VisionClient.from_env()

        # Prepare detection summaries
        summaries: list[PerFileDetectionSummary] = []

        # Process each image
        for idx, img_path in enumerate(images, start=1):
            # Determine output path preserving structure as before
            if drawings_dir.exists() and img_path.is_relative_to(drawings_dir):
                rel_path = img_path.relative_to(drawings_dir)
                out_path = work / rel_path
            else:
                try:
                    rel_path = img_path.relative_to(work / "_pdf_raster")
                    out_path = work / rel_path
                except Exception:
                    out_path = work / img_path.name

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Create small thumbnail for speed (not saved; vision client internally resizes)
            try:
                with Image.open(img_path) as im:
                    # Touch to ensure readable
                    _ = im.size
            except Exception:
                summaries.append(
                    PerFileDetectionSummary(
                        file=str(img_path),
                        found=False,
                        method=None,
                        boxes=[],
                        reason="Failed to open image",
                    )
                )
                # Skip overlay; copy original as fallback
                out_path.write_bytes(Path(img_path).read_bytes())
                continue

            # Detect logo and text regions
            detections: list[Detection] = []
            used_method = None
            try:
                detections = vision.detect_regions(img_path, old_logo_image=old_logo_path)
                # Heuristic primary method label: prefer vision if any vision API set
                used_method = CONFIG.method if CONFIG.method in ("vision", "template") else (
                    "vision" if os.getenv("OPENAI_API_KEY") else "template"
                )
            except Exception:
                detections = []
                used_method = CONFIG.method

            # Sort by confidence, cap replacements per page
            detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
            max_replace = max(0, CONFIG.max_replacements_per_page)
            if max_replace > 0:
                detections_sorted = detections_sorted[:max_replace]

            # Perform overlays for each detection; for text regions, fit by height
            # We will write to a temporary path for chaining overlays onto the output of previous step
            current_input = img_path
            temp_out = out_path
            boxes_for_summary: list[DetectionBox] = []
            replaced_logo = 0
            replaced_text = 0

            if detections_sorted:
                for d in detections_sorted:
                    target_box = (int(d.x), int(d.y), int(d.width), int(d.height))
                    # For text, we maintain aspect but ensure the logo fits the height of the box by setting h as limit.
                    # overlay_logo already fits within box while preserving aspect; for text, we slightly reduce width.
                    overlay_logo(
                        base_image_path=current_input,
                        logo_path=logo_file,
                        output_path=temp_out,
                        target_box=target_box,
                        opacity=CONFIG.logo_opacity,
                    )
                    # Next iteration should overlay on last output
                    current_input = temp_out

                    boxes_for_summary.append(
                        DetectionBox(
                            x=float(d.x),
                            y=float(d.y),
                            width=float(d.width),
                            height=float(d.height),
                            confidence=float(d.confidence),
                            method=used_method,
                            page=None,
                            dtype=d.dtype,
                        )
                    )
                    if d.dtype == "text":
                        replaced_text += 1
                    else:
                        replaced_logo += 1
                found = True
                reason = None
            else:
                # No detections -> place a small default logo
                overlay_logo(
                    base_image_path=img_path,
                    logo_path=logo_file,
                    output_path=out_path,
                    position=(12, 12),
                    scale=0.15,
                    opacity=CONFIG.logo_opacity,
                    max_logo_size_ratio=0.25,
                )
                found = False
                reason = "No region found (default placement used)"

            summaries.append(
                PerFileDetectionSummary(
                    file=str(img_path),
                    found=found,
                    method=used_method,
                    boxes=boxes_for_summary,
                    reason=reason,
                    replaced_count=replaced_logo + replaced_text,
                    replaced_logo_count=replaced_logo,
                    replaced_text_count=replaced_text,
                )
            )

            progress = int(10 + (idx * 80) / total)
            job_store.update_status(job_id, progress=progress, message=f"Processed {idx}/{total} pages", detections=summaries)

        # Preserve per-file outputs under result/out with original structure
        out_dir = result_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in work.rglob("*"):
            if p.is_file():
                rel = p.relative_to(work)
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())

        # Package into zip
        zip_path = result_dir / "processed.zip"
        create_zip_from_directory(out_dir, zip_path)

        download_url = f"/jobs/{job_id}/download"
        job_store.set_result_url(job_id, result_url=download_url)
        job_store.update_status(job_id, status=JobState.COMPLETED, progress=100, message="Completed", detections=summaries)
    except Exception as exc:  # pragma: no cover - safety
        job_store.save_error(job_id, f"Processing failed: {exc}")
