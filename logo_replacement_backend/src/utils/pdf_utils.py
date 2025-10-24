from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PdfRasterizationConfig:
    """Configuration for PDF rasterization behavior."""
    dpi: int = int(os.getenv("PDF_RASTERIZE_DPI", "200"))
    fmt: str = os.getenv("PDF_RASTERIZE_FORMAT", "png").lower()
    # If True, we will try best-effort fallback between engines
    allow_fallback: bool = True


class PdfRasterizerUnavailable(Exception):
    """Raised when no PDF rasterizer is available in the environment."""


def _rasterize_with_pymupdf(pdf_path: Path, out_dir: Path, cfg: PdfRasterizationConfig) -> List[Path]:
    """Rasterize using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise PdfRasterizerUnavailable("PyMuPDF (fitz) is not available") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    images: List[Path] = []
    zoom = cfg.dpi / 72.0  # 72 dpi base in PDF points
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = out_dir / f"{pdf_path.stem}_page-{i:03d}.{cfg.fmt}"
            pix.save(out_path.as_posix())
            images.append(out_path)
    return images


def _rasterize_with_pdf2image(pdf_path: Path, out_dir: Path, cfg: PdfRasterizationConfig) -> List[Path]:
    """Rasterize using pdf2image (requires poppler)."""
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise PdfRasterizerUnavailable("pdf2image is not available") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path.as_posix(), dpi=cfg.dpi)
    images: List[Path] = []
    for i, img in enumerate(pages, start=1):
        out_path = out_dir / f"{pdf_path.stem}_page-{i:03d}.{cfg.fmt}"
        img.save(out_path.as_posix(), cfg.fmt.upper())
        images.append(out_path)
    return images


# PUBLIC_INTERFACE
def rasterize_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    cfg: Optional[PdfRasterizationConfig] = None,
) -> List[Path]:
    """Rasterize a PDF into individual page images.

    The function attempts PyMuPDF (fitz) first for self-contained rasterization.
    If unavailable and allow_fallback=True, it falls back to pdf2image (requires Poppler).

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Directory where resulting images will be written.
        cfg: Optional PdfRasterizationConfig for dpi/format.

    Returns:
        A list of image file paths (one per page).

    Raises:
        PdfRasterizerUnavailable: If no rasterizer is available in the environment.
        Exception: On other IO/rasterization errors.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    cfg = cfg or PdfRasterizationConfig()

    # Try PyMuPDF (preferred)
    try:
        return _rasterize_with_pymupdf(pdf_path, out_dir, cfg)
    except PdfRasterizerUnavailable:
        if not cfg.allow_fallback:
            raise
    except Exception:
        # If PyMuPDF exists but failed unexpectedly, and fallback allowed, try next
        if not cfg.allow_fallback:
            raise

    # Fallback: pdf2image
    return _rasterize_with_pdf2image(pdf_path, out_dir, cfg)
