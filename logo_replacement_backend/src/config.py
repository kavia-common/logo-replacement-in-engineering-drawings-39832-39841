from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration options for logo and text detection and overlay behavior."""
    # Logo detection method (existing)
    method: str = os.getenv("DETECTION_METHOD", "auto").lower()  # vision|template|auto
    confidence_threshold: float = float(os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.5"))
    logo_max_width_px: int = int(os.getenv("LOGO_MAX_WIDTH_PX", "600"))
    logo_opacity: float = float(os.getenv("LOGO_OPACITY", "0.9"))

    # Thumbnail for detection to speed up Vision/Template matching
    detect_thumbnail_max_px: int = int(os.getenv("DETECT_THUMB_MAX_PX", "1280"))

    # OpenAI model name (if using vision)
    openai_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

    # Logging verbosity
    verbose_logging: bool = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

    # New: brand-text configuration and strategy
    brand_keyword: str = os.getenv("BRAND_KEYWORD", "").strip()
    detection_text_method: str = os.getenv("DETECTION_TEXT_METHOD", "auto").lower()  # vision|heuristic|auto
    max_replacements_per_page: int = int(os.getenv("MAX_REPLACEMENTS_PER_PAGE", "3"))

    # Optional OCR configuration (tesseract)
    enable_tesseract: bool = os.getenv("ENABLE_TESSERACT", "false").lower() == "true"
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "").strip()  # optional absolute path

    # Overlay placement configuration
    overlay_fit_mode: str = os.getenv("OVERLAY_FIT_MODE", "contain").lower()  # contain | cover
    overlay_padding_pct: float = float(os.getenv("OVERLAY_PADDING_PCT", "0.0"))  # 0..40 typical

    # Debugging and QA
    # Always render debug overlays if true: detection and placed-logo outlines into result/debug
    debug_overlay: bool = os.getenv("DEBUG_OVERLAY", "false").lower() == "true"
    # If true, enable a QA bundle to be downloadable and add /jobs/{id}/qa endpoint
    enable_qa_bundle: bool = os.getenv("ENABLE_QA_BUNDLE", "true").lower() == "true"
    # If true, force outlines rendering regardless of debug_overlay (temporary dev flag)
    force_debug_outlines: bool = os.getenv("FORCE_DEBUG_OUTLINES", "true").lower() == "true"


CONFIG = DetectionConfig()
