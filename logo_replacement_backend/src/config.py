from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration options for logo detection and overlay behavior."""
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


CONFIG = DetectionConfig()
