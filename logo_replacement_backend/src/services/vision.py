from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Detection:
    """Simple rectangle detection result."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.5


class VisionClient:
    """Pluggable vision detection client.

    If OPENAI_API_KEY is present, a future implementation can call OpenAI's
    vision models to detect logos. For this demo, we provide a safe stub which
    returns an empty detection list so processing falls back to a default
    placement.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "VisionClient":
        """Instantiate from environment variables."""
        return cls(api_key=os.getenv("OPENAI_API_KEY"))

    # PUBLIC_INTERFACE
    def detect_logos(self, image_path: Path) -> List[Detection]:
        """Detect logo bounding boxes on the given image.

        Returns:
            A list of Detection rectangles. Empty list means no detection.

        Notes:
            - For now, this method returns an empty list (stub).
            - If OPENAI_API_KEY is configured, you could extend this class to
              call OpenAI APIs. Keep network calls optional and guarded.
        """
        # Stub: no detections
        return []
