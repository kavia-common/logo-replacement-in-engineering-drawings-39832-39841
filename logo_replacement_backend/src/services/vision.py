from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from src.config import CONFIG


@dataclass
class Detection:
    """Simple rectangle detection result in absolute pixels."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.5


def _encode_image_to_data_url(img: Image.Image) -> str:
    """Encode PIL image to data URL (PNG) for OpenAI vision input."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_box(box: dict) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Parse a box dict that should have:
    { "x": 0-1, "y": 0-1, "w": 0-1, "h": 0-1, "confidence": 0-1 }
    """
    try:
        x = float(box.get("x"))
        y = float(box.get("y"))
        w = float(box.get("w"))
        h = float(box.get("h"))
        conf = float(box.get("confidence", 0.5))
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None
        return x, y, w, h, conf
    except Exception:
        return None


def _load_cv2():
    """Lazy import cv2 to avoid dependency issues when not needed."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    return cv2, np


def _pil_to_cv(img: Image.Image):
    """Convert PIL image to OpenCV BGR numpy array."""
    from numpy import array  # type: ignore
    if img.mode != "RGB":
        img = img.convert("RGB")
    return array(img)[:, :, ::-1]  # RGB -> BGR


def _cv_to_pil(arr):
    """Convert OpenCV BGR numpy array to PIL Image."""
    import cv2  # type: ignore
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _resize_for_detection(img: Image.Image, max_px: int) -> Tuple[Image.Image, float]:
    """Resize image so max(width,height) <= max_px, return resized and scale factor."""
    w, h = img.size
    max_side = max(w, h)
    if max_side <= max_px:
        return img.copy(), 1.0
    scale = max_px / max_side
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS), scale


class VisionClient:
    """Pluggable vision detection client with OpenAI Vision + OpenCV fallback."""

    def __init__(self, api_key: str | None = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or CONFIG.openai_model

    @classmethod
    def from_env(cls) -> "VisionClient":
        """Instantiate from environment variables."""
        return cls(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_VISION_MODEL"))

    # PUBLIC_INTERFACE
    def detect_logos(
        self,
        image_path: Path,
        *,
        old_logo_image: Optional[Path] = None,
    ) -> List[Detection]:
        """Detect logo bounding boxes on the given image.

        Returns:
            A list of Detection rectangles in pixel coordinates, may be empty.

        Strategy:
            - If CONFIG.method == "vision": try OpenAI, fallback to template if fails.
            - If "template": use OpenCV template matching.
            - If "auto": try vision first if OPENAI_API_KEY exists; otherwise template.
        """
        method = CONFIG.method
        api_available = bool(self.api_key)

        if method == "vision":
            boxes = self._detect_with_openai(image_path)
            if boxes is not None:
                return boxes
            # fallback
            return self._detect_with_template(image_path, old_logo_image)
        elif method == "template":
            return self._detect_with_template(image_path, old_logo_image)
        else:
            # auto
            if api_available:
                boxes = self._detect_with_openai(image_path)
                if boxes is not None:
                    return boxes
            return self._detect_with_template(image_path, old_logo_image)

    def _detect_with_openai(self, image_path: Path) -> Optional[List[Detection]]:
        """Use OpenAI Vision to detect normalized bounding boxes. Return None on failure."""
        if not self.api_key:
            return None

        # Prepare thumbnail
        try:
            img = Image.open(image_path)
        except Exception:
            return None
        thumb, scale = _resize_for_detection(img, CONFIG.detect_thumbnail_max_px)
        data_url = _encode_image_to_data_url(thumb)

        # Compose prompt to strictly return JSON
        system_prompt = (
            "You are a vision assistant specialized in document mark-up. "
            "Identify existing company logos or brand marks in the image. "
            "Return a pure JSON object with an array 'boxes' where each box has "
            "keys: x, y, w, h (all normalized 0-1 relative to the image width/height) "
            "and confidence (0-1). Do not include any text outside JSON."
        )
        user_prompt = (
            "Detect existing logos or brand marks (stamps at title block, corner logos, etc.). "
            "If none are present, return {\"boxes\": []}."
        )
        try:
            # Use httpx to call OpenAI REST (avoid adding openai SDK dependency)
            import httpx  # type: ignore

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_data": data_url},
                        ],
                    },
                ],
                "max_output_tokens": 300,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Try the new Responses API endpoint
            url = os.getenv("OPENAI_API_BASE", "https://api.openai.com") + "/v1/responses"
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    return None
                data = resp.json()
                # Depending on API shape, extract text from output
                text_out = None
                if "output" in data and isinstance(data["output"], list):
                    # e.g., responses output array
                    parts = [p.get("content", "") if isinstance(p, dict) else "" for p in data["output"]]
                    text_out = "".join(parts)
                elif "choices" in data and data["choices"]:
                    text_out = data["choices"][0].get("message", {}).get("content")
                if not text_out:
                    # Some models return structured content; try 'output_text'
                    text_out = data.get("output_text")

                if not text_out:
                    return None

                # Parse JSON
                text_out = text_out.strip()
                # Try to find a JSON object in the text
                start = text_out.find("{")
                end = text_out.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    return None
                json_str = text_out[start : end + 1]
                parsed = json.loads(json_str)
                boxes = parsed.get("boxes", [])
                dets: List[Detection] = []
                W, H = thumb.size
                for b in boxes:
                    norm = _normalize_box(b)
                    if not norm:
                        continue
                    nx, ny, nw, nh, conf = norm
                    if conf < CONFIG.confidence_threshold:
                        continue
                    # Map to pixels on thumbnail then we'll rescale up to original using 1/scale
                    px = nx * W
                    py = ny * H
                    pw = nw * W
                    ph = nh * H
                    # Rescale to original
                    if scale != 0:
                        px = px / scale
                        py = py / scale
                        pw = pw / scale
                        ph = ph / scale
                    dets.append(Detection(x=px, y=py, width=pw, height=ph, confidence=conf))
                return dets
        except Exception:
            return None

    def _detect_with_template(self, image_path: Path, old_logo_image: Optional[Path]) -> List[Detection]:
        """
        Template matching fallback using OpenCV.
        If old_logo_image is provided, use it as template.
        Else, auto-derive a candidate template by searching for high-contrast patches.
        """
        try:
            cv2, np = _load_cv2()
        except Exception:
            # If cv2 is not available, no detection
            return []

        try:
            base_pil = Image.open(image_path)
            base_small, scale = _resize_for_detection(base_pil, CONFIG.detect_thumbnail_max_px)
            base = _pil_to_cv(base_small)
            base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        except Exception:
            return []

        template_img = None

        # If user provided template
        if old_logo_image and Path(old_logo_image).exists():
            try:
                tpl_pil = Image.open(old_logo_image)
                # Resize template if too large (relative to detection image)
                tpl_pil_resized, _ = _resize_for_detection(tpl_pil, min(base_small.size))
                template_img = _pil_to_cv(tpl_pil_resized)
            except Exception:
                template_img = None

        # If no template, try to auto-derive via high-contrast regions (simple heuristic)
        if template_img is None:
            try:
                # Use Canny edges and find largest dense subregion as a crude "logo-like" marker
                edges = cv2.Canny(base_gray, 50, 150)
                kernel = np.ones((3, 3), np.uint8)
                dil = cv2.dilate(edges, kernel, iterations=2)
                # Find contours and grab bounding boxes
                contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [cv2.boundingRect(cnt) for cnt in contours]
                # Filter boxes that are reasonably small (logos usually smaller than 25% width/height)
                H, W = base_gray.shape[:2]
                cands = [(x, y, w, h) for (x, y, w, h) in boxes if w < 0.35 * W and h < 0.35 * H and w > 12 and h > 12]
                # Pick the one with highest edge density score (sum of edges in box / area)
                def score(b):
                    x, y, w, h = b
                    roi = edges[y : y + h, x : x + w]
                    return float(roi.sum()) / float(max(1, w * h))
                cands.sort(key=score, reverse=True)
                if cands:
                    x, y, w, h = cands[0]
                    x2 = min(W, x + w)
                    y2 = min(H, y + h)
                    template_img = base[y:y2, x:x2].copy()
            except Exception:
                template_img = None

        if template_img is None:
            return []

        try:
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            # Use normalized cross correlation
            res = cv2.matchTemplate(base_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Estimate confidence between 0-1
            confidence = float(max_val)
            if confidence < CONFIG.confidence_threshold:
                return []

            th, tw = template_gray.shape[:2]
            top_left = max_loc
            # Map coordinates from resized base back to original
            x = top_left[0] / (scale if scale != 0 else 1.0)
            y = top_left[1] / (scale if scale != 0 else 1.0)
            w = tw / (scale if scale != 0 else 1.0)
            h = th / (scale if scale != 0 else 1.0)

            return [Detection(x=x, y=y, width=w, height=h, confidence=confidence)]
        except Exception:
            return []
