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
    dtype: str = "logo"  # logo | text


def _encode_image_to_data_url(img: Image.Image) -> str:
    """Encode PIL image to data URL (PNG) for OpenAI vision input."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_box(box: dict) -> Optional[Tuple[float, float, float, float, float, str]]:
    """
    Parse a box dict supporting multiple shapes:
    - top-left normalized: {x,y,w,h}
    - center-based normalized: {cx,cy,w,h} or {center_x,center_y,width,height}
    - fractional alias keys: {left,top,width,height} as fractions
    All values expected in 0..1.
    Returns tuple (x,y,w,h,confidence,type) normalized 0..1.
    """
    try:
        # Confidence and type
        conf = float(box.get("confidence", box.get("score", 0.5)))
        dtype = str(box.get("type", box.get("category", "logo"))).lower().strip()
        if dtype not in ("logo", "text"):
            dtype = "logo"

        # Prefer explicit x,y,w,h
        if all(k in box for k in ("x", "y", "w", "h")):
            x = float(box["x"]); y = float(box["y"]); w = float(box["w"]); h = float(box["h"])
        elif all(k in box for k in ("left", "top", "width", "height")):
            x = float(box["left"]); y = float(box["top"]); w = float(box["width"]); h = float(box["height"])
        elif all(k in box for k in ("cx", "cy", "w", "h")):
            cx = float(box["cx"]); cy = float(box["cy"]); w = float(box["w"]); h = float(box["h"])
            x = cx - w / 2.0; y = cy - h / 2.0
        elif all(k in box for k in ("center_x", "center_y", "width", "height")):
            cx = float(box["center_x"]); cy = float(box["center_y"]); w = float(box["width"]); h = float(box["height"])
            x = cx - w / 2.0; y = cy - h / 2.0
        else:
            return None

        # Validate normalized ranges, clamp slightly for minor drift
        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))
        x = _clamp01(x); y = _clamp01(y); w = _clamp01(w); h = _clamp01(h)
        if w <= 0.0 or h <= 0.0:
            return None
        # Ensure TL inside image
        if x > 1.0 or y > 1.0:
            return None
        return x, y, w, h, conf, dtype
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
        """Backward-compatible: detect only logos using primary pipeline."""
        regions = self.detect_regions(image_path, old_logo_image=old_logo_image)
        return [d for d in regions if d.dtype == "logo"]

    # PUBLIC_INTERFACE
    def detect_regions(
        self,
        image_path: Path,
        *,
        old_logo_image: Optional[Path] = None,
    ) -> List[Detection]:
        """Detect both logo marks and branded text regions.

        Returns:
            A list of Detection rectangles in pixel coordinates, each with dtype in {logo, text}.

        Strategy:
            - Logo detection via existing 'method' (vision/template/auto).
            - Text detection based on CONFIG.detection_text_method:
              - vision: ask OpenAI Vision for text regions containing BRAND_KEYWORD (if provided)
              - heuristic: OpenCV-based MSER/contours and (optional) OCR filter
              - auto: try vision if API available else heuristic
        """
        api_available = bool(self.api_key)

        detections: List[Detection] = []

        # 1) Logo regions (existing logic)
        logo_method = CONFIG.method
        if logo_method == "vision":
            boxes = self._detect_with_openai(image_path, want_text=False)
            if boxes is None:
                boxes = self._detect_with_template(image_path, old_logo_image)
        elif logo_method == "template":
            boxes = self._detect_with_template(image_path, old_logo_image)
        else:
            boxes = self._detect_with_openai(image_path, want_text=False) if api_available else None
            if boxes is None:
                boxes = self._detect_with_template(image_path, old_logo_image)
        for b in boxes or []:
            b.dtype = "logo"
            detections.append(b)

        # 2) Text regions
        text_method = CONFIG.detection_text_method
        text_boxes: List[Detection] = []
        if text_method == "vision":
            text_boxes = self._detect_with_openai(image_path, want_text=True) or []
        elif text_method == "heuristic":
            text_boxes = self._detect_text_heuristic(image_path)
        else:
            # auto
            if api_available:
                text_boxes = self._detect_with_openai(image_path, want_text=True) or []
            if not text_boxes:
                text_boxes = self._detect_text_heuristic(image_path)

        # Normalize dtype and append
        for tb in text_boxes:
            tb.dtype = "text"
            detections.append(tb)

        return detections

    def _detect_with_openai(self, image_path: Path, want_text: bool = False) -> Optional[List[Detection]]:
        """Use OpenAI Vision to detect normalized bounding boxes. Return None on failure.

        If want_text=True, request detection of branded text regions possibly matching CONFIG.brand_keyword.
        """
        if not self.api_key:
            return None

        # Prepare thumbnail
        try:
            img = Image.open(image_path)
        except Exception:
            return None
        thumb, scale = _resize_for_detection(img, CONFIG.detect_thumbnail_max_px)
        data_url = _encode_image_to_data_url(thumb)

        # Compose prompt to strictly return JSON with 'type' field
        brand = CONFIG.brand_keyword
        text_hint = ""
        if want_text:
            if brand:
                text_hint = (
                    f" Also detect any occurrences of brand text matching '{brand}' (case-insensitive), "
                    "including variants or stylized text."
                )
            else:
                text_hint = " Also detect any prominent brand/name text likely representing company names."

        system_prompt = (
            "You are a vision assistant specialized in document mark-up. "
            "Identify existing company logos, brand marks, and optionally branded text in engineering drawings. "
            "Return a pure JSON object with an array 'boxes' where each box has "
            "keys: x, y, w, h (all normalized 0-1 relative to the image width/height), "
            "confidence (0-1), and type ('logo'|'text'). Do not include any text outside JSON."
        )
        user_prompt = (
            "Detect existing logos or brand marks (stamps at title block, corner logos, etc.)."
            f"{text_hint} "
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
                    nx, ny, nw, nh, conf, dtype = norm
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
                    dets.append(Detection(x=px, y=py, width=pw, height=ph, confidence=conf, dtype=dtype))
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

    def _detect_text_heuristic(self, image_path: Path) -> List[Detection]:
        """Detect text-like regions using OCR-free heuristics (MSER + contour filtering).
        Optionally filter by BRAND_KEYWORD using Tesseract if enabled by config and available.
        """
        try:
            cv2, np = _load_cv2()
        except Exception:
            return []

        # Load image
        try:
            pil = Image.open(image_path)
            small, scale = _resize_for_detection(pil, CONFIG.detect_thumbnail_max_px)
            img = _pil_to_cv(small)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            return []

        H, W = gray.shape[:2]

        # MSER to find text-like stable regions
        try:
            mser = cv2.MSER_create(_min_area=60, _max_area=max(3000, int(0.05 * W * H)))
            regions, _ = mser.detectRegions(gray)
            mask = np.zeros((H, W), dtype=np.uint8)
            for p in regions:
                cv2.fillPoly(mask, [p.reshape(-1, 1, 2)], 255)
        except Exception:
            # Fallback to simple threshold
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Morphological operations to group characters
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        brand = CONFIG.brand_keyword.lower()
        boxes: List[Detection] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Basic geometry filters: width>height (likely text line), reasonable size
            if w < 20 or h < 10:
                continue
            if w > 0.95 * W or h > 0.5 * H:
                continue
            aspect = w / float(h)
            if aspect < 1.2:  # prefer elongated regions
                continue

            # Map back to original coordinates
            def unscale(v: float) -> float:
                return v / (scale if scale != 0 else 1.0)

            det = Detection(x=unscale(x), y=unscale(y), width=unscale(w), height=unscale(h), confidence=0.55, dtype="text")

            # If tesseract is enabled and brand_keyword provided, do quick check
            if CONFIG.enable_tesseract and brand:
                try:
                    import pytesseract  # type: ignore
                    if CONFIG.tesseract_cmd:
                        pytesseract.pytesseract.tesseract_cmd = CONFIG.tesseract_cmd
                    # Crop ROI on thumbnail to speed up
                    roi = gray[y : y + h, x : x + w]
                    roi_pil = Image.fromarray(cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
                    txt = pytesseract.image_to_string(roi_pil)
                    if brand in (txt or "").lower():
                        det.confidence = 0.75
                    else:
                        # Skip if OCR did not find the brand string
                        continue
                except Exception:
                    # Ignore OCR failures and keep heuristic result
                    pass

            boxes.append(det)

        # Deduplicate overlapping boxes (NMS-like)
        boxes = self._nms(boxes, iou_thresh=0.3)
        return boxes

    def _nms(self, dets: List[Detection], iou_thresh: float = 0.3) -> List[Detection]:
        """Simple Non-Max Suppression for overlapping detections."""
        if not dets:
            return dets
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        def iou(a: Detection, b: Detection) -> float:
            ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.width, a.y + a.height
            bx1, by1, bx2, by2 = b.x, b.y, b.x + b.width, b.y + b.height
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = a.width * a.height
            area_b = b.width * b.height
            return inter / (area_a + area_b - inter + 1e-6)
        for d in dets:
            if all(iou(d, k) < iou_thresh for k in kept):
                kept.append(d)
        return kept
