# Logo Replacement Backend

This backend processes engineering drawing images and PDFs to detect and replace existing logos with a new one. It supports:
- OpenAI Vision for robust bounding-box detection of logo regions.
- Deterministic OpenCV template matching as a fallback (with optional user-provided old logo reference).
- PDF rasterization to per-page images before detection.

Key endpoints (see FastAPI docs at /docs):
- POST /jobs -> create a job
- POST /jobs/{job_id}/upload -> upload drawings and logo (supports optional old_logo_image)
- POST /jobs/{job_id}/start -> start processing
- GET /jobs/{job_id}/status -> poll status (includes per-file detection summaries)
- GET /jobs/{job_id}/download -> download processed ZIP
- GET /jobs/{job_id}/files -> list per-file outputs
- GET /jobs/{job_id}/files/{file} -> download a single output
- POST /process -> single-shot upload + start (supports optional old_logo_image)

## Detection Pipeline

The pipeline detects existing logo regions per page/image and overlays the new logo into the detected box:
- Method selection via DETECTION_METHOD:
  - vision: use OpenAI Vision only
  - template: use OpenCV template matching only
  - auto (default): attempt Vision if OPENAI_API_KEY is set, otherwise template
- Vision returns normalized boxes (0..1). The backend maps them back to full resolution and overlays with alpha.
- Template matching uses cv2.matchTemplate against:
  - user-provided old_logo_image if supplied, OR
  - auto-derived high-contrast region as a heuristic template
- For each processed file, a detection summary is recorded and exposed in JobStatus.detections.

## New multipart field

To improve template matching, you can provide the old logo reference:
- old_logo_image (optional): image of the old logo to be replaced

Accepted by:
- POST /jobs/{job_id}/upload
- POST /process

## Environment variables

- DETECTION_METHOD: vision | template | auto (default: auto)
- DETECTION_CONFIDENCE_THRESHOLD: float [0..1] (default: 0.5)
- LOGO_MAX_WIDTH_PX: max rendered logo width in pixels when fitting into detected box (default: 600)
- LOGO_OPACITY: float [0..1] opacity for the overlayed logo (default: 0.9)
- DETECT_THUMB_MAX_PX: max image dimension used for detection thumbnails (default: 1280)
- OPENAI_VISION_MODEL: OpenAI model for Vision (default: gpt-4o-mini)
- VERBOSE_LOGGING: "true" for more verbose logs

OpenAI configuration:
- OPENAI_API_KEY: required for Vision
- OPENAI_API_BASE: optional override base URL (default: https://api.openai.com)

## PDF Support

- PDFs are rasterized to images per page before detection.
- Preferred: PyMuPDF (fitz) [self-contained].
- Fallback: pdf2image (requires Poppler installed on the system).
- Configuration:
  - PDF_RASTERIZE_DPI (default: 200)
  - PDF_RASTERIZE_FORMAT (default: png)

## Installation notes

Install Python dependencies from requirements.txt. If OpenCV requires system libraries in your environment, ensure they are present or use opencv-python-headless as provided.

```bash
pip install -r requirements.txt
```
