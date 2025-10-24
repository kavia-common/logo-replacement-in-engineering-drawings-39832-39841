# logo-replacement-in-engineering-drawings-39832-39841

## PDF Support
- The backend now accepts PDFs inside the uploaded drawings ZIP. Each PDF page is rasterized into images before logo detection.
- Preferred engine: PyMuPDF (fitz) [self-contained].
- Fallback: pdf2image (requires Poppler installed on the system).
- Configuration:
  - PDF_RASTERIZE_DPI (default: 200)
  - PDF_RASTERIZE_FORMAT (default: png)

If you use pdf2image, ensure Poppler is installed and on PATH (Linux: `apt-get install poppler-utils`, Mac: `brew install poppler`).
