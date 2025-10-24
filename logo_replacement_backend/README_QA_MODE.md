This backend includes a QA/debug mode to verify detection and placement:

- Environment flags:
  - DEBUG_OVERLAY=true to render per-detection debug previews with outlines into jobs/{id}/result/debug/
  - ENABLE_QA_BUNDLE=true to write jobs/{id}/result/qa/meta.json with per-file coordinates and placement info
  - FORCE_DEBUG_OUTLINES=true temporarily forces outlines even if DEBUG_OVERLAY is false (default true for diagnosis)

- New endpoint:
  - GET /jobs/{job_id}/qa
    Returns:
      {
        enabled: boolean,
        meta_json: "/jobs/{id}/files/qa/meta.json" | null,
        debug_images: ["/jobs/{id}/files/debug/<img>.png", ...]
      }

- What is logged:
  - For each processed file: original size, detection absolute rectangles, final placement rectangles, fit_mode, padding_pct
  - A boolean assertion per placement stating whether the placed area is fully within the detection box.

- Visual overlays:
  - Debug preview PNGs show detection box (yellow) and placed-logo rectangle (cyan).
