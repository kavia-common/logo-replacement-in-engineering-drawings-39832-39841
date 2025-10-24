from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from PIL import Image, ImageEnhance, ImageDraw

from src.config import CONFIG


def _apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    """Apply opacity to an RGBA image and return new image."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if opacity >= 1.0:
        return img
    alpha = img.split()[-1]
    enhancer = ImageEnhance.Brightness(alpha)
    alpha = enhancer.enhance(max(0.0, min(1.0, opacity)))
    out = img.copy()
    out.putalpha(alpha)
    return out


def _fit_logo(logo: Image.Image, box_w: int, box_h: int, fit_mode: str) -> Image.Image:
    """Resize logo to fit within box using contain or cover, preserving aspect."""
    lw, lh = logo.size
    if lw <= 0 or lh <= 0:
        return logo
    aspect = lw / float(lh)

    # Initial target size equals box
    tw, th = box_w, box_h

    if fit_mode == "cover":
        # Scale so that the resized logo covers the box fully; some parts may overflow and be clipped.
        if (box_w / float(box_h)) > aspect:
            # Box is wider than logo aspect -> scale by width to cover height
            tw = min(box_w, CONFIG.logo_max_width_px)
            th = int(round(tw / aspect))
            if th < box_h:
                th = box_h
                tw = int(round(th * aspect))
        else:
            # Box is taller -> scale by height to cover width
            th = box_h
            tw = int(round(th * aspect))
            if tw < box_w:
                tw = min(box_w, CONFIG.logo_max_width_px)
                th = int(round(tw / aspect))
    else:
        # Default: contain - fit entirely inside
        tw = min(box_w, CONFIG.logo_max_width_px)
        th = int(round(tw / aspect))
        if th > box_h:
            th = box_h
            tw = int(round(th * aspect))

    tw = max(1, int(tw))
    th = max(1, int(th))
    return logo.resize((tw, th), resample=Image.LANCZOS)


def _clamp_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Ensure box lies within image boundaries and has at least 1x1 size."""
    x = max(0, min(x, max(0, img_w - 1)))
    y = max(0, min(y, max(0, img_h - 1)))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _apply_padding(x: int, y: int, w: int, h: int, pad_pct: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Apply symmetric padding inside the box; pad_pct is fraction of min(w,h)."""
    pad_pct = max(0.0, min(0.9, pad_pct))
    pad = int(round(min(w, h) * pad_pct))
    nx = x + pad
    ny = y + pad
    nw = w - 2 * pad
    nh = h - 2 * pad
    if nw < 1 or nh < 1:
        # If over-padded, fallback to minimal inner box of 1px
        nx, ny, nw, nh = x, y, w, h
    return _clamp_box(nx, ny, nw, nh, img_w, img_h)


def _save_debug_preview(base_rgba: Image.Image, x: int, y: int, w: int, h: int, placed_w: int, placed_h: int, out_path: Path) -> None:
    """Save a debug image drawing the detection and placed logo rectangle outline."""
    dbg = base_rgba.convert("RGB").copy()
    draw = ImageDraw.Draw(dbg)
    # Detected box outline in yellow
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 221, 0), width=2)
    # Placed logo rectangle centered inside box in cyan (approx preview)
    cx = x + (w - placed_w) // 2
    cy = y + (h - placed_h) // 2
    draw.rectangle([cx, cy, cx + placed_w - 1, cy + placed_h - 1], outline=(0, 255, 255), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dbg.save(out_path)


# PUBLIC_INTERFACE
def place_logo_in_box(
    base_image_path: Path,
    logo_path: Path,
    output_path: Path,
    *,
    box: Tuple[int, int, int, int],
    opacity: Optional[float] = None,
    fit_mode: Optional[str] = None,
    padding_pct: Optional[float] = None,
    debug_preview_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Place the logo strictly inside the given box and composite within the bounding rectangle.

    - Computes the effective target rectangle after padding and clamping.
    - Resizes the logo using contain or cover fit while preserving aspect ratio.
    - Centers the resized logo within the target rectangle.
    - Alpha-blends only within the target rectangle to avoid spillover.

    Returns:
        A dict with placement info: { x, y, width, height, fit_mode, padding_pct, logo_w, logo_h }
    """
    opacity = CONFIG.logo_opacity if opacity is None else opacity
    fit_mode = (fit_mode or CONFIG.overlay_fit_mode or "contain").lower()
    if fit_mode not in ("contain", "cover"):
        fit_mode = "contain"
    padding_pct = CONFIG.overlay_padding_pct if padding_pct is None else padding_pct

    base = Image.open(base_image_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")
    base_w, base_h = base.size

    # Clamp the input box to image boundaries
    x, y, w, h = _clamp_box(int(box[0]), int(box[1]), int(box[2]), int(box[3]), base_w, base_h)

    # Apply inner padding (optional)
    x, y, w, h = _apply_padding(x, y, w, h, float(padding_pct or 0.0), base_w, base_h)

    # Resize logo
    logo_resized = _fit_logo(logo, w, h, fit_mode)
    logo_resized = _apply_opacity(logo_resized, opacity)
    lw, lh = logo_resized.size

    # Center within target rectangle
    px = x + max(0, (w - lw) // 2)
    py = y + max(0, (h - lh) // 2)

    # Composite only within the bounding rectangle for safety
    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.paste(logo_resized, (px, py), mask=logo_resized)
    out = Image.alpha_composite(base, canvas)

    # Persist with DPI if available
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = base.info.copy()
    dpi = info.get("dpi")

    fmt = output_path.suffix.lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        out_to_save = out.convert("RGB")
        if dpi:
            out_to_save.save(output_path, dpi=dpi, quality=95, subsampling=0)
        else:
            out_to_save.save(output_path, quality=95, subsampling=0)
    else:
        if dpi:
            out.save(output_path, dpi=dpi)
        else:
            out.save(output_path)

    # Optional debug preview
    if debug_preview_path:
        try:
            _save_debug_preview(base, x, y, w, h, lw, lh, Path(debug_preview_path))
        except Exception:
            # Debug is best-effort; ignore failures
            pass

    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "fit_mode": fit_mode,
        "padding_pct": float(padding_pct or 0.0),
        "logo_w": lw,
        "logo_h": lh,
        "placed_x": px,
        "placed_y": py,
    }


# PUBLIC_INTERFACE
def overlay_logo(
    base_image_path: Path,
    logo_path: Path,
    output_path: Path,
    *,
    position: Tuple[int, int] = (10, 10),
    scale: float = 0.2,
    opacity: float = None,
    max_logo_size_ratio: float = 0.4,
    target_box: Tuple[int, int, int, int] | None = None,  # x, y, w, h in pixels (optional, from detection)
) -> None:
    """Overlay a logo image onto a base image with alpha and size constraints.

    Args:
        base_image_path: Base image path.
        logo_path: Logo image path.
        output_path: Output image path.
        position: Top-left position if target_box is not provided.
        scale: Relative scale w.r.t base width if target_box is not provided.
        opacity: Opacity [0-1]; defaults to CONFIG.logo_opacity.
        max_logo_size_ratio: Max ratio of base dimensions used when using scale path.
        target_box: If provided (x, y, w, h), logo will be fitted within this box.

    Notes:
        - If target_box is provided, it takes precedence.
        - Preserves DPI metadata if present.
    """
    opacity = CONFIG.logo_opacity if opacity is None else opacity

    # If a target box is given, delegate to place_logo_in_box to ensure strict behavior
    if target_box:
        place_logo_in_box(
            base_image_path=base_image_path,
            logo_path=logo_path,
            output_path=output_path,
            box=target_box,
            opacity=opacity,
            fit_mode=CONFIG.overlay_fit_mode,
            padding_pct=CONFIG.overlay_padding_pct,
            debug_preview_path=None,  # legacy path doesn't write debug by default
        )
        return

    # Legacy positioning path for no-detections fallback
    base = Image.open(base_image_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    base_w, base_h = base.size
    target_w = int(base_w * scale)
    aspect = logo.width / max(1, logo.height)
    target_h = int(target_w / aspect)
    max_w = min(int(base_w * max_logo_size_ratio), CONFIG.logo_max_width_px)
    max_h = int(base_h * max_logo_size_ratio)
    target_w = min(max_w, target_w)
    target_h = min(max_h, target_h)
    target_w = max(1, target_w)
    target_h = max(1, target_h)
    logo_resized = logo.resize((target_w, target_h), resample=Image.LANCZOS)
    logo_resized = _apply_opacity(logo_resized, opacity)
    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.paste(logo_resized, position, mask=logo_resized)
    out = Image.alpha_composite(base, canvas)

    # Save with DPI preserved when possible
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = base.info.copy()
    dpi = info.get("dpi", None)

    fmt = output_path.suffix.lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        out_to_save = out.convert("RGB")
        if dpi:
            out_to_save.save(output_path, dpi=dpi, quality=95, subsampling=0)
        else:
            out_to_save.save(output_path, quality=95, subsampling=0)
    else:
        if dpi:
            out.save(output_path, dpi=dpi)
        else:
            out.save(output_path)
