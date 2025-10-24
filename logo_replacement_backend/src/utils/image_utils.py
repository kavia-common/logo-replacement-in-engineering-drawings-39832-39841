from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance

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


def _fit_logo_into_box(logo: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize logo to fit within box preserving aspect ratio."""
    lw, lh = logo.size
    if lw == 0 or lh == 0:
        return logo
    aspect = lw / lh
    # Fit by width first, then ensure height fits
    target_w = min(box_w, CONFIG.logo_max_width_px)
    target_h = int(target_w / aspect)
    if target_h > box_h:
        target_h = box_h
        target_w = int(target_h * aspect)
    target_w = max(1, target_w)
    target_h = max(1, target_h)
    return logo.resize((target_w, target_h), resample=Image.LANCZOS)


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

    base = Image.open(base_image_path)
    base = base.convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    base_w, base_h = base.size

    if target_box:
        x, y, w, h = target_box
        # Ensure box within image
        x = max(0, min(int(x), base_w - 1))
        y = max(0, min(int(y), base_h - 1))
        w = max(1, min(int(w), base_w - x))
        h = max(1, min(int(h), base_h - y))

        logo_fitted = _fit_logo_into_box(logo, w, h)
        logo_fitted = _apply_opacity(logo_fitted, opacity)
        canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
        canvas.paste(logo_fitted, (x, y), mask=logo_fitted)
        out = Image.alpha_composite(base, canvas)
    else:
        # Legacy scale flow
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

    # Preserve DPI metadata if present
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
