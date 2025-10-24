from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance


# PUBLIC_INTERFACE
def overlay_logo(
    base_image_path: Path,
    logo_path: Path,
    output_path: Path,
    *,
    position: Tuple[int, int] = (10, 10),
    scale: float = 0.2,
    opacity: float = 0.9,
    max_logo_size_ratio: float = 0.4,
) -> None:
    """Overlay a logo image onto a base image with simple scaling and opacity control.

    Args:
        base_image_path: Path to the base image (e.g., engineering drawing raster).
        logo_path: Path to the logo image to overlay.
        output_path: Path where the output image will be saved.
        position: Top-left position where the logo will be placed.
        scale: Relative scale of the logo with respect to base image width.
        opacity: Opacity for the logo in range [0.0, 1.0].
        max_logo_size_ratio: Maximum ratio of base image width/height that logo may occupy.

    Notes:
        - Preserves alpha channel if present.
        - Ensures the logo does not exceed max_logo_size_ratio dimensions.
    """
    base = Image.open(base_image_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    # Compute target size
    base_w, base_h = base.size
    target_w = int(base_w * scale)
    aspect = logo.width / max(1, logo.height)
    target_h = int(target_w / aspect)

    # Constrain by max ratio
    max_w = int(base_w * max_logo_size_ratio)
    max_h = int(base_h * max_logo_size_ratio)
    target_w = min(target_w, max_w)
    target_h = min(target_h, max_h)

    if target_w <= 0 or target_h <= 0:
        target_w = max(1, target_w)
        target_h = max(1, target_h)

    logo_resized = logo.resize((target_w, target_h), resample=Image.LANCZOS)

    # Apply opacity
    if opacity < 1.0:
        alpha = logo_resized.split()[-1]
        enhancer = ImageEnhance.Brightness(alpha)
        alpha = enhancer.enhance(max(0.0, min(1.0, opacity)))
        logo_resized.putalpha(alpha)

    # Composite
    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.paste(logo_resized, position, mask=logo_resized)
    out = Image.alpha_composite(base, canvas)

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert back to RGB if output type doesn't support alpha (.jpg)
    fmt = output_path.suffix.lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        out = out.convert("RGB")
    out.save(output_path)
