from pathlib import Path

from PIL import Image

from src.utils.image_utils import place_logo_in_box


# PUBLIC_INTERFACE
def test_logo_placement_within_detection_tmp(tmp_path: Path):
    """Ensure placement stays within the detection rectangle after padding and fit."""
    # Create a base image 800x600 white
    base = Image.new("RGB", (800, 600), color=(255, 255, 255))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    # Create a logo 300x100
    logo = Image.new("RGBA", (300, 100), color=(0, 0, 0, 255))
    logo_path = tmp_path / "logo.png"
    logo.save(logo_path)

    # Detection rect at (100, 120) size 200x150
    det_box = (100, 120, 200, 150)

    out_path = tmp_path / "out.png"
    placement = place_logo_in_box(
        base_image_path=base_path,
        logo_path=logo_path,
        output_path=out_path,
        box=det_box,
        opacity=0.9,
        fit_mode="contain",
        padding_pct=0.10,  # 10%
        debug_preview_path=None,
    )

    # Compute rectangles
    dx, dy, dw, dh = det_box
    px, py = placement["placed_x"], placement["placed_y"]
    pw, ph = placement.get("width", placement["logo_w"]), placement.get("height", placement["logo_h"])

    # Check clamping
    assert px >= dx and py >= dy
    assert px + pw <= dx + dw + 0.1
    assert py + ph <= dy + dh + 0.1

    # Ensure positive sizes
    assert pw > 0 and ph > 0
