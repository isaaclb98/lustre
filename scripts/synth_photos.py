"""
scripts/synth_photos.py — generate a rich library of synthetic photos.

Used by:
  - dev_server.py --demo-data (small demo set)
  - scripts/seed-data.sh (large seed for dev Qdrant)

The photos are drawn entirely with PIL — no ML, no external assets.
Each photo is a JPEG that resembles a real photo of its subject via
gradient skies, layered silhouettes, and small detail shapes. They
are not pixel-perfect, but they're unmistakably NOT the flat
SVG-style mocks the rest of the testbed uses.

Output: <output-dir>/demo/<N>.jpg for N in [0, count).
         A sidecar index.json listing every photo and its label.

Run:
    .venv-test/bin/python scripts/synth_photos.py --out /tmp/is-synth --count 80
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# ---------- subjects ----------
# Each subject is a list of render params. We vary colour palettes,
# weather, time of day, and composition across variants so the seed
# library feels like a real photo collection, not 20 copies of one
# thing.

@dataclass(frozen=True)
class Palette:
    sky_top: tuple[int, int, int]
    sky_bottom: tuple[int, int, int]
    horizon_far: tuple[int, int, int]
    ground: tuple[int, int, int]
    accent: tuple[int, int, int]


@dataclass(frozen=True)
class Subject:
    name: str
    palettes: Sequence[Palette]
    canvases: Sequence[tuple[int, int]]  # multiple aspect ratios per subject
    family: str  # 'landscape' | 'urban' | 'wildlife' | 'portrait' | 'still'


# Aspect ratios we exercise. The grid renders 1:1 (uniform squares),
# but the underlying photo bytes vary across these so:
#   - infinite scroll gets a real test (lots of tiles, lots of
#     bytes to load, lots of reflows)
#   - the standalone /photo page and the lightbox surface render
#     at the actual aspect ratio (no crop), so variety is visible
#     there
#   - backend IndexDB stores width/height per point, so any
#     downstream feature that respects the ratio gets exercised
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "panoramic":   (2400, 800),    # 3:1 — landscape banner
    "wide":        (1920, 1080),   # 16:9 — modern photo
    "landscape":   (1600, 1200),   # 4:3 — classic
    "square":      (1200, 1200),   # 1:1
    "portrait":    (900, 1200),    # 3:4
    "tall":        (720, 1280),    # 9:16 — phone portrait
    "ultra_tall":  (540, 1440),    # 9:24 — cinematic poster
}


def _cv(name: str) -> list[tuple[int, int]]:
    """Resolve a single aspect ratio name into its canvas size."""
    return [ASPECT_RATIOS[name]]


def _all() -> list[tuple[int, int]]:
    return list(ASPECT_RATIOS.values())


SUBJECTS: list[Subject] = [
    # ---- landscapes ----
    Subject("mountain", [
        Palette((48, 64, 96), (148, 168, 192), (60, 70, 84), (35, 38, 42), (220, 200, 120)),
        Palette((20, 24, 36), (88, 72, 96), (50, 40, 60), (24, 22, 26), (255, 200, 100)),
        Palette((180, 130, 90), (240, 200, 150), (120, 100, 80), (70, 60, 50), (255, 240, 220)),
        Palette((20, 40, 70), (110, 150, 200), (40, 60, 100), (15, 25, 35), (240, 220, 150)),
    ], _all(), "landscape"),
    Subject("beach", [
        Palette((60, 130, 180), (180, 210, 230), (110, 150, 170), (220, 200, 160), (245, 220, 130)),
        Palette((220, 130, 100), (250, 210, 170), (180, 120, 90), (220, 200, 170), (255, 240, 200)),
        Palette((20, 30, 60), (60, 90, 130), (40, 50, 80), (15, 18, 28), (200, 180, 100)),
        Palette((180, 200, 220), (240, 240, 240), (140, 160, 180), (200, 180, 150), (255, 230, 150)),
    ], _all(), "landscape"),
    Subject("forest", [
        Palette((50, 70, 50), (110, 130, 90), (35, 55, 35), (60, 80, 50), (200, 180, 80)),
        Palette((30, 30, 30), (70, 70, 60), (20, 25, 22), (40, 50, 40), (240, 200, 90)),
        Palette((200, 180, 130), (240, 220, 180), (140, 120, 80), (100, 80, 60), (255, 240, 180)),
        Palette((60, 90, 110), (130, 160, 170), (50, 75, 90), (30, 50, 60), (200, 180, 120)),
    ], _all(), "landscape"),
    Subject("desert", [
        Palette((220, 150, 100), (250, 200, 150), (180, 120, 80), (200, 160, 110), (255, 240, 200)),
        Palette((150, 100, 80), (220, 170, 130), (130, 90, 70), (170, 130, 90), (255, 220, 170)),
        Palette((60, 30, 30), (180, 80, 60), (90, 50, 40), (120, 70, 50), (240, 140, 80)),
    ], _all(), "landscape"),
    Subject("snow", [
        Palette((180, 200, 220), (240, 245, 250), (200, 210, 220), (245, 248, 250), (255, 255, 255)),
        Palette((60, 80, 110), (160, 180, 200), (90, 100, 120), (180, 190, 200), (220, 230, 240)),
        Palette((200, 220, 240), (255, 255, 255), (220, 230, 240), (250, 252, 255), (255, 255, 255)),
    ], _all(), "landscape"),
    Subject("autumn", [
        Palette((170, 80, 50), (240, 160, 80), (130, 70, 40), (90, 50, 30), (250, 220, 100)),
        Palette((100, 60, 40), (200, 130, 80), (80, 50, 30), (70, 40, 25), (240, 180, 80)),
        Palette((60, 80, 100), (130, 150, 160), (50, 70, 90), (40, 60, 80), (200, 180, 140)),
    ], _all(), "landscape"),
    Subject("lake", [
        Palette((60, 130, 160), (180, 210, 220), (100, 140, 160), (40, 90, 110), (200, 220, 200)),
        Palette((220, 150, 100), (250, 200, 160), (180, 120, 80), (200, 180, 160), (240, 220, 180)),
        Palette((30, 50, 70), (90, 110, 130), (50, 70, 90), (20, 40, 60), (180, 180, 150)),
    ], _all(), "landscape"),
    Subject("waterfall", [
        Palette((40, 100, 110), (180, 220, 220), (60, 120, 130), (30, 80, 90), (200, 220, 200)),
        Palette((60, 80, 50), (140, 160, 130), (50, 70, 50), (40, 60, 40), (180, 180, 150)),
    ], _all(), "landscape"),

    # ---- urban ----
    Subject("city_night", [
        Palette((15, 18, 32), (40, 50, 80), (10, 15, 25), (20, 25, 40), (250, 220, 80)),
        Palette((20, 30, 60), (80, 60, 110), (15, 20, 45), (25, 30, 50), (240, 100, 140)),
        Palette((10, 12, 25), (40, 30, 50), (8, 10, 20), (15, 18, 30), (200, 220, 255)),
    ], _all(), "urban"),
    Subject("city_day", [
        Palette((150, 200, 220), (200, 230, 240), (120, 150, 170), (180, 180, 170), (240, 230, 200)),
        Palette((180, 180, 200), (220, 220, 230), (150, 150, 170), (170, 170, 180), (255, 240, 200)),
    ], _all(), "urban"),
    Subject("street", [
        Palette((60, 70, 90), (140, 140, 160), (50, 60, 75), (80, 80, 90), (220, 180, 80)),
        Palette((40, 50, 70), (110, 110, 130), (35, 45, 60), (60, 70, 80), (240, 200, 100)),
    ], _all(), "urban"),

    # ---- wildlife ----
    Subject("fox", [
        Palette((180, 90, 50), (240, 160, 90), (130, 60, 30), (60, 40, 30), (255, 220, 100)),
        Palette((60, 30, 20), (130, 60, 30), (40, 25, 18), (30, 20, 15), (220, 180, 80)),
    ], _all(), "wildlife"),
    Subject("owl", [
        Palette((60, 50, 80), (120, 110, 140), (50, 45, 70), (40, 35, 60), (200, 180, 140)),
        Palette((100, 90, 70), (160, 150, 130), (80, 70, 55), (70, 60, 50), (220, 200, 170)),
    ], _all(), "wildlife"),
    Subject("butterfly", [
        Palette((220, 130, 80), (240, 200, 100), (200, 100, 60), (60, 150, 80), (240, 240, 220)),
        Palette((100, 100, 200), (200, 150, 220), (60, 60, 140), (40, 90, 60), (240, 220, 120)),
    ], _all(), "wildlife"),
    Subject("deer", [
        Palette((60, 100, 50), (140, 160, 100), (50, 80, 40), (40, 60, 30), (180, 150, 80)),
        Palette((100, 70, 50), (180, 130, 90), (80, 50, 30), (60, 40, 25), (220, 200, 140)),
    ], _all(), "wildlife"),

    # ---- portraits ----
    Subject("silhouette", [
        Palette((220, 100, 70), (250, 180, 100), (180, 70, 50), (40, 30, 30), (255, 230, 180)),
        Palette((40, 60, 100), (110, 140, 200), (30, 50, 90), (20, 20, 30), (220, 200, 100)),
        Palette((100, 50, 130), (200, 100, 180), (80, 40, 100), (30, 25, 40), (255, 220, 150)),
    ], _all(), "portrait"),

    # ---- still life ----
    Subject("flowers", [
        Palette((80, 150, 90), (180, 220, 150), (60, 120, 70), (50, 80, 50), (240, 100, 140)),
        Palette((220, 130, 180), (250, 200, 220), (180, 90, 140), (80, 100, 60), (240, 220, 100)),
    ], _all(), "still"),
    Subject("cafe", [
        Palette((180, 140, 100), (220, 190, 150), (150, 110, 80), (100, 80, 60), (240, 220, 180)),
        Palette((80, 60, 50), (140, 110, 90), (60, 45, 35), (50, 40, 30), (200, 170, 130)),
    ], _all(), "still"),
    Subject("neon", [
        Palette((15, 10, 30), (60, 30, 90), (10, 8, 20), (20, 15, 40), (240, 80, 160)),
        Palette((10, 20, 30), (30, 60, 100), (8, 12, 20), (15, 20, 35), (100, 200, 240)),
    ], _all(), "urban"),
    Subject("abstract", [
        Palette((200, 100, 60), (240, 180, 100), (160, 80, 50), (60, 40, 80), (220, 200, 240)),
        Palette((60, 100, 160), (140, 180, 220), (40, 80, 130), (80, 60, 100), (240, 220, 100)),
        Palette((180, 80, 120), (220, 160, 200), (140, 60, 90), (60, 80, 60), (200, 240, 220)),
    ], _all(), "still"),
]


# ---------- helpers ----------

def _vert_gradient(img: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    w, h = img.size
    pixels = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(w):
            pixels[x, y] = (r, g, b)


def _horiz_band(img: Image.Image, y_start: int, y_end: int, color: tuple[int, int, int]) -> None:
    w = img.size[0]
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, y_start, w, y_end), fill=color)


def _h(*parts) -> int:
    """Hash any tuple-of-mixed-types into a stable 32-bit int seed.

    `random.Random` (Python 3, version 2 seeding) refuses tuples, so
    we hash by hand. `str(...)` then `hash()` would be non-stable
    per process; we use built-in hashlib.
    """
    import hashlib

    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, tuple):
            for x in p:
                h.update(str(x).encode())
                h.update(b"|")
        else:
            h.update(str(p).encode())
            h.update(b"|")
    return int.from_bytes(h.digest()[:4], "big")


def _mountains(draw: ImageDraw.ImageDraw, w: int, h: int, base_y: int, color: tuple[int, int, int], peaks: int = 6) -> None:
    rng = random.Random(_h("mountains", w, h, base_y, color[0], peaks))
    cx = 0
    for i in range(peaks):
        peak_w = rng.randint(int(w / peaks * 0.8), int(w / peaks * 1.4))
        peak_h = rng.randint(int(h * 0.18), int(h * 0.40))
        pts = [
            (cx, base_y),
            (cx + peak_w // 2, base_y - peak_h),
            (cx + peak_w, base_y),
        ]
        draw.polygon(pts, fill=color)
        cx += peak_w - rng.randint(0, int(peak_w * 0.3))


def _trees(draw: ImageDraw.ImageDraw, w: int, h: int, base_y: int, color: tuple[int, int, int], count: int = 16) -> None:
    rng = random.Random(_h("trees", w, h, base_y, color[0], count))
    for _ in range(count):
        tx = rng.randint(0, w)
        th = rng.randint(int(h * 0.10), int(h * 0.22))
        tw = rng.randint(int(th * 0.3), int(th * 0.5))
        # trunk
        draw.rectangle((tx - tw // 6, base_y - th // 4, tx + tw // 6, base_y), fill=(int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6)))
        # canopy as a triangle
        draw.polygon([(tx - tw, base_y - th // 4), (tx, base_y - th), (tx + tw, base_y - th // 4)], fill=color)


def _clouds(draw: ImageDraw.ImageDraw, w: int, h: int, color: tuple[int, int, int], count: int = 4) -> None:
    rng = random.Random(_h("clouds", w, h, color[0], count))
    for _ in range(count):
        cx = rng.randint(0, w)
        cy = rng.randint(int(h * 0.05), int(h * 0.30))
        cw = rng.randint(int(w * 0.06), int(w * 0.14))
        # 3 ellipses hugging the centre
        for ox in (-cw // 3, 0, cw // 3):
            draw.ellipse((cx + ox - cw // 3, cy - cw // 6, cx + ox + cw // 3, cy + cw // 6), fill=color)


def _stars(draw: ImageDraw.ImageDraw, w: int, h: int, count: int = 80) -> None:
    rng = random.Random(_h("stars", w, h, count))
    for _ in range(count):
        sx = rng.randint(0, w)
        sy = rng.randint(0, int(h * 0.6))
        sr = rng.choice([0, 0, 1, 1, 2])
        draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=(255, 255, 220))


def _sun(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color: tuple[int, int, int], glow: int = 80) -> None:
    # Outer glow
    for i in range(glow, 0, -8):
        a = max(20, int(255 * (i / glow)))
        rgba = color + (a,)
        draw.ellipse((cx - r - i, cy - r - i, cx + r + i, cy + r + i), fill=rgba)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)


def _building_windows(draw: ImageDraw.ImageDraw, bx: int, by: int, bw: int, bh: int, window_color=(255, 220, 130)) -> None:
    # 5x8 grid of windows
    rows = 8
    cols = 5
    ww = bw // (cols * 2)
    hh = bh // (rows * 2)
    for r in range(rows):
        for c in range(cols):
            wx = bx + (c * 2 + 1) * (bw // (cols * 2)) - ww // 2
            wy = by + (r * 2 + 1) * (bh // (rows * 2)) - hh // 2
            # 60% lit
            if random.random() < 0.6:
                draw.rectangle((wx, wy, wx + ww, wy + hh), fill=window_color)


def _city_skyline(draw: ImageDraw.ImageDraw, w: int, h: int, base_y: int, color: tuple[int, int, int]) -> None:
    rng = random.Random(_h("skyline", w, h, color[0]))
    x = 0
    while x < w:
        bw = rng.randint(int(w * 0.05), int(w * 0.12))
        bh = rng.randint(int(h * 0.30), int(h * 0.75))
        # darker building silhouette
        draw.rectangle((x, base_y - bh, x + bw, base_y), fill=color)
        # a couple of windows
        _building_windows(draw, x, base_y - bh, bw, bh)
        x += bw - rng.randint(0, int(w * 0.01))


def _water_reflection(img: Image.Image, y_start: int) -> None:
    """Mirror-and-shake the top portion into the bottom for a
    sketchy water reflection."""
    w, h = img.size
    band_h = (h - y_start) // 2
    src = img.crop((0, y_start, w, y_start + band_h))
    flip = src.transpose(Image.FLIP_TOP_BOTTOM)
    flipped = flip.resize((w, band_h))
    # shift + a little blur for ripple
    flip_blur = flipped.filter(ImageFilter.GaussianBlur(2))
    img.paste(flip_blur, (0, y_start + band_h))


# ---------- renderers per family ----------

def render_landscape(img: Image.Image, palette: Palette, rng: random.Random) -> None:
    w, h = img.size
    _vert_gradient(img, palette.sky_top, palette.sky_bottom)
    horizon = int(h * rng.uniform(0.55, 0.70))
    _horiz_band(img, horizon, h, palette.ground)
    if rng.random() < 0.95:
        _mountains(img.draw if hasattr(img, "draw") else ImageDraw.Draw(img), w, h, horizon, palette.horizon_far,
                   peaks=rng.randint(3, 7))
    if rng.random() < 0.7:
        _clouds(ImageDraw.Draw(img), w, h,
                tuple(min(255, c + 60) for c in palette.sky_bottom),
                count=rng.randint(2, 5))
    if rng.random() < 0.6:
        # sun or moon
        sun_y = rng.randint(int(h * 0.10), int(h * 0.35))
        sun_x = rng.randint(int(w * 0.10), int(w * 0.90))
        _sun(ImageDraw.Draw(img), sun_x, sun_y, rng.randint(20, 60), palette.accent)
    if rng.random() < 0.5:
        # water reflection below horizon
        _water_reflection(img, horizon)


def render_forest(img: Image.Image, palette: Palette, rng: random.Random) -> None:
    w, h = img.size
    _vert_gradient(img, palette.sky_top, palette.sky_bottom)
    horizon = int(h * rng.uniform(0.55, 0.70))
    _horiz_band(img, horizon, h, palette.ground)
    # distant haze
    ImageDraw.Draw(img).rectangle((0, horizon - int(h * 0.05), w, horizon), fill=tuple(min(255, c + 30) for c in palette.horizon_far))
    _trees(ImageDraw.Draw(img), w, h, h, palette.horizon_far, count=rng.randint(8, 18))


def render_urban(img: Image.Image, palette: Palette, rng: random.Random) -> None:
    w, h = img.size
    _vert_gradient(img, palette.sky_top, palette.sky_bottom)
    base_y = int(h * 0.78)
    _city_skyline(ImageDraw.Draw(img), w, h, base_y, palette.horizon_far)
    # ground
    _horiz_band(img, base_y, h, palette.ground)
    if rng.random() < 0.4:
        _stars(ImageDraw.Draw(img), w, h, count=rng.randint(20, 80))


def render_wildlife(img: Image.Image, palette: Palette, rng: random.Random) -> None:
    w, h = img.size
    _vert_gradient(img, palette.sky_top, palette.sky_bottom)
    horizon = int(h * rng.uniform(0.55, 0.70))
    _horiz_band(img, horizon, h, palette.ground)
    _trees(ImageDraw.Draw(img), w, h, h, palette.horizon_far, count=4)
    # a stylised animal silhouette in the centre
    cx, cy = w // 2, int(horizon * 0.95)
    body_w = int(w * 0.22)
    body_h = int(h * 0.22)
    ImageDraw.Draw(img).ellipse(
        (cx - body_w, cy - body_h, cx + body_w, cy + body_h // 2),
        fill=palette.accent
    )
    ImageDraw.Draw(img).ellipse(
        (cx + body_w - body_w // 4, cy - body_h * 3 // 2, cx + body_w + body_w // 3, cy - body_h // 2),
        fill=palette.accent
    )


def render_portrait(img: Image.Image, palette: Palette, rng: random.Random) -> None:
    w, h = img.size
    _vert_gradient(img, palette.sky_top, palette.sky_bottom)
    # silhouetted head + shoulders centred
    cx = w // 2
    head_r = int(min(w, h) * 0.20)
    head_cy = int(h * 0.40)
    ImageDraw.Draw(img).ellipse(
        (cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r),
        fill=palette.horizon_far
    )
    # shoulders extending below frame
    ImageDraw.Draw(img).polygon([
        (cx - int(w * 0.35), h),
        (cx - int(w * 0.25), int(h * 0.65)),
        (cx + int(w * 0.25), int(h * 0.65)),
        (cx + int(w * 0.35), h),
    ], fill=palette.horizon_far)


def render_still(img: Image.Image, palette: Palette, rng: random.Random) -> None:
    w, h = img.size
    # table-top gradient + scattered accent circles
    _vert_gradient(img, palette.sky_top, palette.sky_bottom)
    _horiz_band(img, int(h * 0.75), h, palette.ground)
    for _ in range(rng.randint(4, 8)):
        ax = rng.randint(int(w * 0.10), int(w * 0.90))
        ay = rng.randint(int(h * 0.30), int(h * 0.70))
        ar = rng.randint(int(min(w, h) * 0.04), int(min(w, h) * 0.10))
        ImageDraw.Draw(img).ellipse(
            (ax - ar, ay - ar, ax + ar, ay + ar),
            fill=palette.accent
        )


RENDERERS = {
    "landscape": render_landscape,
    "urban": render_urban,
    "wildlife": render_wildlife,
    "portrait": render_portrait,
    "still": render_still,
    "forest": render_forest,
}


# ---------- public API ----------

@dataclass
class SynthResult:
    image_paths: list[Path]
    index: list[dict]


def generate(
    out_dir: Path,
    count: int = 200,
    *,
    seed: int = 0xC0FFEE,
    prefix: str = "synth",
) -> SynthResult:
    """Generate `count` synthetic JPEGs into `<out_dir>/`.

    Subject order cycles through SUBJECTS, then within each
    subject we cycle through its canvases (aspect ratios). So a
    200-photo set with 20 subjects × 7 aspect ratios = 140
    unique (subject, aspect) pairs, with a few repeats to reach
    `count`. The seeded RNG is stable across runs.

    The 7 aspect ratios per subject are: panoramic (3:1), wide
    (16:9), landscape (4:3), square (1:1), portrait (3:4),
    tall (9:16), ultra_tall (9:24). The grid renders all as 1:1
    for visual uniformity, but the bytes underneath carry the
    shape so /photo/[id], lightbox, and any future masonry-style
    view can render correctly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    paths: list[Path] = []
    index: list[dict] = []

    n_subjects = len(SUBJECTS)

    # The ratio cycle has to reach every ratio even when the set is
    # SMALLER than the subject count.
    #
    # The ratio index used to be `(i // n_subjects)`, which advances
    # only once per full pass through the subjects. With 20 subjects
    # and a 20-photo set, every photo landed on ratio 0 — the whole
    # set was 3:1 panoramic.
    #
    # That is invisible in a square grid, where every tile is a
    # square anyway. It is NOT invisible in a justified layout: a
    # row of 3:1 photos fits exactly ONE photo per row, so the
    # layout degenerates to a single-column strip and every
    # row-based assertion (a row's tiles summing to its width, a
    # row holding more than one photo) fails for a reason that has
    # nothing to do with the layout.
    #
    # So the ratio advances per photo whenever the set is too small
    # to reach every ratio by cycling the subjects, and the set is
    # always diverse.
    # The ratio index is the photo's position in the sequence of
    # SUBJECT x RATIO pairs, walked so that the ratios advance even
    # when the set is SMALLER than the subject count.
    #
    # It used to be `(i // n_subjects)`, which advances only once
    # per full pass through the subjects. With 20 subjects and a
    # 20-photo set, every photo landed on ratio 0 — the whole set
    # was 3:1 panoramic.
    #
    # That is invisible in a square grid, where every tile is a
    # square anyway. It is NOT invisible in a justified layout: a
    # row of 3:1 photos fits exactly ONE photo per row, so the
    # layout degenerates to a single-column strip and every
    # row-based assertion (a row's tiles summing to its width, a
    # row holding more than one photo) fails for a reason that has
    # nothing to do with the layout.
    #
    # The ratio now advances once per `n_subjects / n_ratios`
    # photos, so a small set still sees every ratio.
    n_ratios = len(ASPECT_RATIOS)
    ratio_stride = max(1, n_subjects // n_ratios)

    for i in range(count):
        subject = SUBJECTS[i % n_subjects]
        # Within each subject, cycle through ratios as well.
        ratio_idx = (i // ratio_stride) % len(subject.canvases)
        canvas = subject.canvases[ratio_idx]
        palette = subject.palettes[rng.randrange(len(subject.palettes))]

        sub_rng = random.Random(_h("sub", seed, subject.name, palette.sky_top, i, canvas))
        img = Image.new("RGB", canvas, (0, 0, 0))
        RENDERERS[subject.family if subject.family != "forest" else "forest"](
            img, palette, sub_rng
        )

        ratio_name = next(
            (k for k, v in ASPECT_RATIOS.items() if v == canvas),
            "unknown",
        )
        path = out_dir / f"{prefix}_{i:04d}.jpg"
        img.save(path, "JPEG", quality=85)
        paths.append(path)
        index.append({
            "id": f"{prefix}_{i:04d}",
            "path": str(path),
            "subject": subject.name,
            "family": subject.family,
            "aspect_ratio": ratio_name,
            "width": canvas[0],
            "height": canvas[1],
        })

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return SynthResult(paths, index)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--count",
        type=int,
        default=200,
        help="total photo count (default: 200 — enough to exercise infinite scroll)",
    )
    p.add_argument("--prefix", default="synth")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    args = p.parse_args()

    res = generate(args.out, count=args.count, prefix=args.prefix, seed=args.seed)
    print(f"wrote {len(res.image_paths)} photos into {args.out}/")
    print(f"index: {args.out}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
