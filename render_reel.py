#!/usr/bin/env python3
"""THIS KID BREATHES MUSIC — 12-second silent aligned reel renderer.

Save this file in the same folder as your project files. The script:
- creates a 1080x1920, 30 fps video exactly 12 seconds long;
- needs your background photograph, but does NOT need the reference video or audio;
- extracts `song poster 2.zip` when found beside the script or in Downloads;
- creates a deterministic, aligned 3-row overlapping poster deck (not random scatter);
- makes each newly arriving poster grow into place;
- slowly zooms the entire poster deck larger throughout the 12 seconds;
- keeps every poster visible after it appears;
- keeps the title above all posters;
- exports a silent MP4 so you can add music later.

The script creates `assets/posters_raw/` automatically. If only the new ZIP is
available, it will render the posters in that ZIP. To use additional existing posters,
place those original image files in `assets/posters_raw/` too.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# -----------------------------------------------------------------------------
# EDITABLE SETTINGS
# -----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
W, H, FPS = 1080, 1920, 30
DURATION = 12.0
NFRAMES = int(round(DURATION * FPS))  # 360 frames
FIRST_COVER_FRAME = 3                 # 0.10 seconds
LAST_COVER_FRAME = 342                # 11.40 seconds
POP_FRAMES = 6                        # 0.20 second arrival growth
POP_START_SCALE = 0.55                 # each new cover grows from 55% to 100%
ZOOM_START = 0.82                       # poster deck begins slightly smaller
ZOOM_END = 1.20                         # poster deck grows smoothly by the end
ZOOM_ANCHOR_Y = 1190                    # keep the zoom centered on the poster area
DECK_TOP = 800
DECK_BOTTOM = 1600

TITLE_LINES = ["THIS KID", "BREATHES MUSIC"]
TITLE_TOP = 72
TITLE_SIZE = 110
TITLE_LINE_GAP = 6

# Aligned horizontal poster deck. No random positions are used.
COVER_SIZE = 190
ROWS = 3
ROW_CENTERS_Y = [1020, 1200, 1380]
SIDE_MARGIN = 20
MAX_HORIZONTAL_STEP = 40.0

POSTER_DIR = BASE / "assets" / "posters_raw"
LAYOUT_PATH = BASE / "layout.json"
MANIFEST_PATH = BASE / "poster_manifest.json"
OUTPUT_PATH = BASE / "THIS_KID_BREATHES_MUSIC_GROWING_12S_SILENT.mp4"
PREVIEW_PATH = BASE / "preview_growing_collage.png"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# -----------------------------------------------------------------------------
# FILE DISCOVERY AND POSTER PREPARATION
# -----------------------------------------------------------------------------
def natural_key(path: Path):
    """Sort unnamed.jpg first, then unnamed (1).jpg, ... unnamed (50).jpg."""
    stem = path.stem.lower()
    if stem == "unnamed":
        return (0, 0, path.suffix.lower())
    match = re.fullmatch(r"unnamed \((\d+)\)", stem)
    if match:
        return (0, int(match.group(1)), path.suffix.lower())
    parts = re.split(r"(\d+)", path.name.lower())
    return (1, tuple((1, int(p)) if p.isdigit() else (0, p) for p in parts))


def candidate_search_dirs():
    dirs = [
        BASE,
        BASE / "assets" / "input",
        BASE / "input",
        BASE / "uploads",
        Path.home() / "Downloads",
    ]
    found = []
    for d in dirs:
        if d.is_dir() and d not in found:
            found.append(d)
    return found


def find_named_file(names: tuple[str, ...]) -> Path | None:
    """Find a known input filename in the project/input/Downloads locations."""
    dirs = candidate_search_dirs()
    # Prefer exact names in the project root, then known input folders.
    for d in dirs:
        for name in names:
            p = d / name
            if p.is_file():
                return p
    # Also look one or two directory levels below project root.
    for name in names:
        for parent in (BASE / "assets", BASE / "input", BASE / "uploads"):
            if parent.is_dir():
                matches = list(parent.rglob(name))
                if matches:
                    return matches[0]
    return None


def find_background() -> Path:
    p = find_named_file((
        "background.jpg", "background.jpeg", "background.png",
        "IMG_20261019_034214.jpg", "1000085515.jpg", "monument.jpg",
    ))
    if p:
        return p
    raise FileNotFoundError(
        "Could not find your background photo. Copy your monument photograph "
        "into the same folder as render_reel.py and name it background.jpg."
    )


def ensure_video_tools_available() -> None:
    """Confirm FFmpeg is available; this render intentionally has no audio."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and reopen VS Code, then try again."
        )


def extract_poster_zip_if_found() -> None:
    """Extract the additional poster ZIP without requiring manual folder setup."""
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    zip_candidates = [
        BASE / "song poster 2.zip",
        BASE / "song_poster_2.zip",
        Path.home() / "Downloads" / "song poster 2.zip",
        Path.home() / "Downloads" / "song_poster_2.zip",
    ]
    archive = next((p for p in zip_candidates if p.is_file()), None)
    if archive is None:
        return

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                src_name = Path(info.filename)
                if src_name.suffix.lower() not in IMAGE_EXTS:
                    continue
                # Flatten the supplied archive into the poster folder. These files
                # are user-supplied images; do not extract arbitrary non-image data.
                out = POSTER_DIR / src_name.name
                with zf.open(info, "r") as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        print(f"Extracted poster images from: {archive}")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"The poster archive is not a valid ZIP: {archive}") from exc


def poster_directories() -> list[Path]:
    candidates = [
        POSTER_DIR,
        BASE / "posters_raw",
        BASE / "posters",
        BASE / "song_posters",
        BASE / "covers",
        BASE / "assets" / "covers",
    ]
    result = []
    for p in candidates:
        if p.is_dir() and p not in result:
            result.append(p)
    return result


def get_poster_files() -> list[Path]:
    extract_poster_zip_if_found()
    candidates: list[Path] = []
    for folder in poster_directories():
        candidates.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    # Also allow the named poster images to have been extracted directly beside
    # this script, without accidentally using preview/contact-sheet images.
    candidates.extend(
        p for p in BASE.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and (p.stem.lower().startswith("unnamed") or p.stem.lower().startswith("poster_"))
    )

    unique: list[Path] = []
    seen_hashes: set[str] = set()
    for p in sorted(set(candidates), key=natural_key):
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique.append(p)

    if not unique:
        raise FileNotFoundError(
            "No album-cover images were found. Put song poster 2.zip beside "
            "render_reel.py, or place your poster JPG/PNG files in "
            "assets/posters_raw/. The script creates that folder automatically "
            "when it finds the ZIP."
        )

    if len(unique) < 51:
        print(
            f"WARNING: found {len(unique)} unique poster images, not 51. "
            "Your new ZIP contains 23 posters; the original 28 actual poster "
            "image files were not included in files.zip. Put those original "
            "images into assets/posters_raw/ if you want the full set of 51."
        )
    else:
        print(f"Found {len(unique)} unique poster images.")
    return unique

# -----------------------------------------------------------------------------
# DETERMINISTIC ALIGNED LAYOUT
# -----------------------------------------------------------------------------
def make_aligned_layout(files: list[Path]) -> list[dict]:
    """Create three clean horizontal rows that grow left-to-right in columns."""
    n = len(files)
    cols = math.ceil(n / ROWS)
    if cols <= 1:
        step_x = 0.0
    else:
        max_fit_step = (W - COVER_SIZE - 2 * SIDE_MARGIN) / (cols - 1)
        step_x = min(MAX_HORIZONTAL_STEP, max_fit_step)

    total_width = COVER_SIZE + step_x * max(0, cols - 1)
    first_cx = (W - total_width) / 2 + COVER_SIZE / 2

    layout = []
    # Column-major order: three covers stack vertically at each x-position,
    # then the deck grows to the right. Every coordinate is deterministic.
    for i, path in enumerate(files):
        col = i // ROWS
        row = i % ROWS
        layout.append({
            "order": i,
            "source_file": path.name,
            "column": col,
            "row": row,
            "cx": round(first_cx + col * step_x, 2),
            "cy": float(ROW_CENTERS_Y[row]),
            "size": COVER_SIZE,
            "angle": 0.0,
            "visible": 1.0,
            "salient_visible": 1.0,
            "max_overlap": 0.0,
        })
    with open(LAYOUT_PATH, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote aligned layout: {LAYOUT_PATH} ({n} posters)")
    return layout

# -----------------------------------------------------------------------------
# IMAGE COMPOSITION
# -----------------------------------------------------------------------------
def load_background(photo_path: Path) -> np.ndarray:
    im = Image.open(photo_path).convert("RGB")
    fitted = ImageOps.fit(
        im,
        (W, H),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return np.asarray(fitted, dtype=np.uint8).copy()


def locate_font() -> str | None:
    candidates = [
        BASE / "fonts" / "Anton-Regular.ttf",
        Path("C:/Windows/Fonts/impact.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def render_title() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build title RGB and alpha arrays for an upper strip of the canvas."""
    y0, y1 = 35, 330
    overlay = Image.new("RGBA", (W, y1 - y0), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_path = locate_font()
    font = None
    size = TITLE_SIZE
    while size >= 68:
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        widest = max(draw.textbbox((0, 0), line, font=font, stroke_width=3)[2] for line in TITLE_LINES)
        if widest <= W - 70:
            break
        size -= 2
    if font is None:
        font = ImageFont.load_default()

    # Soft shadow behind the white title.
    shadow = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    y = TITLE_TOP - y0 + 4
    for line in TITLE_LINES:
        bbox = sd.textbbox((0, 0), line, font=font, stroke_width=0)
        x = (W - (bbox[2] - bbox[0])) // 2 - bbox[0]
        sd.text((x + 2, y + 3), line, font=font, fill=(0, 0, 0, 175))
        y += font.getbbox("Hg")[3] - font.getbbox("Hg")[1] + TITLE_LINE_GAP
    shadow = shadow.filter(ImageFilter.GaussianBlur(4))
    overlay.alpha_composite(shadow)

    # Main title with a subtle black outline.
    y = TITLE_TOP - y0
    for line in TITLE_LINES:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
        x = (W - (bbox[2] - bbox[0])) // 2 - bbox[0]
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                  stroke_width=3, stroke_fill=(0, 0, 0, 230))
        y += font.getbbox("Hg")[3] - font.getbbox("Hg")[1] + TITLE_LINE_GAP

    rgba = np.asarray(overlay, dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return rgb, alpha, y0, y1


def make_sprite(src_rgb: np.ndarray, size: int, angle: float, cx: float, cy: float, scale: float = 1.0):
    """Create one sharp square cover with a soft drop shadow."""
    b = max(8, int(round(size * scale)))
    base = cv2.resize(src_rgb, (b, b), interpolation=cv2.INTER_AREA if b <= src_rgb.shape[0] else cv2.INTER_CUBIC)
    pad = 28
    side = b + 2 * pad
    x0 = int(round(cx - b / 2 - pad))
    y0 = int(round(cy - b / 2 - pad))
    M = cv2.getRotationMatrix2D(((b - 1) / 2, (b - 1) / 2), float(angle), 1.0)
    M[0, 2] += pad
    M[1, 2] += pad
    colour = cv2.warpAffine(base, M, (side, side), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ones = np.ones((b, b), np.float32)
    mask = cv2.warpAffine(ones, M, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    soft = cv2.GaussianBlur(np.roll(mask, 5, axis=0), (0, 0), 6) * 0.32
    contact = cv2.GaussianBlur(np.roll(mask, 2, axis=0), (0, 0), 1.8) * 0.16
    shadow = 1 - (1 - soft) * (1 - contact)
    return x0, y0, colour.astype(np.float32), mask, shadow


def paste_sprite(canvas: np.ndarray, sprite) -> None:
    x0, y0, colour, mask, shadow = sprite
    h, w = mask.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    roi = canvas[ya:yb, xa:xb].astype(np.float32)
    roi *= (1.0 - shadow[sl])[..., None]
    m = mask[sl][..., None]
    roi = roi * (1.0 - m) + colour[sl] * m
    canvas[ya:yb, xa:xb] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


def paste_sprite_rgba(canvas_rgba: np.ndarray, sprite, y_offset: int = 0) -> None:
    """Composite one poster/shadow sprite onto a transparent RGBA poster layer."""
    x0, y0, colour, mask, shadow = sprite
    y0 -= y_offset
    h, w = mask.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(canvas_rgba.shape[1], x0 + w), min(canvas_rgba.shape[0], y0 + h)
    if xa >= xb or ya >= yb:
        return

    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    dst = canvas_rgba[ya:yb, xa:xb].astype(np.float32)
    src_rgb = np.clip(colour[sl], 0, 255)
    src_a = np.clip(mask[sl] + shadow[sl] * (1.0 - mask[sl]), 0.0, 1.0)[..., None]
    dst_rgb = dst[..., :3]
    dst_a = dst[..., 3:4] / 255.0

    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb_premultiplied = src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)
    out_rgb = np.divide(
        out_rgb_premultiplied,
        np.maximum(out_a, 1e-6),
        out=np.zeros_like(out_rgb_premultiplied),
        where=out_a > 1e-6,
    )
    result = np.concatenate([out_rgb, out_a * 255.0], axis=2)
    canvas_rgba[ya:yb, xa:xb] = np.clip(result + 0.5, 0, 255).astype(np.uint8)


def scale_rgba_to_frame(deck_crop: np.ndarray, zoom: float) -> np.ndarray:
    """Smoothly zoom just the poster deck, leaving the background photograph still."""
    crop_h, crop_w = deck_crop.shape[:2]
    new_w = max(1, int(round(crop_w * zoom)))
    new_h = max(1, int(round(crop_h * zoom)))

    rgba = deck_crop.astype(np.float32)
    alpha = rgba[:, :, 3] / 255.0
    # Resize premultiplied RGB to avoid dark fringes around transparent edges.
    premul_rgb = rgba[:, :, :3] * alpha[:, :, None]
    resized_a = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_premul = cv2.resize(premul_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_rgb = np.divide(
        resized_premul,
        np.maximum(resized_a[:, :, None], 1e-6),
        out=np.zeros_like(resized_premul),
        where=resized_a[:, :, None] > 1e-6,
    )
    resized = np.concatenate(
        [resized_rgb, resized_a[:, :, None] * 255.0], axis=2
    )
    resized = np.clip(resized + 0.5, 0, 255).astype(np.uint8)

    full = np.zeros((H, W, 4), dtype=np.uint8)
    x0 = int(round((W - new_w) / 2))
    y0 = int(round(ZOOM_ANCHOR_Y - new_h / 2))
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + new_w), min(H, y0 + new_h)
    if xa < xb and ya < yb:
        full[ya:yb, xa:xb] = resized[
            ya - y0:yb - y0,
            xa - x0:xb - x0
        ]
    return full


def composite_rgba_over_rgb(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Place a transparent poster layer over the fixed RGB background."""
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    out = (
        background.astype(np.float32) * (1.0 - alpha)
        + overlay[:, :, :3].astype(np.float32) * alpha
    )
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def apply_title(frame: np.ndarray, title) -> None:
    rgb, alpha, y0, y1 = title
    roi = frame[y0:y1].astype(np.float32)
    roi = roi * (1.0 - alpha) + rgb * alpha
    frame[y0:y1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3

# -----------------------------------------------------------------------------
# RENDER
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_video_tools_available()
    photo_path = find_background()
    files = get_poster_files()

    if len(files) > LAST_COVER_FRAME - FIRST_COVER_FRAME + 1:
        raise ValueError("There are too many posters to give each a unique arrival frame in 12 seconds.")

    layout = make_aligned_layout(files)
    posters = []
    for p in files:
        with Image.open(p) as im:
            # Album art is expected to be square. Crop-to-square only if necessary;
            # no stretching is performed.
            square = ImageOps.fit(im.convert("RGB"), (600, 600), method=Image.Resampling.LANCZOS)
            posters.append(np.asarray(square, dtype=np.uint8).copy())

    print(f"Background: {photo_path}")
    print("Audio: none (you will add it after editing)")
    print(f"Posters to render: {len(files)}")
    print(f"Output: {OUTPUT_PATH}")

    bg = load_background(photo_path)
    title = render_title()

    # Build cumulative TRANSPARENT poster layers. The background image is not
    # part of these layers, so only the posters zoom while the photograph stays still.
    deck_h = DECK_BOTTOM - DECK_TOP
    poster_layer = np.zeros((H, W, 4), dtype=np.uint8)
    cumulative = []
    for poster, d in zip(posters, layout):
        sprite = make_sprite(poster, d["size"], d["angle"], d["cx"], d["cy"])
        paste_sprite_rgba(poster_layer, sprite)
        cumulative.append(poster_layer[DECK_TOP:DECK_BOTTOM].copy())

    if len(files) == 1:
        appear = [FIRST_COVER_FRAME]
    else:
        appear = np.rint(
            np.linspace(FIRST_COVER_FRAME, LAST_COVER_FRAME, num=len(files))
        ).astype(int).tolist()
    if len(set(appear)) != len(appear):
        raise ValueError("The calculated poster arrival frames are not unique. Reduce poster count or adjust timing.")
    last_anim_frame = appear[-1] + POP_FRAMES - 1

    empty_deck = np.zeros((deck_h, W, 4), dtype=np.uint8)

    def frame_at(frame_number: int) -> np.ndarray:
        # Count only posters whose entrance animation has finished.
        n_done = sum(1 for a in appear if frame_number >= a + POP_FRAMES)
        deck = empty_deck.copy() if n_done == 0 else cumulative[n_done - 1].copy()

        # Animate covers arriving now: each grows from 55% to its final deck size.
        for i, a in enumerate(appear):
            if a <= frame_number < a + POP_FRAMES:
                k = frame_number - a
                local_t = k / max(1, POP_FRAMES - 1)
                entrance_scale = POP_START_SCALE + (1.0 - POP_START_SCALE) * ease_out_cubic(local_t)
                d = layout[i]
                sprite = make_sprite(
                    posters[i], d["size"], d["angle"], d["cx"], d["cy"],
                    scale=entrance_scale,
                )
                paste_sprite_rgba(deck, sprite, y_offset=DECK_TOP)

        # The complete deck slowly grows over the entire reel.
        progress = frame_number / max(1, NFRAMES - 1)
        zoom = ZOOM_START + (ZOOM_END - ZOOM_START) * progress
        zoomed_deck = scale_rgba_to_frame(deck, zoom)

        # Keep the photo stationary, composite the growing poster deck, then draw
        # the title on the very top so album covers can never obscure it.
        frame = composite_rgba_over_rgb(bg, zoomed_deck)
        apply_title(frame, title)
        return frame

    # Silent video only. Music can be added later in Canva or another editor.
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-stats",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-map", "0:v:0",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-level", "4.2",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "60", "-bf", "2",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-an", "-t", f"{DURATION:.2f}", "-movflags", "+faststart", str(OUTPUT_PATH),
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(NFRAMES):
            current = frame_at(f)
            process.stdin.write(current.tobytes())
        process.stdin.close()
    except BrokenPipeError as exc:
        process.stdin.close()
        rc = process.wait()
        raise RuntimeError(
            f"FFmpeg stopped while rendering (exit code {rc}). Check that FFmpeg is installed "
            "and that the audio input is valid."
        ) from exc

    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"FFmpeg exited with code {rc}. The output may be incomplete.")

    # Final frame preview and timing manifest.
    final_frame = frame_at(NFRAMES - 1)
    Image.fromarray(final_frame).save(PREVIEW_PATH)
    manifest = []
    for i, (p, d) in enumerate(zip(files, layout)):
        manifest.append({
            "arrival_order": i + 1,
            "appears_at_s": round(appear[i] / FPS, 3),
            "appear_frame": int(appear[i]),
            "source_file": p.name,
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            "source_px": Image.open(p).size,
            "display_px": d["size"],
            "center_xy": [d["cx"], d["cy"]],
            "rotation_deg": d["angle"],
            "row": d["row"],
            "column": d["column"],
        })
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nSUCCESS")
    print(f"Rendered: {OUTPUT_PATH}")
    print(f"Duration: {DURATION:.2f} seconds | Resolution: {W}x{H} | FPS: {FPS}")
    print("Audio: none (silent video as requested)")
    print(f"Poster count used: {len(files)}")
    print(f"Poster deck zoom: {ZOOM_START:.2f}x to {ZOOM_END:.2f}x")
    print(f"New-cover entrance: {POP_START_SCALE:.2f}x to 1.00x")
    print(f"Final collage preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
