#!/usr/bin/env python3
"""Regenerate LocalBooru Android launcher icons from the brand photo-frame mark.

The brand (assets/logo.svg) is a dark rounded square (#1e1e1e) with a neon-green
(#00FFA3) picture-frame containing a sun and mountains. We rasterize that simple
geometry directly with Pillow (supersampled for crisp edges) and emit the mipmap
PNGs the Android app actually references (android:icon="@mipmap/ic_launcher"),
plus refresh the (currently unreferenced) adaptive-foreground set so no stock
Tauri placeholder icon remains in the res tree.

Usage: python3 scripts/gen-android-icons.py [--staging DIR] [--write]
Without --write, outputs land under a staging dir for visual verification.
"""
import argparse
import os
from PIL import Image, ImageDraw

# Brand palette (matches assets/logo.svg)
BG      = (30, 30, 30, 255)        # #1e1e1e
FRAME   = (40, 40, 40, 255)        # #282828
NEON    = (0, 255, 163, 255)       # #00FFA3

SS = 6  # supersample factor for smooth edges


def _draw_brand(draw, size, glyph_only, margin_frac):
    """Draw the brand on a `size`-square draw canvas.

    glyph_only: draw only the neon picture-frame mark (no dark outer square).
    margin_frac: fraction of `size` kept as empty margin around the square.
    """
    sq = size * (1 - 2 * margin_frac)          # dark-square side length
    m  = size * margin_frac                     # square top-left offset
    u  = sq / 60.0                              # one SVG unit in pixels

    if not glyph_only:
        # Outer dark rounded square, rx=12/60 of the square side.
        draw.rounded_rectangle([m, m, m + sq, m + sq], radius=12 * u, fill=BG)

    # Picture frame (inset 8 units from square origin), rx=6/60.
    fx = m + 8 * u
    fy = m + 8 * u
    fw = 44 * u
    draw.rounded_rectangle([fx, fy, fx + fw, fy + fw], radius=6 * u,
                           fill=FRAME, outline=NEON, width=int(2.5 * u))

    # Sun.
    cx, cy = m + 20 * u, m + 20 * u
    r = 6 * u
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NEON)

    # Mountains path in SVG units -> pixels (relative to square origin).
    pts = [(8, 44), (24, 26), (32, 36), (44, 22), (52, 44)]
    poly = [(m + x * u, m + y * u) for x, y in pts]
    neon = (0, 255, 163, int(255 * 0.85))
    draw.polygon(poly, fill=neon)


def render_full_ss(big):
    """Full brand icon (dark rounded square + mark) drawn at `big` px (SS res)."""
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _draw_brand(d, big, glyph_only=False, margin_frac=2 / 64)
    return img


def render_full(size):
    """Full brand icon (dark rounded square + mark) at `size`."""
    return render_full_ss(size * SS).resize((size, size),
                                            Image.Resampling.LANCZOS)


def render_round(size):
    """Full brand icon clipped to a circle (for ic_launcher_round).

    Built at SS resolution so the circular mask has smooth, anti-aliased edges,
    then downsampled to `size`.
    """
    big = size * SS
    img = render_full_ss(big)
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, big - 1, big - 1], fill=255)
    img.putalpha(mask)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def render_foreground(size):
    """Adaptive-icon foreground: the picture-frame glyph in the safe zone.

    Glyph art fills ~62% of the canvas (inside Android's 66dp safe zone),
    centered on transparent so the dark adaptive background shows behind it.
    """
    big = size * SS
    glyph = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glyph)
    _draw_brand(gd, big, glyph_only=True, margin_frac=0)
    # Drawn content is a centred 44/60-square; scale that to 62% of the canvas.
    content_px = big * (44 / 60)
    scale = (size * 0.62) / content_px
    gs = max(1, int(round(big * scale)))
    glyph = glyph.resize((gs, gs), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(glyph, ((size - gs) // 2, (size - gs) // 2), glyph)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=".icon-staging")
    ap.add_argument("--write", action="store_true",
                    help="write into src-tauri/gen/android res (default: staging only)")
    args = ap.parse_args()

    RES = os.path.join(os.path.dirname(__file__), "..", "src-tauri",
                       "gen", "android", "app", "src", "main", "res")
    RES = os.path.normpath(RES)
    out_root = RES if args.write else args.staging

    legacy = {  # density -> pixel size for ic_launcher / _round
        "mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192,
    }
    fg = {  # density -> pixel size for adaptive foreground
        "mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432,
    }

    for density, px in legacy.items():
        d = os.path.join(out_root, f"mipmap-{density}")
        os.makedirs(d, exist_ok=True)
        render_full(px).save(os.path.join(d, "ic_launcher.png"))
        render_round(px).save(os.path.join(d, "ic_launcher_round.png"))
        print(f"wrote mipmap-{density}: ic_launcher.png / ic_launcher_round.png ({px}px)")

    for density, px in fg.items():
        d = os.path.join(out_root, f"mipmap-{density}")
        os.makedirs(d, exist_ok=True)
        render_foreground(px).save(os.path.join(d, "ic_launcher_foreground.png"))
        print(f"wrote mipmap-{density}: ic_launcher_foreground.png ({px}px)")

    print(f"\nAll icons written under: {os.path.abspath(out_root)}")
    print("Run with --write to commit into the Android res tree.")


if __name__ == "__main__":
    main()
