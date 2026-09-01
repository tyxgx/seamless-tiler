"""
seamless_tiler.py — turn any input pattern/photo into a seamlessly tileable
texture, then render a repeated-grid preview so you can visually check that
no square/grid boundary seam is visible.

Both methods below start with the same first step:

  1. Circular-shift (np.roll) the image by half its width and half its
     height. The original left/right (and top/bottom) edge mismatch — the
     actual "seam" — mathematically relocates to the CENTER of the shifted
     image. The new edges (after the shift) come from pixels that were
     already adjacent in the middle of the source photo, so they already
     match — we never have to touch the edges directly. Healing then only
     needs to fix the center cross-seam.

Two healing methods are available (--method):

  blend (v1, legacy): mirror-reflect a band around the center seam and
    linearly cross-fade original vs mirrored content with a fixed alpha
    ramp, uniformly across the whole band. Simple, but on detailed/busy
    patterns this visibly "double-exposes" content (a flower blended with
    its own mirrored copy looks like a ghost/overlay) — confirmed on real
    textile samples.

  seamcut (v2, default): instead of blending everywhere uniformly, use a
    minimum-error-boundary cut (the classic Efros-Freeman "image quilting"
    idea, same family as Avidan-Shamir seam carving). Compare the original
    band against its own mirror pixel-by-pixel, find the lowest-cost path
    through that error map via dynamic programming, and take a HARD choice
    per pixel (original OR mirrored, never an average) split along that
    path. The path naturally snakes through low-detail/background areas
    (where original and mirror already agree), so the seam becomes nearly
    invisible with no ghosting, because dissimilar content is never
    overlaid — only one candidate's pixel is ever kept at any point.

No OpenCV/scipy dependency — PIL + numpy only.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# v1: linear mirror-blend (legacy — kept for comparison via --method blend)
# ---------------------------------------------------------------------------

def _heal_seam_horizontal_blend(arr: np.ndarray, blend_ratio: float) -> np.ndarray:
    h, w = arr.shape[:2]
    band = max(2, int(w * blend_ratio))
    cx = w // 2
    lo, hi = cx - band, cx + band
    lo_c, hi_c = max(0, lo), min(w, hi)

    region = arr[:, lo_c:hi_c].astype(np.float32)
    mirrored = region[:, ::-1]

    n = region.shape[1]
    x = np.linspace(-1, 1, n)
    alpha = 1.0 - np.abs(x)
    alpha = alpha.reshape(1, n, 1)

    blended = region * (1 - alpha) + mirrored * alpha
    out = arr.copy()
    out[:, lo_c:hi_c] = blended
    return out


def _heal_seam_vertical_blend(arr: np.ndarray, blend_ratio: float) -> np.ndarray:
    h, w = arr.shape[:2]
    band = max(2, int(h * blend_ratio))
    cy = h // 2
    lo, hi = cy - band, cy + band
    lo_c, hi_c = max(0, lo), min(h, hi)

    region = arr[lo_c:hi_c, :].astype(np.float32)
    mirrored = region[::-1, :]

    n = region.shape[0]
    y = np.linspace(-1, 1, n)
    alpha = 1.0 - np.abs(y)
    alpha = alpha.reshape(n, 1, 1)

    blended = region * (1 - alpha) + mirrored * alpha
    out = arr.copy()
    out[lo_c:hi_c, :] = blended
    return out


# ---------------------------------------------------------------------------
# v2: minimum-error-boundary cut (default)
# ---------------------------------------------------------------------------

def _gradient_magnitude(region: np.ndarray) -> np.ndarray:
    """Simple central-difference gradient magnitude per pixel, summed over
    channels. Used to bias the seam-cut toward existing high-detail edges
    (Kwatra et al.'s graph-cut texture insight: a seam is far less visible
    when it runs along a contour that's already there than when it cuts
    through a flat/bold area)."""
    gy = np.zeros_like(region)
    gx = np.zeros_like(region)
    gy[1:-1] = region[2:] - region[:-2]
    gx[:, 1:-1] = region[:, 2:] - region[:, :-2]
    return np.sqrt(np.sum(gx ** 2, axis=2) + np.sum(gy ** 2, axis=2))


def _min_cost_vertical_path(cost: np.ndarray) -> np.ndarray:
    """cost: (H, W) per-pixel error map. Returns an int array of length H —
    the column index of the least-cost top-to-bottom path (a seam that can
    step -1/0/+1 column per row), found via dynamic programming."""
    h, w = cost.shape
    dp = cost.astype(np.float64).copy()
    back = np.zeros((h, w), dtype=np.int8)

    for r in range(1, h):
        prev = dp[r - 1]
        left = np.empty(w)
        left[0] = np.inf
        left[1:] = prev[:-1]
        right = np.empty(w)
        right[-1] = np.inf
        right[:-1] = prev[1:]
        center = prev

        stacked = np.stack([left, center, right], axis=0)
        choice = np.argmin(stacked, axis=0)  # 0=left(-1), 1=center(0), 2=right(+1)
        best = stacked[choice, np.arange(w)]

        dp[r] = cost[r] + best
        back[r] = choice - 1

    path = np.zeros(h, dtype=np.int64)
    path[-1] = int(np.argmin(dp[-1]))
    for r in range(h - 2, -1, -1):
        nxt = path[r + 1]
        path[r] = int(np.clip(nxt + back[r + 1, nxt], 0, w - 1))
    return path


def _seamcut_mask_from_path(path: np.ndarray, w: int) -> np.ndarray:
    """Build a boolean (H, W) mask: True = keep original ('A' side, left of
    the cut), False = use the alternate candidate ('B' side, right of cut)."""
    h = path.shape[0]
    col_idx = np.arange(w).reshape(1, w)
    mask = col_idx <= path.reshape(h, 1)
    return mask


def _heal_seam_horizontal_seamcut(arr: np.ndarray, blend_ratio: float) -> np.ndarray:
    """Fix the vertical center-seam using a min-cost cut between the region
    and its own horizontal mirror, instead of a uniform blend."""
    h, w = arr.shape[:2]
    band = max(2, int(w * blend_ratio))
    cx = w // 2
    lo, hi = cx - band, cx + band
    lo_c, hi_c = max(0, lo), min(w, hi)

    region = arr[:, lo_c:hi_c].astype(np.float32)
    mirrored = region[:, ::-1]

    color_diff = np.sum((region - mirrored) ** 2, axis=2)  # (H, band_w)
    gmag = _gradient_magnitude(region) + _gradient_magnitude(mirrored)
    cost = color_diff / (gmag + 25.0)  # edge-aware: prefer cutting along
    # existing high-detail contours over flat/bold interior areas
    path = _min_cost_vertical_path(cost)
    mask = _seamcut_mask_from_path(path, region.shape[1])[:, :, None]

    healed_region = np.where(mask, region, mirrored)

    # feather only the 1px straddling the cut itself, to avoid a hard jaggy
    # single-pixel line, without reintroducing a wide ghosting band.
    feather = 1
    for r in range(h):
        c = path[r]
        for d in range(-feather, feather + 1):
            cc = c + d
            if 0 <= cc < region.shape[1]:
                t = (d + feather + 1) / (2 * feather + 2)
                healed_region[r, cc] = (1 - t) * region[r, cc] + t * mirrored[r, cc]

    out = arr.copy()
    out[:, lo_c:hi_c] = healed_region
    return out


def _heal_seam_vertical_seamcut(arr: np.ndarray, blend_ratio: float) -> np.ndarray:
    """Fix the horizontal center-seam using a min-cost cut between the
    region and its own vertical mirror. Implemented by transposing and
    reusing the horizontal logic."""
    transposed = np.transpose(arr, (1, 0, 2))
    healed_t = _heal_seam_horizontal_seamcut(transposed, blend_ratio)
    return np.transpose(healed_t, (1, 0, 2))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def make_seamless(img: Image.Image, blend_ratio: float = 0.10,
                   method: str = "seamcut") -> Image.Image:
    """Return a seamlessly-tileable version of img.

    blend_ratio: fraction of width/height used as the healing band around
    the center seam. For "seamcut", keep this SMALL (0.08-0.12) — it
    directly controls how much of the tile shows self-mirrored content, so
    smaller = less visible artifact residue. For "blend" (legacy), a wider
    band (0.2-0.3) was used to spread the ghosting more gradually.
    method: "seamcut" (default, min-error-boundary cut, no ghosting, small
    residual mirror artifact) or "blend" (legacy uniform mirror cross-fade
    — kept for comparison, has a visible ghosting artifact).
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]

    shifted = np.roll(arr, shift=(h // 2, w // 2), axis=(0, 1))

    if method == "blend":
        healed = _heal_seam_horizontal_blend(shifted, blend_ratio)
        healed = _heal_seam_vertical_blend(healed, blend_ratio)
    elif method == "seamcut":
        healed = _heal_seam_horizontal_seamcut(shifted, blend_ratio)
        healed = _heal_seam_vertical_seamcut(healed, blend_ratio)
    else:
        raise ValueError(f"unknown method: {method!r} (use 'blend' or 'seamcut')")

    healed = np.clip(healed, 0, 255).astype(np.uint8)
    return Image.fromarray(healed)


def tile_preview(tile: Image.Image, cols: int = 3, rows: int = 3) -> Image.Image:
    """Repeat `tile` in a cols x rows grid so seams (if any) become visible."""
    w, h = tile.size
    preview = Image.new("RGB", (w * cols, h * rows))
    for r in range(rows):
        for c in range(cols):
            preview.paste(tile, (c * w, r * h))
    return preview


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="input pattern/photo")
    ap.add_argument("output", type=Path, help="output seamless tile (png)")
    ap.add_argument("--method", choices=["seamcut", "blend"], default="seamcut",
                     help="healing method (default: seamcut)")
    ap.add_argument("--blend", type=float, default=0.10,
                     help="healing band as fraction of size (default 0.10 — "
                          "tuned so the seamcut method's residual mirror "
                          "artifact stays small and localized)")
    ap.add_argument("--tile-cols", type=int, default=3)
    ap.add_argument("--tile-rows", type=int, default=3)
    ap.add_argument("--preview", type=Path, default=None,
                     help="also write a repeated-grid preview image here")
    ap.add_argument("--compare", action="store_true",
                     help="also write an ORIGINAL-tiled preview next to the "
                          "seamless one, for a clear before/after")
    args = ap.parse_args()

    img = Image.open(args.input)
    tile = make_seamless(img, blend_ratio=args.blend, method=args.method)
    tile.save(args.output)
    print(f"seamless tile written ({args.method}): {args.output}")

    if args.preview or args.compare:
        preview_path = args.preview or args.output.with_name(
            args.output.stem + "_preview.png")
        seamless_grid = tile_preview(tile, args.tile_cols, args.tile_rows)

        if args.compare:
            original_grid = tile_preview(
                img.convert("RGB"), args.tile_cols, args.tile_rows)
            w, h = original_grid.size
            combo = Image.new("RGB", (w, h * 2 + 10), "white")
            combo.paste(original_grid, (0, 0))
            combo.paste(seamless_grid, (0, h + 10))
            combo.save(preview_path)
        else:
            seamless_grid.save(preview_path)
        print(f"tiled preview written: {preview_path}")


if __name__ == "__main__":
    main()
