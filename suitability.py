"""
suitability.py — decide whether an input image is a good candidate for the
offset+heal seamless-tiling approach (seamless_tiler.py / generative_seamless.py),
and if not, automatically pick a better sub-region to tile instead.

Why this exists: the offset+heal technique (np.roll by half-size, then patch
only a narrow band around the resulting center seam) works great on genuine
repeat-unit textile swatches — dense, all-over motifs with no directional
"up vs down" composition (confirmed on all 8 of the Ruchita Design samples in
seamless/). It silently produces a bad result on a different class of image:
a one-off photo/artwork with a directional/gravity-bound composition (e.g.
sky above, ground/water below) — tiling that vertically repeats the up/down
narrative no matter how well the seam pixels are healed, because that's a
*compositional* mismatch, not a seam-pixel mismatch (confirmed empirically —
see project MEMORY.md).

Earlier version of this module tried to detect that via hand-guessed pixel
heuristics (a grid edge-density "dominant object" scan, a row-brightness
"directional gravity" correlation). Both were tested against the real 8
provided samples plus the failing scenic test image and gave WRONG verdicts
on both — flagging a genuine, previously-working textile sample as bad and
missing the actual scenic photo entirely. Guessing from raw pixels doesn't
generalize.

This version instead uses a MEASURED signal already produced by the tiling
pipeline itself: seamless_tiler.measure_seam_tileability() runs the real
seam-cut cost map + dynamic-program path (the same code that actually heals
the seam) and reports how expensive the best available cut is. A low cost
means the DP found rows/columns where the offset band and its mirror already
agree closely — a natural place to hide the cut, typical of an all-over
motif. A high cost means original and mirror disagree badly along the
*entire* band — there's nowhere good to cut, typical of a directional scene
where content near the seam differs by more than any local rearrangement can
fix. Combined with a simple aspect-ratio check (genuine textile repeat units
are conventionally square; the failing scenic test was 1.79:1), this was
calibrated directly against the 8 real samples + the failing case (see
project MEMORY.md for the numbers) and cleanly separates them with margin —
every real sample stayed well under both thresholds, only the scenic photo
crossed both.
"""

import numpy as np
from PIL import Image

from seamless_tiler import measure_seam_tileability

# Calibrated against the 8 real Ruchita Design samples (all square, avg
# seam-cut cost 0.92-25.49) vs. the failing scenic test image (1.79:1,
# avg cost 34.80) — both thresholds sit in the gap with margin on both sides.
ASPECT_RATIO_THRESHOLD = 1.3
AVG_COST_THRESHOLD = 28.0


def _edge_density_grid(arr: np.ndarray, grid: int = 8) -> np.ndarray:
    """Mean gradient magnitude per grid cell, as a (grid, grid) array. Used
    only by auto_crop_for_tiling to pick the most texturally-uniform square
    sub-region — NOT used for the suitability verdict itself (see module
    docstring for why the earlier pixel-heuristic approach was dropped)."""
    gray = arr.mean(axis=2)
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1] = gray[2:] - gray[:-2]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    mag = np.sqrt(gx ** 2 + gy ** 2)

    h, w = mag.shape
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    out = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            cell = mag[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            out[r, c] = cell.mean() if cell.size else 0
    return out


def analyze(img: Image.Image, blend_ratio: float = 0.12) -> dict:
    """Run the suitability check and return a report dict:

    is_pattern_like: bool — overall verdict
    aspect_ratio: float — max(w,h)/min(w,h)
    horizontal_cost / vertical_cost / avg_cost: float — measured seam-cut
      cost from seamless_tiler.measure_seam_tileability (see module
      docstring)
    reasons: list[str] — human-readable explanation of the verdict
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    aspect_ratio = max(w, h) / max(1, min(w, h))

    cost = measure_seam_tileability(img, blend_ratio=blend_ratio)
    avg_cost = (cost["horizontal_cost"] + cost["vertical_cost"]) / 2

    bad_aspect = aspect_ratio > ASPECT_RATIO_THRESHOLD
    bad_cost = avg_cost > AVG_COST_THRESHOLD
    is_pattern_like = not (bad_aspect and bad_cost)
    # require BOTH signals to agree before overriding the pipeline — see
    # calibration note in the module docstring: the hardest real sample
    # (bold motif, needed generative v3 tuning) crossed the cost threshold
    # on one axis alone but stayed square and under the AVERAGE cost
    # threshold, so a single-signal trigger would have wrongly flagged it.

    reasons = []
    if bad_aspect and bad_cost:
        reasons.append(
            f"aspect ratio {aspect_ratio:.2f}:1 (genuine textile repeat units are "
            "conventionally square) AND a high measured seam-cut cost "
            f"(avg={avg_cost:.1f}, threshold={AVG_COST_THRESHOLD:.0f} — the best "
            "available cut still disagrees badly with its mirror almost "
            "everywhere) together indicate this is a directional scene/"
            "composition, not an all-over repeating motif — tiling it will "
            "show a visible repeat/seam no matter how the pipeline is tuned"
        )
    else:
        if bad_aspect:
            reasons.append(
                f"aspect ratio {aspect_ratio:.2f}:1 is far from square (soft "
                "warning only — the measured seam cost is fine, so proceeding)")
        if bad_cost:
            reasons.append(
                f"seam-cut cost is elevated (avg={avg_cost:.1f}) — a harder-than-"
                "usual case (like a bold, sparse motif) but the aspect ratio is "
                "fine, so this proceeds through the normal pipeline (consider "
                "the generative method with a smaller --band for best results)")
        if not reasons:
            reasons.append(
                f"square-ish aspect ratio ({aspect_ratio:.2f}:1) and a low measured "
                f"seam-cut cost (avg={avg_cost:.1f}) — looks like a genuine "
                "all-over repeating pattern")

    return {
        "is_pattern_like": is_pattern_like,
        "aspect_ratio": aspect_ratio,
        "horizontal_cost": cost["horizontal_cost"],
        "vertical_cost": cost["vertical_cost"],
        "avg_cost": avg_cost,
        "reasons": reasons,
    }


def auto_crop_for_tiling(img: Image.Image, report: dict = None, grid: int = 8) -> tuple:
    """Search the image for the best square sub-region to tile instead of
    the whole photo: square (removes directional top/bottom composition by
    construction), and internally as texturally uniform as possible (low
    variance in per-cell edge density = pattern-like) among the candidates,
    with the LOWEST measured seam-cut cost of any candidate winning ties.

    Returns (cropped_image, crop_box_px, notes: list[str]).
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    side = min(h, w)

    n_steps = 9
    if h >= w:
        offsets = [(int(t * (h - side)), 0) for t in np.linspace(0, 1, n_steps)]
    else:
        offsets = [(0, int(t * (w - side))) for t in np.linspace(0, 1, n_steps)]

    best = None
    best_score = None
    for oy, ox in offsets:
        crop_arr = arr[oy:oy + side, ox:ox + side]
        cdensity = _edge_density_grid(crop_arr, grid=grid)
        uniformity = -cdensity.std() / (cdensity.mean() + 1e-6)  # higher = more uniform
        if best_score is None or uniformity > best_score:
            best_score = uniformity
            best = (oy, ox)

    oy, ox = best
    crop_box = (ox, oy, ox + side, oy + side)
    cropped = img.convert("RGB").crop(crop_box)

    notes = [
        f"auto-cropped to a {side}x{side} square at ({ox},{oy}) — chosen as the "
        "most texturally-uniform square region available, to remove the "
        "directional top-to-bottom composition that a non-square crop can't avoid"
    ]
    return cropped, crop_box, notes


def print_report(report: dict, label: str = "input") -> None:
    verdict = "PATTERN-LIKE (proceeding normally)" if report["is_pattern_like"] \
        else "SCENE-LIKE (not a good tiling candidate as-is)"
    print(f"suitability check for {label}: {verdict}  "
          f"[aspect={report['aspect_ratio']:.2f}:1, "
          f"seam-cost avg={report['avg_cost']:.1f} "
          f"(h={report['horizontal_cost']:.1f}, v={report['vertical_cost']:.1f})]")
    for reason in report["reasons"]:
        print(f"  - {reason}")
