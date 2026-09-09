"""
blend_quality.py — automate the two steps that were, until now, done by hand
for every generative run: writing a style prompt, and picking the best of
several random-seed candidates by eye.

1. auto_style_prompt(img): derive a generic-but-descriptive prompt straight
   from the image's own dominant colors (via a small numpy-only k-means),
   instead of a human describing it. No captioning model/extra API needed.

2. score_blend_quality(shifted, mask, result): a MEASURED signal for how
   well a candidate fill actually blends — not a guess. It compares local
   texture (edge density) and color variance INSIDE the filled band against
   the region immediately OUTSIDE it. A good blend has similar texture/
   variance on both sides (content flows continuously); a bad blend — like
   the flat "hazy patch" failures seen during manual testing — has a filled
   region that's much flatter/smoother than its surroundings, which this
   catches directly. Lower score = better blend.

Together these let generate_best() below run N candidate seeds and return
the best-scoring one automatically, with no human picking a favorite.
"""

import numpy as np
from PIL import Image


def dominant_colors(img: Image.Image, k: int = 5, sample: int = 4000) -> list:
    """Small numpy-only k-means over a random pixel sample. Returns a list
    of (r,g,b) tuples, most-populous cluster first."""
    arr = np.asarray(img.convert("RGB")).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(arr), size=min(sample, len(arr)), replace=False)
    pts = arr[idx]

    centers = pts[rng.choice(len(pts), size=k, replace=False)]
    for _ in range(12):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for c in range(k):
            members = pts[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)

    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    return [tuple(int(v) for v in centers[i]) for i in order]


def _color_name(rgb: tuple) -> str:
    """RGB -> a muted, qualified hue-family name via HSV.

    Deliberately avoids ever emitting a bare strong/vivid color word (e.g.
    "red") on its own. A real photo/painting's dominant tones are almost
    always desaturated/muted versions of a hue — labeling one "red" when
    it's really a muted rust-tan caused a real, reproducible bug: Flux-Fill
    took the word literally and painted a bold, saturated red/orange splash
    into the seam, completely unlike the source (confirmed on two separate
    runs — see project MEMORY.md). Every name here is either a compound
    "muted/soft <hue>" term or an explicitly neutral one, which does not
    trigger that literal-vivid-color reading.
    """
    import colorsys
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    deg = h * 360

    if s < 0.12:
        if v > 0.85:
            return "white/cream"
        if v < 0.25:
            return "charcoal/dark"
        return "grey"

    qualifier = "muted" if s < 0.55 else "soft"
    if deg < 20 or deg >= 345:
        hue = "rust-red" if v < 0.6 else "coral"
    elif deg < 45:
        hue = "terracotta/tan"
    elif deg < 70:
        hue = "gold/amber"
    elif deg < 90:
        hue = "olive"
    elif deg < 160:
        hue = "green"
    elif deg < 200:
        hue = "teal"
    elif deg < 250:
        hue = "blue"
    elif deg < 290:
        hue = "purple"
    elif deg < 320:
        hue = "mauve"
    else:
        hue = "dusty rose"
    return f"{qualifier} {hue}"


def auto_style_prompt(img: Image.Image) -> str:
    """Build a style prompt directly from the image's own color palette —
    no manual description needed."""
    colors = dominant_colors(img, k=5)
    names = []
    for c in colors:
        n = _color_name(c)
        if n not in names:
            names.append(n)
    palette = ", ".join(names[:4])
    return (
        f"seamless pattern, {palette} color palette, matching the existing "
        "style, texture and brushwork exactly, natural continuous blend, "
        "no text, no signature, no logo, no watermark"
    )


def _edge_density(arr: np.ndarray) -> float:
    gray = arr.mean(axis=2)
    gy = np.abs(gray[2:] - gray[:-2]) if gray.shape[0] > 2 else np.zeros_like(gray[:1])
    gx = np.abs(gray[:, 2:] - gray[:, :-2]) if gray.shape[1] > 2 else np.zeros_like(gray[:, :1])
    return float(gy.mean() + gx.mean())


def detect_artifact_blob(result: Image.Image, mask: Image.Image,
                          sat_z: float = 2.5, flat_z: float = 2.5,
                          min_area_frac: float = 0.03):
    """Look inside the filled mask region for a patch that stands out from
    its own immediate surroundings — either anomalously SATURATED (catches
    literal-color-word hallucinations, like the bold orange/red splash bug
    that came from the auto-prompt once saying "red") or anomalously FLAT/
    textureless (catches the "hazy patch"/fog hallucination seen repeatedly
    during manual testing). Both were real, reproducible failure modes —
    this automates the visual check that was previously done by eye, so
    generate_and_fix() below can catch and locally re-fix them without a
    human reviewing every run. See project MEMORY.md for the failure
    history this is built from.

    Returns a pixel bbox (x0,y0,x1,y1) of the flagged area, or None if
    nothing stands out enough to be worth re-fixing.
    """
    arr = np.asarray(result.convert("RGB")).astype(np.float32) / 255.0
    mask_arr = np.asarray(mask.convert("L")) > 127
    if not mask_arr.any():
        return None

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)

    gray = arr.mean(axis=2)
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1] = np.abs(gray[2:] - gray[:-2])
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    texture = gy + gx

    # surrounding ring: dilate the mask, subtract the mask itself — this is
    # the local baseline to compare against (not a fixed global threshold,
    # since "normal" saturation/texture varies a lot by image)
    ring = mask_arr.copy()
    for _ in range(4):
        d = ring.copy()
        d[1:] |= ring[:-1]
        d[:-1] |= ring[1:]
        d[:, 1:] |= ring[:, :-1]
        d[:, :-1] |= ring[:, 1:]
        ring = d
    ring = ring & ~mask_arr
    if not ring.any():
        return None

    sat_ref_mean, sat_ref_std = sat[ring].mean(), sat[ring].std() + 1e-6
    tex_ref_mean, tex_ref_std = texture[ring].mean(), texture[ring].std() + 1e-6

    sat_outlier = mask_arr & (sat > sat_ref_mean + sat_z * sat_ref_std)
    flat_outlier = mask_arr & (texture < max(0.0, tex_ref_mean - flat_z * tex_ref_std))
    flagged = sat_outlier | flat_outlier

    if flagged.sum() >= min_area_frac * mask_arr.sum():
        ys, xs = np.where(flagged)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # A third, distinct failure mode: a "flat wash" fill that has real
    # per-pixel graininess (so per-pixel texture/edge-density looks normal —
    # the two checks above miss it) but is a single uniform BLOCK color
    # throughout, unlike the surrounding scene's varied content. Confirmed
    # on a real run: a stucco/sand-textured but perfectly uniform tan cross
    # filled the entire healing band — per-pixel std was normal, but the
    # whole region was one flat color. Caught by comparing block-level MEAN
    # color variance (not pixel-level) inside the mask vs. the reference
    # ring: a real, correct fill's block-means vary with the diverse scene
    # around it; a flat wash's block-means barely vary at all.
    block = 20
    h, w = mask_arr.shape
    block_means = []
    block_in_mask = []
    for by in range(0, h - block, block):
        for bx in range(0, w - block, block):
            cell_mask = mask_arr[by:by + block, bx:bx + block]
            frac_in = cell_mask.mean()
            if frac_in < 0.9 and frac_in > 0.1:
                continue  # skip cells straddling the mask boundary
            cell_color = arr[by:by + block, bx:bx + block].reshape(-1, 3).mean(axis=0)
            block_means.append(cell_color)
            block_in_mask.append(frac_in >= 0.9)
    block_means = np.array(block_means)
    block_in_mask = np.array(block_in_mask, dtype=bool)
    if block_in_mask.sum() >= 4 and (~block_in_mask).sum() >= 4:
        inside_block_std = block_means[block_in_mask].std(axis=0).mean()
        outside_block_std = block_means[~block_in_mask].std(axis=0).mean()
        if inside_block_std < 0.35 * outside_block_std and outside_block_std > 0.03:
            ys, xs = np.where(mask_arr)
            return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    return None


def local_patch_prompt(result: Image.Image, bbox: tuple, ring_px: int = 40) -> str:
    """Build a prompt from the colors immediately SURROUNDING a specific
    artifact bbox (not the whole image) — more targeted than
    auto_style_prompt() for a local re-fix, since it describes exactly what
    the patch needs to blend into rather than the image as a whole."""
    x0, y0, x1, y1 = bbox
    w, h = result.size
    rx0, ry0 = max(0, x0 - ring_px), max(0, y0 - ring_px)
    rx1, ry1 = min(w, x1 + ring_px), min(h, y1 + ring_px)
    ring_img = result.crop((rx0, ry0, rx1, ry1))

    # mask out the interior bbox within this crop so only the surrounding
    # ring's colors are sampled, not the artifact itself
    arr = np.asarray(ring_img.convert("RGB")).copy()
    ix0, iy0 = x0 - rx0, y0 - ry0
    ix1, iy1 = x1 - rx0, y1 - ry0
    keep = np.ones(arr.shape[:2], dtype=bool)
    keep[max(0, iy0):iy1, max(0, ix0):ix1] = False
    if keep.sum() < 50:
        return auto_style_prompt(result)

    sample_img = Image.fromarray(arr)
    colors = dominant_colors(sample_img, k=4)
    names = []
    for c in colors:
        n = _color_name(c)
        if n not in names:
            names.append(n)
    palette = ", ".join(names[:3])
    return (
        f"{palette} color palette, photorealistic, seamlessly continuing "
        "the exact surrounding texture and content, no text, no signature, "
        "no logo, no watermark"
    )


def score_blend_quality(shifted: Image.Image, mask: Image.Image,
                         result: Image.Image, ring_px: int = 24) -> float:
    """Lower = better. Compares texture (edge density) and color variance
    inside the filled band vs. a thin ring of untouched pixels just outside
    it. A candidate that produced a flat/hazy patch (seen repeatedly during
    manual testing) will have MUCH lower inside-texture than its
    surroundings — this shows up as a large mismatch score.
    """
    result_arr = np.asarray(result).astype(np.float32)
    mask_arr = np.asarray(mask.convert("L")) > 127
    if not mask_arr.any():
        return 0.0

    # outside ring: dilate the mask by ring_px via cheap manual array-shift
    # dilation (no OpenCV/scipy dependency — matches the rest of this project)
    dilated = mask_arr.copy()
    for _ in range(max(1, ring_px // 6)):
        d = dilated.copy()
        d[1:] |= dilated[:-1]
        d[:-1] |= dilated[1:]
        d[:, 1:] |= dilated[:, :-1]
        d[:, :-1] |= dilated[:, 1:]
        dilated = d
    ring = dilated & ~mask_arr
    if not ring.any():
        return 0.0

    inside_pixels = result_arr[mask_arr]
    outside_pixels = result_arr[ring]

    texture_in = inside_pixels.std()
    texture_out = outside_pixels.std()
    texture_mismatch = abs(texture_in - texture_out) / (texture_out + 1e-6)

    color_in = inside_pixels.mean(axis=0)
    color_out = outside_pixels.mean(axis=0)
    color_mismatch = float(np.linalg.norm(color_in - color_out)) / 255.0

    return float(texture_mismatch + color_mismatch)
