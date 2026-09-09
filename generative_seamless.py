"""
generative_seamless.py — v3: fix the center seam (after the same offset trick
as seamless_tiler.py) using REAL generative inpainting (Flux-Fill on
Replicate) instead of reusing/mirroring existing pixels.

Why this exists: seamless_tiler.py's classical methods (blend / seamcut)
only ever reuse pixels already in the image (mirrored). On dense patterns
that hides well; on bold motifs with plain backgrounds it always shows a
visible mirror-symmetry artifact, because there's no new content to fill
the seam with — a fundamental limit of pixel-reuse approaches (see that
file's README section on "Known limitation").

Generative inpainting sidesteps this: the model sees both sides of the
masked seam and hallucinates NEW, contextually-consistent content to
connect them — no mirroring, no ghosting.

Pipeline:
  1. Same offset step: np.roll the image by half-width/half-height so the
     real seam relocates to the center cross (edges become already-adjacent
     original pixels — free, no cost).
  2. Build a mask: white band around the center cross (the region to
     regenerate), black everywhere else (preserve exactly).
  3. Send image + mask + a style-descriptive prompt to Flux-Fill via the
     Replicate API. The model fills the masked band with new pixels that
     blend into both sides.
  4. Save result; verify via seamless_tiler.tile_preview.

Requires: REPLICATE_API_TOKEN in environment (or in a local .env file next
to this script — loaded automatically).
"""

import argparse
import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

from seamless_tiler import tile_preview
from suitability import analyze, auto_crop_for_tiling, print_report
from blend_quality import auto_style_prompt

REPLICATE_API = "https://api.replicate.com/v1"


def _load_token() -> str:
    tok = os.environ.get("REPLICATE_API_TOKEN")
    if tok:
        return tok
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REPLICATE_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "REPLICATE_API_TOKEN not found in environment or .env — "
        "export it in your terminal first."
    )


def _to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_offset_and_mask(img: Image.Image, band_ratio: float = 0.12,
                           feather: int = 0, edge_margin_ratio: float = 0.03):
    """Same offset as seamless_tiler.make_seamless. Returns (shifted_image,
    mask_image) — mask is white in the healing band around the center
    cross, black elsewhere, feathered at the mask edge so the inpainted
    region blends smoothly into the untouched pixels around it.

    edge_margin_ratio: keeps the cross arms away from the tile's outer
    edges by this fraction of size. Without this, a full-width/height
    cross touches the very edges the offset step already guarantees match
    — regenerating them there breaks that guarantee (confirmed via a real
    edge-pixel-diff test: ~half the edge pixels differed before this fix,
    because the model has no reason to make its new content at the left
    edge match its new content at the right edge — they're independently
    generated). Keeping a margin means the outermost rows/columns are
    NEVER touched by the model — they stay byte-identical to the offset
    image's already-matching pixels.
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    shifted = np.roll(arr, shift=(h // 2, w // 2), axis=(0, 1))
    shifted_img = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

    band_w = max(2, int(w * band_ratio))
    band_h = max(2, int(h * band_ratio))
    cx, cy = w // 2, h // 2
    margin_x = max(1, int(w * edge_margin_ratio))
    margin_y = max(1, int(h * edge_margin_ratio))

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    # vertical seam band — stop short of the top/bottom edges
    draw.rectangle([cx - band_w, margin_y, cx + band_w, h - margin_y], fill=255)
    # horizontal seam band — stop short of the left/right edges
    draw.rectangle([margin_x, cy - band_h, w - margin_x, cy + band_h], fill=255)
    # optional small feather just for anti-aliasing the rectangle edge —
    # keep this tiny; a wide blur turns the mask into a soft alpha over a
    # big region, which some inpainting backends read as "blend the new
    # content in at reduced opacity here", producing a hazy/desaturated
    # band exactly at the transition zone (confirmed via a real test).
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    return shifted_img, mask


def run_flux_fill(image: Image.Image, mask: Image.Image, prompt: str,
                   model: str = "black-forest-labs/flux-fill-dev",
                   guidance: float = 30, seed: int = None) -> Image.Image:
    token = _load_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "input": {
            "prompt": prompt,
            "image": _to_data_uri(image),
            "mask": _to_data_uri(mask),
            "guidance": guidance,
            "megapixels": "match_input",
            "output_format": "png",
        }
    }
    if seed is not None:
        payload["input"]["seed"] = seed

    resp = requests.post(
        f"{REPLICATE_API}/models/{model}/predictions",
        headers=headers, json=payload, timeout=60,
    )
    resp.raise_for_status()
    pred = resp.json()
    pred_url = pred["urls"]["get"]

    print(f"  prediction started ({pred['id']}), polling...")
    while True:
        time.sleep(2)
        r = requests.get(pred_url, headers=headers, timeout=30)
        r.raise_for_status()
        pred = r.json()
        status = pred["status"]
        if status == "succeeded":
            break
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Replicate prediction {status}: {pred.get('error')}")
        print(f"  ... {status}")

    output = pred["output"]
    out_url = output[0] if isinstance(output, list) else output
    img_resp = requests.get(out_url, timeout=60)
    img_resp.raise_for_status()
    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


def auto_fix_artifacts(mask: Image.Image, result: Image.Image,
                        model: str = "black-forest-labs/flux-fill-dev",
                        guidance: float = 30, max_retries: int = 2) -> Image.Image:
    """After the main fill, automatically look for a visible artifact blob
    (see blend_quality.detect_artifact_blob — an anomalously saturated or
    anomalously flat patch relative to its own surroundings) and, if found,
    re-inpaint just that local region with a prompt built from its own
    immediate surroundings, up to max_retries times.

    This automates the manual "spot-fix" workflow that fixed a real bold
    orange-splash artifact by hand (detect the bad region -> build a small
    local mask -> re-inpaint with a prompt describing just the surrounding
    colors) so it happens automatically on every run instead of requiring a
    human to notice and fix it. See project MEMORY.md for the failure this
    was built from.
    """
    from blend_quality import detect_artifact_blob, local_patch_prompt

    current = result
    mask_arr = np.asarray(mask) > 127
    for attempt in range(max_retries):
        bbox = detect_artifact_blob(current, mask)
        if bbox is None:
            break
        x0, y0, x1, y1 = bbox
        print(f"  -> artifact detected at {bbox}, auto-fixing "
              f"(attempt {attempt + 1}/{max_retries})...")

        local_mask_img = Image.new("L", current.size, 0)
        ImageDraw.Draw(local_mask_img).rectangle([x0, y0, x1, y1], fill=255)
        # never let the local fix stray outside the original healing band —
        # keeps the tile-repeat edge guarantee intact
        local_arr = (np.asarray(local_mask_img) > 127) & mask_arr
        if not local_arr.any():
            break
        local_mask_img = Image.fromarray((local_arr * 255).astype(np.uint8))

        prompt = local_patch_prompt(current, bbox)
        fill = run_flux_fill(current, local_mask_img, prompt, model=model, guidance=guidance)
        if fill.size != current.size:
            fill = fill.resize(current.size, Image.LANCZOS)

        composite_mask = local_mask_img.filter(ImageFilter.GaussianBlur(radius=3))
        cur_arr = np.asarray(current).astype(np.float32)
        fill_arr = np.asarray(fill).astype(np.float32)
        alpha = (np.asarray(composite_mask).astype(np.float32) / 255.0)[:, :, None]
        current = Image.fromarray(np.clip(
            cur_arr * (1 - alpha) + fill_arr * alpha, 0, 255).astype(np.uint8))

    return current


def make_seamless_generative(img: Image.Image, prompt: str,
                              band_ratio: float = 0.12,
                              model: str = "black-forest-labs/flux-fill-dev",
                              seed: int = None, guidance: float = 30,
                              feather: int = 0,
                              composite_feather: int = 2,
                              auto_fix: bool = True) -> Image.Image:
    """Generate the fill, then COMPOSITE it back onto the original shifted
    image — only keep the model's pixels strictly inside the mask, restore
    the exact original pixels everywhere else.

    Why this matters: inpainting models typically round-trip the WHOLE
    image through their own encoder/decoder (not just the masked region),
    which introduces tiny, imperceptible-at-a-glance reconstruction noise
    everywhere — including the tile edges, which the offset step had
    already made match *exactly*. Left un-composited, that noise silently
    breaks pixel-perfect edge matching (confirmed on a real sample: ~half
    the edge pixels showed a measurable difference, invisible normally but
    visible on physical-print-level zoom). Compositing restores the edges
    to be byte-identical to the offset image's guaranteed-matching pixels,
    and confines anything the model changed to strictly the seam region.

    composite_feather: a tiny (1-3px) blur applied ONLY to this post-hoc
    compositing mask, to avoid a single hard pixel line at the mask
    boundary. This is unrelated to (and much safer than) feathering the
    mask sent to the model — that feathering caused the hazy-band bug;
    this one just blends two already-finished images together.
    """
    shifted, mask = build_offset_and_mask(img, band_ratio, feather=feather)
    result = run_flux_fill(shifted, mask, prompt, model=model, seed=seed,
                            guidance=guidance)
    if result.size != shifted.size:
        result = result.resize(shifted.size, Image.LANCZOS)

    composite_mask = mask
    if composite_feather > 0:
        composite_mask = mask.filter(ImageFilter.GaussianBlur(radius=composite_feather))

    shifted_arr = np.asarray(shifted).astype(np.float32)
    result_arr = np.asarray(result).astype(np.float32)
    alpha = (np.asarray(composite_mask).astype(np.float32) / 255.0)[:, :, None]
    composited = shifted_arr * (1 - alpha) + result_arr * alpha
    composited_img = Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8))

    if auto_fix:
        composited_img = auto_fix_artifacts(mask, composited_img, model=model,
                                             guidance=guidance)
    return composited_img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--prompt", default=None,
                     help="describe the pattern's style/colors/motifs so the "
                          "model fills the seam with matching new content. "
                          "If omitted, a prompt is auto-generated from the "
                          "image's own dominant colors (see blend_quality.py)")
    ap.add_argument("--band", type=float, default=0.12,
                     help="healing band as fraction of size (default 0.12). "
                          "A wider band was tried as an automatic fix for "
                          "scene-like inputs but made things WORSE, not "
                          "better — more area for the model to fill means "
                          "more chance of a bad flat/hazy hallucination "
                          "(confirmed: 0.25 produced a large flat patch "
                          "where 0.12 hadn't — see project MEMORY.md). Keep "
                          "the default; if you see a visible internal seam "
                          "just outside the band, that's random-seed "
                          "variance — retry (see --candidates) rather than "
                          "widening the band.")
    ap.add_argument("--model", default="black-forest-labs/flux-fill-dev")
    ap.add_argument("--guidance", type=float, default=30)
    ap.add_argument("--feather", type=int, default=0,
                     help="mask edge blur radius in px sent to the model "
                          "(default 0 = hard edge)")
    ap.add_argument("--composite-feather", type=int, default=2,
                     help="post-hoc compositing blur in px (default 2) — "
                          "restores exact original pixels outside the mask "
                          "so tile edges stay pixel-perfect")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tile-cols", type=int, default=3)
    ap.add_argument("--tile-rows", type=int, default=3)
    ap.add_argument("--auto-crop", dest="auto_crop", action="store_true", default=True,
                     help="(default on) if the input isn't pattern-like — a "
                          "directional scene or one with a dominant focal "
                          "object — automatically crop to the most tileable "
                          "square sub-region before running the pipeline, "
                          "instead of silently producing a bad seamless tile")
    ap.add_argument("--no-auto-crop", dest="auto_crop", action="store_false",
                     help="disable the suitability check / auto-crop and "
                          "always run on the input as given")
    ap.add_argument("--candidates", type=int, default=1,
                     help="generate this many candidates with different "
                          "random seeds and save all of them (labeled "
                          "_candidateN), instead of just one. There is NO "
                          "reliable automatic way to pick the single best one "
                          "(tested and dropped — see blend_quality.py's "
                          "docstring/project MEMORY.md) — review the "
                          "candidates yourself and keep the best.")
    args = ap.parse_args()

    img = Image.open(args.input)

    report = analyze(img)
    print_report(report, label=str(args.input))
    if not report["is_pattern_like"] and args.auto_crop:
        img, crop_box, notes = auto_crop_for_tiling(img, report)
        for note in notes:
            print(f"  -> {note}")
        autocrop_path = args.output.with_name(args.output.stem + "_autocrop_source.png")
        img.save(autocrop_path)
        print(f"  -> auto-crop source saved: {autocrop_path}")
    elif not report["is_pattern_like"]:
        print("  -> --no-auto-crop set: proceeding on the full input anyway "
              "(result may show a visible repeat/duplication artifact)")

    band = args.band

    prompt = args.prompt
    if prompt is None:
        prompt = auto_style_prompt(img)
        print(f"  -> no --prompt given, auto-generated from dominant colors: "
              f"{prompt!r}")

    n = max(1, args.candidates)
    for i in range(n):
        seed = args.seed
        if n > 1 and seed is None:
            seed = 1000 + i  # deterministic-but-distinct seeds across candidates
        out_path = args.output if n == 1 else args.output.with_name(
            f"{args.output.stem}_candidate{i+1}{args.output.suffix}")

        print(f"generating seamless fill ({i+1}/{n}, seed={seed}) for "
              f"{args.input} via {args.model} (guidance={args.guidance})...")
        result = make_seamless_generative(
            img, prompt, band_ratio=band, model=args.model,
            seed=seed, guidance=args.guidance, feather=args.feather,
            composite_feather=args.composite_feather)
        result.save(out_path)
        print(f"seamless tile written: {out_path}")

        if args.compare:
            preview_path = out_path.with_name(out_path.stem + "_preview.png")
            seamless_grid = tile_preview(result, args.tile_cols, args.tile_rows)
            original_grid = tile_preview(img.convert("RGB"), args.tile_cols, args.tile_rows)
            w, h = original_grid.size
            combo = Image.new("RGB", (w, h * 2 + 10), "white")
            combo.paste(original_grid, (0, 0))
            combo.paste(seamless_grid, (0, h + 10))
            combo.save(preview_path)
            print(f"tiled preview written: {preview_path}")

    if n > 1:
        print(f"-> {n} candidates written ({args.output.stem}_candidate1..{n}"
              f"{args.output.suffix}) — review them and keep the best; no "
              "automatic picker is applied (see note above).")


if __name__ == "__main__":
    main()
