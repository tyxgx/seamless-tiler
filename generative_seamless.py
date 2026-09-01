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


def make_seamless_generative(img: Image.Image, prompt: str,
                              band_ratio: float = 0.12,
                              model: str = "black-forest-labs/flux-fill-dev",
                              seed: int = None, guidance: float = 30,
                              feather: int = 0,
                              composite_feather: int = 2) -> Image.Image:
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
    return Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--prompt", required=True,
                     help="describe the pattern's style/colors/motifs so the "
                          "model fills the seam with matching new content")
    ap.add_argument("--band", type=float, default=0.12)
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
    args = ap.parse_args()

    img = Image.open(args.input)
    print(f"generating seamless fill for {args.input} via {args.model} "
          f"(guidance={args.guidance})...")
    result = make_seamless_generative(
        img, args.prompt, band_ratio=args.band, model=args.model,
        seed=args.seed, guidance=args.guidance, feather=args.feather,
        composite_feather=args.composite_feather)
    result.save(args.output)
    print(f"seamless tile written: {args.output}")

    if args.compare:
        preview_path = args.output.with_name(args.output.stem + "_preview.png")
        seamless_grid = tile_preview(result, args.tile_cols, args.tile_rows)
        original_grid = tile_preview(img.convert("RGB"), args.tile_cols, args.tile_rows)
        w, h = original_grid.size
        combo = Image.new("RGB", (w, h * 2 + 10), "white")
        combo.paste(original_grid, (0, 0))
        combo.paste(seamless_grid, (0, h + 10))
        combo.save(preview_path)
        print(f"tiled preview written: {preview_path}")


if __name__ == "__main__":
    main()
