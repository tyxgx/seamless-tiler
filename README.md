# Seamless Tiler

Turns a single pattern/photo into a seamlessly tileable texture — no visible
square grid-boundary when repeated — using classical image-processing only
(PIL + numpy, no ML/GPU), with an optional generative-fill mode for the
highest-quality results.

## Setup

```bash
pip install -r requirements.txt
```

For the generative-fill method (best quality — see "v3" below), you also need
a [Replicate](https://replicate.com/account/api-tokens) API token:

```bash
cp .env.example .env   # then edit .env and paste your token in
```

## Quickest way to try it — the UI

```bash
streamlit run app.py
```

Opens a local web UI: upload a photo, pick classical (instant, free) or
generative (best quality, needs the token above), tweak the seam-band size,
and see the tiled result next to the original.

## Usage (CLI)

```bash
python3 seamless_tiler.py input/pattern.jpg output/result.png --compare
```

- `--compare` writes a before/after grid preview (original tiled on top,
  seamless tiled below) so the fix is immediately visible.
- `--blend 0.10` controls the healing-band size as a fraction of image size
  (default 0.10 — see tuning notes below).
- `--method seamcut` (default) or `--method blend` (legacy, kept for
  comparison — see "Why seamcut, not blend" below).

## How it works

**Step 1 — offset.** Circular-shift (`np.roll`) the image by half its width
and half its height. This is the classic Photoshop "offset filter" trick:
the original left/right and top/bottom edge mismatch (the real seam)
mathematically relocates to the *center* of the shifted image, and the new
edges become pixels that were already adjacent in the source photo's middle
— so edges already match, and only the center cross-seam needs fixing.

**Step 2 — heal the center seam via minimum-error-boundary cut.** Instead of
blending the seam region with its own mirror reflection (which double-
exposes content — see below), compute a per-pixel cost between the region
and its mirror, weighted down by local gradient magnitude (Kwatra et al.,
*Graphcut Textures*, SIGGRAPH 2003 — a seam is far less visible when it
runs along an existing contour than through a flat/bold area). A dynamic-
program finds the lowest-cost path through that cost map, and pixels are
taken from *either* the original *or* the mirror — never averaged — split
along that path. Only a 1px feather straddles the actual cut line.

## Why seamcut, not linear blend (v1)

The first version blended a wide band with its mirror using a fixed
triangular alpha. It removes the hard square line, but on detailed patterns
it visibly "ghosts" — a flower averaged with its own mirrored copy looks
like a double-exposure. Confirmed on real samples (see `output/` history).
Seam-carving removes this because it's a hard per-pixel choice, never an
average of two different flowers.

## Tuning: band size

`--blend` (0.08–0.12 recommended) controls how much of the tile is subject
to self-mirroring. Smaller = less area shows any duplicate content = less
visible residual artifact. Wider bands (0.2+) were only useful for the old
blend method, which needed more room to fade gradually.

## v3 — generative fill (recommended, solves the limitation below)

`generative_seamless.py` uses the same offset step, but instead of reusing/
mirroring pixels, it masks the center-seam band and sends it to **Flux-Fill**
(via the Replicate API) with a prompt describing the pattern's style/colors/
motifs. The model hallucinates genuinely new content that connects both
sides — no mirroring, so no ghosting and no mirror-symmetry fold, even on
bold sparse-background motifs where v1/v2 both failed.

```bash
python3 generative_seamless.py input.jpg output.png \
  --prompt "seamless <style> textile pattern, <colors>, <motifs>, <background>, plain fabric print with no text, no signature, no logo, no watermark" \
  --band 0.12 --guidance 30 --feather 0 --compare
```

Tuning notes learned the hard way:
- **`--feather 0` (hard-edge mask) is important.** A blurred/soft mask edge
  gets read as a partial-opacity blend region by the model, producing a
  visible hazy/desaturated band exactly at the transition — confirmed by
  A/B testing feather=20 vs feather=0 on the same image/prompt/seed.
- **Higher `--guidance` does NOT fix desaturation** — it made it worse in
  testing (guidance 60 → a wider, grayer band than guidance 30's default).
- Flux-Fill has no `negative_prompt` param (guidance-distilled model) — put
  exclusions ("no text, no signature, no logo, no watermark") directly in
  the positive prompt.
- A nice side effect: if the source pattern has an artist signature/logo
  near a tile edge, the offset step often relocates it into the seam band,
  so it gets regenerated away for free (confirmed on 2 of the 8 samples).
  Not guaranteed — depends on exact logo placement.

Requires `REPLICATE_API_TOKEN` (env var or `.env` next to the script).
Cost: a few cents per image on `black-forest-labs/flux-fill-dev`.

### Pixel-perfect edges (important for physical print)

A "looks seamless in a tiled preview" result can still fail at the *literal* tile-repeat
boundary — for fabric printing specifically, the two edges that touch when the tile repeats
must be near pixel-identical, not just visually similar at a glance. Two real bugs were found
and fixed here, confirmed via measuring the literal left-edge-vs-right-edge pixel difference:

1. **Composite the model's output back onto the original** (`composite_feather`, default 2px) —
   inpainting APIs round-trip the *whole* image through their own encoder/decoder, introducing
   tiny reconstruction noise everywhere (including edges the offset step already made match).
   Keep the model's pixels strictly inside the mask; restore exact original pixels outside it.
2. **Keep the mask's cross-shape away from the outer edges** (`edge_margin_ratio`, default 0.03) —
   without this, the horizontal/vertical seam bands literally touch the left/right/top/bottom
   edges, so those get sent to the model too — and the model has no reason to make its
   independently-generated left-edge content match its right-edge content. Confirmed via a direct
   edge-diff test: fixing this dropped mean edge difference from ~17 to ~6 on the worst sample
   (matching the theoretical floor — the offset image's own natural adjacent-pixel similarity).

## Known limitation of the classical (v1/v2) methods — solved by v3 above

| Pattern class | Result |
|---|---|
| Dense/busy all-over prints (small motifs, textured background) | **Excellent** — seam nearly invisible |
| Bold, sparse motifs on a plain/solid background | **Visible artifact remains** — a large recognizable shape (e.g. a single flower) sitting near the seam still reads as a mirror-symmetric "bowtie" reflection when duplicated, regardless of cut path, because there's no low-detail area to hide the duplication in |

This is a structural limit of any *pixel-reuse* technique (blend or cut) —
there's only one source tile, so any seam-fix must reuse existing pixels,
and reusing a bold recognizable shape via mirroring is visible no matter
how the boundary is chosen. Making it fully invisible for that class of
pattern needs **generative inpainting** (e.g. SDXL) to synthesize genuinely
new, plausible content in the seam region instead of reusing existing
pixels — the direction a production tool (like an AI Studio "Seamless"
feature) would likely take.

## Files

- `seamless_tiler.py` — the tool (CLI + `make_seamless()` / `tile_preview()`
  importable functions)
- `seamless/` — the 8 provided sample patterns
- `input/` — a synthetic dummy test pattern used for the first smoke test
- `output/final/` — seamless result + before/after preview for all 8 samples
