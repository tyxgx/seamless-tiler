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

Opens a local web UI, deliberately minimal: upload an image, click Generate, get the
seamless tile + a tiled preview + a download button. No exposed settings — suitability
check, auto-crop, style prompt, and band size all run automatically with the tuned
defaults from the CLI. Uses the generative method (needs the token above); for the free
classical method or to tweak individual parameters, use the CLI directly (below).

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

## Automatic suitability check + auto-crop

Every run (both `seamless_tiler.py` and `generative_seamless.py`) starts by analyzing the
input via `suitability.py`:

- **Aspect ratio** — genuine textile repeat units are conventionally square. A strongly
  elongated input is a soft warning sign.
- **Measured seam-cut cost** — the classical seam-cut's own cost map (see "Why seamcut"
  above) is run as a probe: a low cost means the DP found a natural place to hide the cut
  (typical of an all-over motif); a high cost means original and mirror disagree badly
  everywhere along the band — typical of a directional scene (sky above / ground below)
  rather than a repeating pattern. This is a *measured* signal, calibrated directly against
  the 8 real provided samples (all square, cost 0.9–25.5) vs. a failing one-off scenic photo
  (1.79:1, cost 34.8) — both thresholds sit in the gap with margin.

If **both** signals disagree with "pattern-like" (never just one — the hardest real sample,
a bold bird-of-paradise print, crossed the cost threshold on one axis alone but stayed
square and passed on average; a single-signal trigger would have wrongly flagged it), the
input is auto-cropped to the most texturally-uniform square sub-region before continuing —
removing the directional composition a non-square crop can't avoid. `--no-auto-crop`
disables this and forces the full input through as-is.

## Automatic style prompt (generative method only)

`generative_seamless.py`'s `--prompt` is now optional. If omitted, `blend_quality.py`
derives one from the input's own dominant colors (a small numpy-only k-means + HSV hue
naming) — no manual description needed. Color names are deliberately always a "muted/soft
<hue>" compound term, never a bare vivid word: labeling a muted rust-tan "red" once made
Flux-Fill paint a literal bold red/orange splash into the seam (confirmed, reproduced twice)
— the model reads plain color words literally.

## Automatic local artifact fix (generative method only)

After the main fill, `auto_fix_artifacts()` scans the healed band for a patch that's
anomalously **saturated** relative to its own immediate surroundings (catches literal-
color-word hallucinations like the red/orange splash above) and, if found, re-inpaints just
that local region using a prompt built from the colors immediately surrounding it — up to 2
retries. This is on by default (`auto_fix=True` in `make_seamless_generative`).

**Known gap, stated plainly:** this only reliably catches the *oversaturated* failure mode.
A separate failure mode — a flat, uniformly-colored "wash" patch (real per-pixel graininess,
but the same block-level average color throughout, unlike the varied scene around it) —
was tested against three different heuristics (saturation outlier, texture/edge-density
outlier, block-level color-variance outlier) and **none reliably caught it**. Generative
inpainting output is inherently stochastic: the same prompt and mostly-default settings can
occasionally produce this on one run and not the next. **Building a fully reliable automatic
detector for arbitrary generative artifacts is a genuinely open problem** (real no-reference
image-quality-assessment territory) — a cheap, much more reliable option that was scoped
but not implemented here (no API key available in this environment) is a Claude Haiku
vision call per generation (~$0.002/check) to actually *look at* the result and judge it,
the same way a human reviews it today. Until that's wired in: **if a result looks off,
just regenerate** (seed is random by default) — this is the normal, expected workflow for
this class of tool, not a bug to chase.

## v3 — generative fill (recommended, solves the limitation below)

`generative_seamless.py` uses the same offset step, but instead of reusing/
mirroring pixels, it masks the center-seam band and sends it to **Flux-Fill**
(via the Replicate API) with a prompt describing the pattern's style/colors/
motifs. The model hallucinates genuinely new content that connects both
sides — no mirroring, so no ghosting and no mirror-symmetry fold, even on
bold sparse-background motifs where v1/v2 both failed.

```bash
# --prompt is optional — auto-generated from the image's own colors if omitted
python3 generative_seamless.py input.jpg output.png --compare

# or be explicit / override any of the automatic defaults
python3 generative_seamless.py input.jpg output.png \
  --prompt "seamless <style> textile pattern, <colors>, <motifs>, <background>, plain fabric print with no text, no signature, no logo, no watermark" \
  --band 0.12 --guidance 30 --feather 0 --compare

# generate several random-seed candidates to review by hand (no reliable
# auto-picker exists — see "Automatic local artifact fix" above)
python3 generative_seamless.py input.jpg output.png --candidates 3 --compare
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

## Scope: what this tool is (and isn't) for

Built and tuned for genuine **all-over repeating textile/pattern designs** — the kind a
print/textile designer already treats as one repeat unit (dense motifs, no single unique
focal object, square-ish). On that class of input this works reliably end to end with no
manual tuning.

It is **not** built for arbitrary one-off photos/scenes (a landscape, a building, a portrait
composition) — those have directional composition and unique focal objects that will always
show a visible repeat when tiled, no matter how the pipeline is tuned. The suitability check
above will flag this class of input and auto-crop to the least-bad sub-region as a
best-effort fallback, but that fallback intentionally cannot be made to look like a proper
seamless pattern the way a real repeat-unit design can. Test with a real pattern sample from
`seamless/` first if you want to see the tool working as designed.

## Files

- `seamless_tiler.py` — the classical tool (CLI + `make_seamless()` / `tile_preview()` /
  `measure_seam_tileability()` importable functions)
- `generative_seamless.py` — the generative-fill tool (CLI + `make_seamless_generative()` /
  `auto_fix_artifacts()` importable functions)
- `suitability.py` — pattern-vs-scene suitability check + auto-crop (see above)
- `blend_quality.py` — auto style-prompt generation + artifact-blob detection (see above)
- `app.py` — minimal Streamlit UI (upload → Generate → tiled preview), runs everything above
  automatically with no exposed settings
- `seamless/` — the 8 provided sample patterns (real repeat-unit textile designs — use these
  to see the tool working within its intended scope)
- `input/` — test images (gitignored — not part of the repo)
- `output/` — generated results (gitignored — not part of the repo)
