"""
app.py — local Streamlit UI for the seamless tiler.

Upload any pattern photo, tweak a few settings, hit Generate, and see the
tiled result (repeated in a grid) right next to the original so the seam
fix is immediately visible.

Run with:
    streamlit run app.py
"""

import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from seamless_tiler import make_seamless, tile_preview
from generative_seamless import make_seamless_generative

st.set_page_config(page_title="Seamless Tiler", page_icon="🧵", layout="wide")
st.title("🧵 Seamless Tiler")
st.caption("Upload a pattern/photo, generate a seamlessly tileable version, "
           "and preview it repeated in a grid.")

with st.sidebar:
    st.header("Settings")

    method = st.radio(
        "Method",
        ["Generative fill (best quality, needs API key)", "Classical (local, free, instant)"],
        index=0,
    )
    is_generative = method.startswith("Generative")

    band = st.slider("Seam band size", 0.05, 0.30, 0.12, 0.01,
                      help="How much of the tile is used to heal the seam. "
                           "Smaller = less area touched.")

    if is_generative:
        guidance = st.slider("Guidance", 5, 100, 30, 5,
                              help="Higher isn't always better — 30 (default) "
                                   "tested best. Higher can wash out colors.")
        seed = st.number_input("Seed (0 = random)", min_value=0, value=123, step=1)
        prompt = st.text_area(
            "Style prompt",
            value="seamless textile pattern, matching colors and motifs, "
                  "same illustration style, plain fabric print with no text, "
                  "no signature, no logo, no watermark",
            height=120,
            help="Describe the pattern's style/colors/motifs — the model "
                 "fills the seam with matching new content. Edit this to "
                 "mention the actual colors/subject for best results.",
        )
        st.caption("Needs REPLICATE_API_TOKEN set (env var or .env next to "
                   "this script).")
    else:
        seamcut_method = st.radio("Classical sub-method", ["seamcut", "blend"], index=0)
        st.caption("Free, instant, no API — but may show a residual mirror "
                   "artifact on bold/sparse patterns. See README.md.")

    st.divider()
    cols = st.slider("Preview grid columns", 2, 5, 3)
    rows = st.slider("Preview grid rows", 2, 5, 3)

uploaded = st.file_uploader("Upload a pattern image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Original")
        st.image(img, use_container_width=True)

    generate = st.button("Generate seamless tile", type="primary")

    if generate:
        with st.spinner("Generating..." if is_generative else "Processing..."):
            try:
                if is_generative:
                    tile = make_seamless_generative(
                        img, prompt, band_ratio=band,
                        seed=(seed or None), guidance=guidance,
                    )
                else:
                    tile = make_seamless(img, blend_ratio=band, method=seamcut_method)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                if "REPLICATE_API_TOKEN" in str(e):
                    st.info("Set your Replicate API token in a terminal:\n\n"
                            "`export REPLICATE_API_TOKEN=r8_...`\n\n"
                            "then restart this app.")
                st.stop()

        st.session_state["tile"] = tile

    if "tile" in st.session_state:
        tile = st.session_state["tile"]
        with col_b:
            st.subheader("Seamless tile")
            st.image(tile, use_container_width=True)

        st.subheader(f"Tiled preview ({cols}×{rows})")
        preview = tile_preview(tile, cols=cols, rows=rows)
        st.image(preview, use_container_width=True)

        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        st.download_button("Download seamless tile (PNG)", buf.getvalue(),
                            file_name="seamless_tile.png", mime="image/png")
else:
    st.info("Upload an image above to get started.")
