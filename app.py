"""
app.py — minimal local UI for the seamless tiler.

Upload an image, get a seamless tile back, see it repeated as a tiled
preview. No settings — everything (suitability check, auto-crop, style
prompt, seed) runs automatically behind the scenes.

Run with:
    streamlit run app.py
"""

import io

import streamlit as st
from PIL import Image

from seamless_tiler import make_seamless, tile_preview
from generative_seamless import make_seamless_generative
from suitability import analyze, auto_crop_for_tiling
from blend_quality import auto_style_prompt

st.set_page_config(page_title="Seamless Tiler", page_icon="🧵", layout="wide")
st.title("🧵 Seamless Tiler")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    report = analyze(img)
    if not report["is_pattern_like"]:
        img, _crop_box, _notes = auto_crop_for_tiling(img, report)

    method = st.radio(
        "Method", ["Generative (best quality, ~5-15 min, random each time)",
                    "Classical (instant, free, same result every time)"],
        horizontal=True,
    )
    is_generative = method.startswith("Generative")

    generate = st.button("Generate", type="primary")

    if generate:
        with st.spinner("Generating..." if is_generative else "Processing..."):
            try:
                if is_generative:
                    prompt = auto_style_prompt(img)
                    tile = make_seamless_generative(img, prompt, band_ratio=0.12, guidance=30)
                else:
                    tile = make_seamless(img, blend_ratio=0.10, method="seamcut")
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()
        st.session_state["tile"] = tile
        st.session_state["src"] = img

    if "tile" in st.session_state:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(st.session_state["src"], caption="Input", use_container_width=True)
        with col_b:
            st.image(st.session_state["tile"], caption="Output", use_container_width=True)

        st.image(tile_preview(st.session_state["tile"], 3, 3),
                  caption="Tiled", use_container_width=True)

        buf = io.BytesIO()
        st.session_state["tile"].save(buf, format="PNG")
        st.download_button("Download", buf.getvalue(), file_name="seamless_tile.png",
                            mime="image/png")
