"""
Minimal Streamlit app: find similar designs for an existing part or an uploaded STEP file.

Run (brepnet env, from the project root):  streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from filtered_search import load_resources, query_by_stem, search  # noqa: E402
from part_properties import get_part_properties  # noqa: E402
from search_new_part import embed_step_file  # noqa: E402

THUMBNAIL_DIR = ROOT / "data" / "processed" / "fusion360_subset"


@st.cache_resource
def get_resources():
    return load_resources()


@st.cache_data(show_spinner="Embedding the uploaded part (takes a few seconds)...")
def analyse_upload(data):
    """Properties + embedding for uploaded STEP bytes (cached, so slider changes are instant)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.step"
        path.write_bytes(data)
        props = get_part_properties(path)  # fails cleanly on an unreadable file, before BRepNet runs
        vector = embed_step_file(path)
    return props, vector


def show_properties(p):
    st.caption(
        f"Size: {p['dim1']:.1f} × {p['dim2']:.1f} × {p['dim3']:.1f}  \n"
        f"Volume: {p['volume']:,.0f}  \n"
        f"Area: {p['area']:,.0f}  \n"
        f"Faces: {p['num_faces']}"
    )


def show_thumbnail(stem):
    st.image(str(THUMBNAIL_DIR / f"{stem}.png"), width="stretch")


st.set_page_config(page_title="Shape Similarity Search", layout="wide")
st.title("Shape Similarity Search")

embeddings, manifest, index, props = get_resources()

with st.sidebar:
    source = st.radio("Query part", ["Existing part", "Upload STEP file"])
    if source == "Existing part":
        stem = st.selectbox("Part", manifest)
        upload = None
    else:
        upload = st.file_uploader("STEP file", type=["step", "stp"])
    size_filter = st.checkbox("Size filter")
    tolerance = st.slider("Size tolerance (±%)", 5, 50, 20, step=5, disabled=not size_filter) / 100

# Run the search for whichever input is active
if source == "Existing part":
    query_label, query_props, thumbnail_stem = stem, props[stem], stem
    matches = query_by_stem(stem, embeddings, manifest, index, props, size_filter, tolerance)
elif upload is None:
    st.info("Upload a STEP file in the sidebar.")
    st.stop()
else:
    try:
        query_props, vector = analyse_upload(upload.getvalue())
    except Exception as e:
        st.error(f"Could not process this file: {e}")
        st.stop()
    query_label, thumbnail_stem = upload.name, None
    matches = search(vector, query_props, index, manifest, props, size_filter, tolerance)

st.subheader("Query")
col_img, col_props = st.columns([1, 3])
with col_img:
    if thumbnail_stem:
        show_thumbnail(thumbnail_stem)
    else:
        st.write("No preview")
with col_props:
    st.write(f"**{query_label}**")
    show_properties(query_props)

st.subheader("Top matches")
if not matches:
    st.info("No matches within tolerance")
else:
    for col, m in zip(st.columns(5), matches):
        with col:
            show_thumbnail(m["stem"])
            st.write(f"**{m['stem']}**")
            st.write(f"Distance: {m['distance']:.4f}")
            show_properties(m)
