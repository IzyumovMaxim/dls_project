import streamlit as st
import sys
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fever_search.config import load_config
from fever_search.search import SearchEngine

st.set_page_config(page_title="FEVER Evidence Search", page_icon="🔍", layout="centered")

st.title("🔍 FEVER Evidence Search")
st.markdown("**Deep Learning for Search** — Vector Search over Wikipedia")

st.sidebar.header("Settings")

config_path = st.sidebar.text_input(
    "Config Path",
    value="configs/e5_base_flat.yaml"
)

top_k = st.sidebar.slider("Number of Results", 5, 20, 10)

model_path = st.sidebar.text_input(
    "Fine-tuned Model Path (optional)",
    value="",
    help="Leave empty to use base model"
)

@st.cache_resource
def get_engine(config_path: str, model_path: str):
    config = load_config(config_path)
    if not config.model.device or "cuda" in str(config.model.device).lower():
        config.model.device = "cpu"
    mp = model_path.strip() if model_path.strip() else None
    return SearchEngine(config, model_path=mp)

try:
    engine = get_engine(config_path, model_path)
    st.sidebar.success(f"✅ Loaded {engine.document_count:,} documents")
    st.sidebar.info(f"Model: {engine.manifest.get('model_name', 'unknown')}")
except Exception as e:
    st.error(f"Loading Error: {e}")
    st.stop()

query = st.text_input("Enter claim:", placeholder="The Eiffel Tower was built in 1889.")

if st.button("🔍 Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a query!")
    else:
        with st.spinner("Searching..."):
            results = engine.search(query.strip(), top_k=top_k)

        if not results:
            st.error("Nothing found")
        else:
            st.subheader(f"Top-{len(results)} Results")
            for hit in results:
                with st.expander(f"#{hit.rank} • score = {hit.score:.4f} • {hit.title or 'No title'}", expanded=True):
                    st.write(hit.text)
                    st.caption(f"ID: `{hit.doc_id}`")
