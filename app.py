"""Streamlit UI for FEVER evidence search: one search box, ranked evidence below."""

import html
import os
import sys
from pathlib import Path
from urllib.parse import quote

import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fever_search.config import load_config  # noqa: E402
from fever_search.search import SearchEngine  # noqa: E402
from fever_search.text import detokenize, evidence_candidates, split_sentences  # noqa: E402

CONFIG_PATH = "configs/e5_base_opq192.yaml"
TOP_K = 10

EXAMPLES = [
    "The Eiffel Tower was built in 1889.",
    "Nikola Tesla was born in Croatia.",
    "Antarctica is the driest continent.",
]

SNIPPET_CHARS = 420


def wikipedia_url(doc_id: str) -> str:
    """A FEVER doc id is the Wikipedia article title, underscores and all."""
    return "https://en.wikipedia.org/wiki/" + quote(detokenize(doc_id), safe="()_:,'")


def render_passage(sentences: list[str], best: int | None) -> str:
    """Snippet centred on the winning sentence, which is marked; neighbours give it context."""
    if best is None:
        joined = " ".join(sentences)
        return html.escape(joined[:SNIPPET_CHARS] + ("…" if len(joined) > SNIPPET_CHARS else ""))

    parts = [f"<mark>{html.escape(sentences[best])}</mark>"]
    budget = SNIPPET_CHARS - len(sentences[best])
    before, after = best - 1, best + 1
    while budget > 0 and (before >= 0 or after < len(sentences)):
        if before >= 0 and len(sentences[before]) <= budget:
            parts.insert(0, html.escape(sentences[before]))
            budget -= len(sentences[before])
            before -= 1
        elif after < len(sentences) and len(sentences[after]) <= budget:
            parts.append(html.escape(sentences[after]))
            budget -= len(sentences[after])
            after += 1
        else:
            break
    prefix = "… " if before >= 0 else ""
    suffix = " …" if after < len(sentences) else ""
    return prefix + " ".join(parts) + suffix

st.set_page_config(page_title="FEVER Evidence Search", page_icon="🔍", layout="centered")

st.markdown(
    """
<style>
  header, #MainMenu, footer { visibility: hidden; }
  .block-container { max-width: 760px; padding-top: 4rem; }

  .brand { text-align: center; margin-bottom: 2rem; }
  .brand h1 { font-size: 2.6rem; font-weight: 650; letter-spacing: -0.02em; margin: 0; }
  .brand p { color: #6b7280; font-size: 0.95rem; margin-top: 0.4rem; }

  /* Streamlit draws the input's border on a BaseWeb wrapper, in the theme's primaryColor
     (indigo, see .streamlit/config.toml). Round every layer instead of clipping with
     overflow:hidden — clipping cuts the ends off the border that the inner layer draws. */
  div[data-testid="stTextInput"] > div,
  div[data-testid="stTextInput"] > div > div,
  div[data-testid="stTextInput"] div[data-baseweb="input"],
  div[data-testid="stTextInput"] div[data-baseweb="base-input"],
  div[data-testid="stTextInput"] input {
    border-radius: 999px;
  }
  div[data-testid="stTextInput"] input {
    height: 3.4rem;
    padding: 0 1.4rem;
    font-size: 1.05rem;
  }
  div[data-testid="stTextInput"] input:focus { outline: none; box-shadow: none; }

  /* "Press Enter to apply" badge overlapping the pill */
  div[data-testid="InputInstructions"] { display: none; }

  div[data-testid="stButton"] button {
    min-height: 2.6rem;
    border-radius: 999px;
    border: 1px solid rgba(128, 128, 128, 0.28);
    background: transparent;
    color: #6b7280;
    font-size: 0.8rem;
    font-weight: 400;
    line-height: 1.25;
    padding: 0.35rem 0.9rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  div[data-testid="stButton"] button:hover { border-color: #6366f1; color: #6366f1; }
  div[data-testid="stButton"] button:focus,
  div[data-testid="stButton"] button:active { color: #6366f1; box-shadow: none; }

  .hit { padding: 1.05rem 0; border-bottom: 1px solid rgba(128, 128, 128, 0.16); }
  .hit-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.3rem; }
  .hit-title a { color: inherit; text-decoration: none; }
  .hit-title a:hover { color: #6366f1; text-decoration: underline; }
  .hit-title a::after { content: " ↗"; font-size: 0.7em; color: #9ca3af; }
  .hit-text { color: #4b5563; font-size: 0.93rem; line-height: 1.55; }
  .hit-text mark {
    background: rgba(99, 102, 241, 0.22);
    color: inherit;
    filter: brightness(1.45);
    padding: 0.1em 0.2em;
    border-radius: 3px;
    box-shadow: inset 0 -1px 0 rgba(99, 102, 241, 0.55);
  }
  .hit-meta { color: #9ca3af; font-size: 0.76rem; margin-top: 0.45rem; font-variant-numeric: tabular-nums; }
  .note { color: #9ca3af; font-size: 0.82rem; margin: 1.6rem 0 0.2rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading index (first run only)…")
def get_engine() -> SearchEngine:
    config = load_config(CONFIG_PATH)
    config.model.device = "cpu"  # single-query encode; avoids MPS/CUDA surprises on a laptop
    return SearchEngine(config)


st.markdown(
    '<div class="brand"><h1>🔍 Evidence Search</h1>'
    "<p>Semantic search over 500k Wikipedia passages — FEVER fact verification</p></div>",
    unsafe_allow_html=True,
)

def fill_query(example: str) -> None:
    # Must be an on_click callback: assigning session_state.query after the widget exists raises.
    st.session_state.query = example


query = st.text_input(
    "Search",
    key="query",
    placeholder="Enter a claim to find supporting evidence…",
    label_visibility="collapsed",
)

for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    column.button(
        example,
        key=f"ex_{example}",
        use_container_width=True,
        on_click=fill_query,
        args=(example,),
    )

try:
    engine = get_engine()
except Exception as error:  # missing index or corpus: a message beats a raw traceback
    st.error(f"Could not load the search index: {error}")
    st.stop()

if query.strip():
    with st.spinner("Searching…"):
        hits = engine.search(query.strip(), top_k=TOP_K)

    if not hits:
        st.markdown('<p class="note">No evidence found.</p>', unsafe_allow_html=True)
    else:
        passages = [split_sentences(hit.text) for hit in hits]
        candidates = [evidence_candidates(sentences) for sentences in passages]
        evidence = engine.locate_evidence(query.strip(), hits, candidates)

        st.markdown(f'<p class="note">{len(hits)} results</p>', unsafe_allow_html=True)
        for hit, sentences, found in zip(hits, passages, evidence):
            title = html.escape(detokenize(hit.title)) or "Untitled"
            url = html.escape(wikipedia_url(hit.doc_id), quote=True)
            st.markdown(
                f'<div class="hit">'
                f'<div class="hit-title">'
                f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
                f"</div>"
                f'<div class="hit-text">{render_passage(sentences, found.sentence_index)}</div>'
                f'<div class="hit-meta">score {hit.score:.4f}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
