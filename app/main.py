import os
import shutil
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.extractor import extract_all
from app.embedder import describe_images, build_vector_store
from app.retriever import retrieve_multimodal
from app.generator import generate_answer

st.set_page_config(
    page_title="Multimodal RAG — Attention Is All You Need",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Pipeline Control")

kb_exists = os.path.exists("./chroma_db") and os.path.isdir("./chroma_db")
if kb_exists:
    st.sidebar.success("Knowledge base ready")
else:
    st.sidebar.warning("Run ingestion first")

st.sidebar.metric("Source", "Attention Is All You Need Paper")
st.sidebar.metric("Model", "gemini-2.5-flash")
st.sidebar.metric("Embeddings", "text-embedding-004")
st.sidebar.metric("Vector Store", "ChromaDB (cosine similarity)")

st.sidebar.divider()

if st.sidebar.button("Rebuild Knowledge Base"):
    pdf_path = "data/Attention Is All You Need - Research Paper.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = "data/attention.pdf"
        if not os.path.exists(pdf_path):
            st.sidebar.error("Source PDF not found at data/Attention Is All You Need - Research Paper.pdf")
            st.stop()

    try:
        progress_bar = st.sidebar.progress(0)
        st.sidebar.text("[Step 1/3] Extracting content...")
        extracted = extract_all(pdf_path)
        progress_bar.progress(33)

        st.sidebar.text("[Step 2/3] Describing figures with Gemini Vision...")
        image_descriptions = describe_images(extracted["images"])
        progress_bar.progress(66)

        st.sidebar.text("[Step 3/3] Building ChromaDB store...")
        build_vector_store(extracted["text_chunks"], image_descriptions)
        progress_bar.progress(100)

        st.sidebar.success("Done! Knowledge base rebuilt.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {str(e)}")

# Main Area
st.title("Multimodal RAG — Transformer Paper")
st.caption("Grounded Q&A over text, tables, and figures")
st.divider()

col1, col2, col3 = st.columns(3)

if col1.button("BLEU scores table?"):
    st.session_state["prefill"] = "What BLEU score did the big Transformer achieve on WMT 2014 English-German translation task?"

if col2.button("Encoder-decoder architecture?"):
    st.session_state["prefill"] = "Describe the encoder and decoder architecture shown in Figure 1. What are the main components?"

if col3.button("What is multi-head attention?"):
    st.session_state["prefill"] = "What is multi-head attention? How many heads were used and what dimensions were used for each head?"

default_query = st.session_state.get("prefill", "")

with st.form("query_form"):
    user_query = st.text_input("Ask a question about the paper:", value=default_query)
    submitted = st.form_submit_button("Ask Question", type="primary")

if submitted:
    if not user_query.strip():
        st.warning("Enter a question")
        st.stop()

    try:
        col_answer, col_context = st.columns([3, 2])

        with st.spinner("Retrieving + generating answer..."):
            chunks = retrieve_multimodal(user_query, n_results=5)
            result = generate_answer(user_query, chunks["all"])
            st.session_state["last_result"] = (user_query, chunks, result)

        with col_answer:
            st.subheader("Answer")
            st.write(result["answer"])
            st.divider()

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Sources Used", result["sources_used"])
            mcol2.metric(
                "Pages Referenced",
                ", ".join(str(p) for p in result["pages_referenced"]) if result["pages_referenced"] else "None"
            )
            mcol3.metric(
                "Content Types",
                ", ".join(result["source_types"]) if result["source_types"] else "None"
            )

        with col_context:
            st.subheader("Retrieved Context")
            for chunk in chunks["all"]:
                ctype = chunk["metadata"].get("type", "unknown")
                page = chunk["metadata"].get("page", "?")
                score = chunk.get("relevance_score", "?")
                icon = {
                    "text": "📄",
                    "heading": "📌",
                    "table": "📊",
                    "image_description": "🖼️"
                }.get(ctype, "📄")

                with st.expander(f"{icon} {ctype.upper()} | Page {page} | Score {score}"):
                    st.write(
                        chunk["content"][:600] + "..."
                        if len(chunk["content"]) > 600
                        else chunk["content"]
                    )

    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

elif "last_result" in st.session_state:
    prev_query, chunks, result = st.session_state["last_result"]
    col_answer, col_context = st.columns([3, 2])

    with col_answer:
        st.subheader("Answer")
        st.write(result["answer"])
        st.divider()

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Sources Used", result["sources_used"])
        mcol2.metric(
            "Pages Referenced",
            ", ".join(str(p) for p in result["pages_referenced"]) if result["pages_referenced"] else "None"
        )
        mcol3.metric(
            "Content Types",
            ", ".join(result["source_types"]) if result["source_types"] else "None"
        )

    with col_context:
        st.subheader("Retrieved Context")
        for chunk in chunks["all"]:
            ctype = chunk["metadata"].get("type", "unknown")
            page = chunk["metadata"].get("page", "?")
            score = chunk.get("relevance_score", "?")
            icon = {
                "text": "📄",
                "heading": "📌",
                "table": "📊",
                "image_description": "🖼️"
            }.get(ctype, "📄")

            with st.expander(f"{icon} {ctype.upper()} | Page {page} | Score {score}"):
                st.write(
                    chunk["content"][:600] + "..."
                    if len(chunk["content"]) > 600
                    else chunk["content"]
                )
