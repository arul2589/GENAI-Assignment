"""Streamlit frontend for the three-agent CrewAI customer-support demo."""

from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from crew_agents import run_support_crew
from rag import build_or_load_index

load_dotenv()
st.set_page_config(page_title="CrewAI Hybrid RAG Customer Support", page_icon="📚", layout="wide")

st.title("📚 CrewAI Hybrid RAG — 3 Sequential Agents")
st.caption("Agent 1: Hugging Face + FAISS + BM25 | Agent 2: web search | Agent 3: TXT report")

with st.sidebar:
    st.header("1. Upload your knowledge base")
    uploaded_pdf = st.file_uploader(
        "Select your legally obtained book PDF", type=["pdf"], accept_multiple_files=False
    )
    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        current_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if st.button("Build / load RAG index", type="primary", use_container_width=True):
            with st.spinner("Extracting PDF, generating local Hugging Face embeddings, and building FAISS + BM25..."):
                try:
                    metadata = build_or_load_index(pdf_bytes)
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
                else:
                    st.session_state["index_dir"] = metadata["index_dir"]
                    st.session_state["active_hash"] = metadata["pdf_hash"]
                    st.session_state["index_metadata"] = metadata
                    st.success("Existing hybrid index loaded!" if metadata["reused"] else "FAISS + BM25 index created!")

        if st.session_state.get("active_hash") != current_hash:
            st.info("Click 'Build / load RAG index' to use this selected PDF.")

    metadata = st.session_state.get("index_metadata")
    if metadata:
        st.divider()
        st.write("**Active knowledge base**")
        st.write(f"PDF pages: **{metadata['pages']}**")
        st.write(f"Pages with text: **{metadata['text_pages']}**")
        st.write(f"RAG chunks: **{metadata['chunks']}**")
        st.caption("Retriever: FAISS + BM25 with Reciprocal Rank Fusion")
        st.caption("Embedding model: all-MiniLM-L6-v2 (local CPU)")
        st.caption("PDF page numbers refer to physical PDF pages, which may differ from printed page numbers.")

    st.divider()
    st.markdown("**API configuration**")
    st.write("OpenAI:", "✅" if os.getenv("OPENAI_API_KEY") else "❌ Missing")
    st.write("Serper:", "✅" if os.getenv("SERPER_API_KEY") else "❌ Missing")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("2. Ask the support agents")
st.write("Try: *What is the creature inside the Chamber of Secrets?*")

for idx, result in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(result["question"])
    with st.chat_message("assistant"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Agent 1 — Hybrid RAG")
            st.write(result["answer_1"])
        with right:
            st.markdown("#### Agent 2 — Web Search")
            st.write(result["answer_2"])
        with st.expander("Agent 3 — Report details"):
            st.write(result["agent_3_output"])
            st.code(result["report_path"])
        st.download_button(
            "Download .txt report",
            data=result["report_text"],
            file_name=result["report_path"].split("/")[-1].split("\\")[-1],
            mime="text/plain",
            key=f"download_{idx}",
        )

question = st.chat_input("Ask a question about the book...")
if question:
    if not st.session_state.get("index_dir"):
        st.error("First upload the PDF and build the RAG index in the sidebar.")
    elif uploaded_pdf is not None and st.session_state.get("active_hash") != hashlib.sha256(uploaded_pdf.getvalue()).hexdigest():
        st.error("The selected PDF has changed. Build its index before asking questions.")
    elif not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
        st.error("Set OPENAI_API_KEY and SERPER_API_KEY in .env before asking a question.")
    else:
        with st.spinner("Running Agent 1 → Agent 2 → Agent 3 sequentially..."):
            try:
                result = run_support_crew(question, st.session_state["index_dir"])
            except Exception as exc:
                st.error(f"The agents could not complete the request: {exc}")
            else:
                st.session_state.history.append(result)
                st.rerun()
