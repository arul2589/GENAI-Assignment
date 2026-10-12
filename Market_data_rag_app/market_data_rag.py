"""Single-file RAG app for the Market Data Agreement PDF.

Keep this script, `.env`, and `Market Data Agreement.pdf` in the same folder.
Run with: streamlit run market_data_rag.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# The source PDF, .env, and local FAISS data live beside this script.
APP_DIR = Path(__file__).resolve().parent
PDF_PATH = APP_DIR / "Market Data Agreement.pdf"
FAISS_DIR = APP_DIR / "faiss_index"
load_dotenv(APP_DIR / ".env")

st.set_page_config(page_title="Market Data Agreement RAG", page_icon="📄", layout="wide")
st.title("Market Data Agreement — RAG Assistant")
st.caption("Ask a question and get an answer grounded in the PDF, with page references.")


def required_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        raise RuntimeError(f"{name} is missing. Add it to a .env file beside this script.")
    return value


def embeddings() -> OpenAIEmbeddings:
    required_key("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_local_index() -> int:
    """Read the sibling PDF and save its vectors and text locally."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Put '{PDF_PATH.name}' in the same folder as this script.")

    reader = PdfReader(str(PDF_PATH))
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": PDF_PATH.name, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    if not any(page.page_content.strip() for page in pages):
        raise ValueError("No selectable text found in the PDF. OCR may be required.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")

    vectors = np.asarray(
        embeddings().embed_documents([chunk.page_content for chunk in chunks]),
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))
    with (FAISS_DIR / "chunks.json").open("w", encoding="utf-8") as file:
        json.dump(
            [{"text": doc.page_content, "metadata": doc.metadata} for doc in chunks],
            file,
            ensure_ascii=False,
            indent=2,
        )
    return len(chunks)


def load_local_index() -> tuple[Any, list[dict[str, Any]]]:
    """Load the FAISS index and matching chunk text saved by this app."""
    index_path = FAISS_DIR / "index.faiss"
    chunks_path = FAISS_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("The FAISS index is not built yet. Click Build / rebuild index.")

    index = faiss.read_index(str(index_path))
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunk data do not match. Rebuild the index.")
    return index, chunks


def answer_question(
    question: str,
    store: tuple[Any, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Retrieve similar chunks from FAISS and answer from those excerpts only."""
    index, chunks = store
    query_vector = np.asarray([embeddings().embed_query(question)], dtype="float32")
    faiss.normalize_L2(query_vector)
    _, positions = index.search(query_vector, min(4, len(chunks)))
    docs = [
        Document(page_content=chunks[i]["text"], metadata=chunks[i]["metadata"])
        for i in positions[0]
        if i >= 0
    ]
    if not docs:
        return {"answer": "I could not find relevant information in the agreement.", "sources": []}

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You answer questions about the provided market data agreement. "
            "Use only the supplied excerpts. If the answer is not stated in them, "
            "say you could not find it in the agreement. Do not provide legal advice. "
            "Cite page numbers, for example [Page 3].",
        ),
        ("human", "Question: {question}\n\nRetrieved excerpts:\n{context}"),
    ])
    model = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    response = (prompt | model).invoke({"question": question, "context": context})
    answer = response.content
    if isinstance(answer, list):
        answer = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in answer
        )

    sources = []
    seen = set()
    for doc in docs:
        key = (doc.metadata.get("source", PDF_PATH.name), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({"file": key[0], "page": key[1], "excerpt": doc.page_content})
    return {"answer": str(answer), "sources": sources}


with st.sidebar:
    st.subheader("Files and local index")
    st.write(f"PDF: `{PDF_PATH.name}`")
    st.write("FAISS vectors and text chunks: `faiss_index/` beside this script")
    if st.button("Build / rebuild index", use_container_width=True):
        try:
            with st.spinner("Reading PDF and creating the local FAISS index..."):
                count = build_local_index()
            st.success(f"Indexed {count} text chunks locally.")
        except Exception as exc:
            st.error(str(exc))
    st.divider()
    tracing = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
    st.caption(f"LangSmith tracing: {'On' if tracing else 'Off'}")
    if tracing:
        st.caption(f"Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(f"Source: {source['file']} — page {source['page']}"):
                st.write(source["excerpt"])

question = st.chat_input("Ask a question about the agreement…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching FAISS and preparing an answer..."):
                result = answer_question(question, load_local_index())
            st.markdown(result["answer"])
            for source in result["sources"]:
                with st.expander(f"Source: {source['file']} — page {source['page']}"):
                    st.write(source["excerpt"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        except Exception as exc:
            st.error(str(exc))
