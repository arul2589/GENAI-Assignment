"""Multi-PDF RAG app for the supplied government reference PDFs.

Keep rag_app.py, the included PDFs, and .env in the same folder.
Launch with: streamlit run rag_app.py
"""
from __future__ import annotations

import os
from io import BytesIO
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

APP_DIR = Path(__file__).resolve().parent
DEFAULT_PDFS = [
    "hfw_e_cc_2026_27.pdf",
    "hfw_e_pb_2026_27.pdf",
    "hfw_e_pn_2026_27.pdf",
    "lwsd_e_pn_2026_27.pdf",
    "energy_e_pn_2026_27.pdf",
]
load_dotenv(APP_DIR / ".env")

st.set_page_config(page_title="Government Documents RAG", page_icon="📚", layout="wide")
st.title("Government Documents — RAG Assistant")
st.caption("Ask questions across the supplied PDFs or add more PDFs. Answers show source files and pages.")


def required_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        raise RuntimeError(f"{name} is missing. Add it to a .env file beside rag_app.py.")
    return value


def make_embeddings() -> OpenAIEmbeddings:
    required_key("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small")


def read_pdf_bytes(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Extract PDF text page-by-page and retain source filename and page metadata."""
    reader = PdfReader(BytesIO(pdf_bytes))
    page_docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            page_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": Path(filename).name, "page": page_number},
                )
            )
    return page_docs


def build_vector_store(files: list[tuple[str, bytes]]) -> tuple[Any, list[Document], int]:
    """Chunk and embed every PDF, then create one in-memory FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks: list[Document] = []
    failures: list[str] = []
    for filename, pdf_bytes in files:
        try:
            pages = read_pdf_bytes(pdf_bytes, filename)
            if not pages:
                failures.append(f"{filename}: no selectable text was found")
                continue
            chunks.extend(splitter.split_documents(pages))
        except Exception as exc:
            failures.append(f"{filename}: {exc}")

    if failures:
        st.warning("Some PDFs could not be indexed:\n\n" + "\n".join(f"- {item}" for item in failures))
    if not chunks:
        raise ValueError("No PDF text was indexed. Check that the PDFs contain selectable text.")

    vectors = np.asarray(
        make_embeddings().embed_documents([chunk.page_content for chunk in chunks]),
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks, len(chunks)


def retrieve(question: str, store: dict[str, Any], k: int = 5) -> list[Document]:
    """Find the top matching chunks from all indexed PDFs."""
    query_vector = np.asarray(
        [make_embeddings().embed_query(question)],
        dtype="float32",
    )
    faiss.normalize_L2(query_vector)
    _, positions = store["index"].search(query_vector, min(k, len(store["chunks"])))
    return [store["chunks"][i] for i in positions[0] if i >= 0]


def answer_question(question: str, store: dict[str, Any]) -> tuple[str, list[Document]]:
    docs = retrieve(question, store)
    if not docs:
        return "I could not find relevant information in the indexed documents.", []

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question using only the retrieved excerpts from the supplied government PDFs. "
            "If the excerpts do not contain the answer, say so clearly. When sources disagree, "
            "describe the conflicting statements and cite each source instead of choosing one silently. "
            "Cite claims inline using the exact PDF filename and page, such as [hfw_e_pn_2026_27.pdf, page 12]. "
            "Do not invent facts or page numbers.",
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
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in answer
        )
    return str(answer), docs


def source_label(doc: Document) -> str:
    return f"{doc.metadata.get('source', 'unknown')} — page {doc.metadata.get('page', '?')}"


# The provided government PDFs are indexed together by default. Users may add
# more PDFs in one upload; same-named uploads replace the bundled copy.
def get_input_files(uploaded_files: list[Any]) -> list[tuple[str, bytes]]:
    by_name: dict[str, bytes] = {}
    for name in DEFAULT_PDFS:
        path = APP_DIR / name
        if path.exists():
            by_name[name] = path.read_bytes()
    for uploaded in uploaded_files:
        by_name[Path(uploaded.name).name] = uploaded.getvalue()
    return list(by_name.items())


with st.sidebar:
    st.subheader("PDF documents")
    st.write("Bundled government PDFs:")
    for filename in DEFAULT_PDFS:
        st.caption(("✓ " if (APP_DIR / filename).exists() else "Missing: ") + filename)
    additional_files = st.file_uploader(
        "Add multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select several PDFs at once. They will be added to the same FAISS index.",
    )
    if st.button("Build / refresh index", type="primary", use_container_width=True):
        try:
            inputs = get_input_files(additional_files or [])
            if not inputs:
                st.error("No PDFs found. Keep the supplied PDFs beside rag_app.py or upload PDFs above.")
            else:
                with st.spinner(f"Reading and embedding {len(inputs)} PDF file(s)..."):
                    faiss_index, chunks, chunk_count = build_vector_store(inputs)
                st.session_state["rag_store"] = {
                    "index": faiss_index,
                    "chunks": chunks,
                    "filenames": [name for name, _ in inputs],
                }
                st.session_state["rag_messages"] = []
                st.success(f"Indexed {chunk_count:,} chunks from {len(inputs)} PDFs.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    st.divider()
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
    st.caption(f"LangSmith tracing: {'On' if tracing_enabled else 'Off'}")
    if tracing_enabled:
        st.caption(f"Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")

store = st.session_state.get("rag_store")
if store:
    st.success(
        f"Ready: {len(store['chunks']):,} chunks from {len(store['filenames'])} PDFs are indexed in this session."
    )
    st.caption("Indexed files: " + ", ".join(store["filenames"]))
else:
    st.info("Click **Build / refresh index** in the sidebar before asking a question.")

if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []

for message in st.session_state["rag_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for item in message.get("sources", []):
            with st.expander(item["label"]):
                st.write(item["excerpt"])

question = st.chat_input("Ask a question across the indexed PDFs...", disabled=not store)
if question and store:
    st.session_state["rag_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching all indexed PDFs..."):
                answer, retrieved_docs = answer_question(question, store)
            st.markdown(answer)
            sources = []
            seen = set()
            for doc in retrieved_docs:
                label = source_label(doc)
                if label not in seen:
                    seen.add(label)
                    source = {"label": label, "excerpt": doc.page_content}
                    sources.append(source)
                    with st.expander(label):
                        st.write(doc.page_content)
            st.session_state["rag_messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")
