"""Private PDF Hybrid RAG: local Hugging Face embeddings + FAISS + BM25 + RRF.

Indexing is offline/API-free, apart from the one-time Hugging Face model download.
The generative LLM is used only when CrewAI agents answer user questions.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from io import BytesIO

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from retrieval_utils import tokenize, reciprocal_rank_fusion

INDEX_ROOT = Path("data") / "hybrid_indexes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_VERSION = "hybrid-v2-pypdf-900-150-minilm"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def _path_for_hash(pdf_hash: str) -> Path:
    return INDEX_ROOT / f"{pdf_hash[:24]}-{INDEX_VERSION}"


def build_or_load_index(pdf_bytes: bytes) -> dict:
    """Build both indices from one PDF; re-use persistent indices on later runs."""
    if not pdf_bytes:
        raise ValueError("Please upload a non-empty PDF.")

    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    index_dir = _path_for_hash(pdf_hash)
    metadata_file = index_dir / "metadata.json"
    chunks_file = index_dir / "chunks.json"

    required = [
        index_dir / "index.faiss",
        index_dir / "index.pkl",
        chunks_file,
        metadata_file,
    ]
    if all(path.is_file() for path in required):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (metadata.get("pdf_hash") == pdf_hash and
                metadata.get("embedding_model") == EMBEDDING_MODEL and
                metadata.get("index_version") == INDEX_VERSION):
            return {**metadata, "index_dir": str(index_dir), "reused": True}

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise ValueError("This PDF is password-protected; use an unlocked copy.")
        page_count = len(reader.pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc

    documents = []
    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(page_content=text, metadata={"page": page_index + 1})
            )

    if not documents:
        raise ValueError("No selectable text found. OCR the scanned PDF first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The PDF did not yield any searchable text chunks.")

    for index, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = index

    # FAISS stores dense vectors; BM25 can be reconstructed from the same chunk text.
    vectorstore = FAISS.from_documents(chunks, _embeddings())
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    # JSON keeps the BM25 corpus inspectable and avoids storing an extra pickle.
    chunk_records = [
        {"id": i, "page": doc.metadata["page"], "text": doc.page_content}
        for i, doc in enumerate(chunks)
    ]
    chunks_file.write_text(
        json.dumps(chunk_records, ensure_ascii=False), encoding="utf-8"
    )

    metadata = {
        "pdf_hash": pdf_hash,
        "pages": page_count,
        "text_pages": len(documents),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "index_version": INDEX_VERSION,
        "retrieval": "FAISS + BM25 + Reciprocal Rank Fusion",
    }
    # Write metadata last, so an interrupted build is never treated as complete.
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _load_indexes.cache_clear()
    return {**metadata, "index_dir": str(index_dir), "reused": False}


@lru_cache(maxsize=2)
def _load_indexes(index_dir: str):
    directory = Path(index_dir).resolve()
    root = INDEX_ROOT.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Index directory must be within the application's data folder.")
    if not (directory / "index.faiss").is_file():
        raise FileNotFoundError("Build the PDF knowledge base first.")

    # Security: FAISS uses a pickle for metadata. Only load indices created by
    # this application. Never import arbitrary third-party index files.
    store = FAISS.load_local(
        str(directory), _embeddings(), allow_dangerous_deserialization=True
    )
    records = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    bm25 = BM25Okapi([tokenize(record["text"]) for record in records])
    return store, records, bm25


def retrieve_hybrid(question: str, index_dir: str, k: int = 5) -> list[dict]:
    """Search FAISS and BM25 independently; return fused source excerpts."""
    question = question.strip()
    if not question:
        return []
    k = max(1, min(k, 10))
    store, records, bm25 = _load_indexes(str(index_dir))
    candidate_count = min(len(records), max(k * 3, 12))

    dense_docs = store.similarity_search(question, k=candidate_count)
    dense_ids = [int(doc.metadata["chunk_id"]) for doc in dense_docs]

    query_tokens = tokenize(question)
    sparse_ids: list[int] = []
    if query_tokens:
        bm25_scores = bm25.get_scores(query_tokens)
        sparse_ids = sorted(
            (i for i, score in enumerate(bm25_scores) if score > 0),
            key=lambda i: (-bm25_scores[i], i),
        )[:candidate_count]

    ranked_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)
    return [records[chunk_id] for chunk_id in ranked_ids[:k]]


def search_pdf(question: str, index_dir: str, k: int = 5) -> str:
    """Text result safe to insert into Agent 1's prompt and custom RAG tool."""
    matches = retrieve_hybrid(question, index_dir=index_dir, k=k)
    if not matches:
        return "No relevant passages were found."
    return "\n\n".join(
        f"Passage {i} — PDF page {doc['page']} (chunk {doc['id']}):\n"
        f"{doc['text'][:1400]}"
        for i, doc in enumerate(matches, start=1)
    )
