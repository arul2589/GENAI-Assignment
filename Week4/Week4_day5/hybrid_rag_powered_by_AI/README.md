# CrewAI Three-Agent Customer Support — FAISS + BM25 Hybrid RAG

A beginner-friendly, sequential CrewAI + Streamlit demo. Bring your own legally obtained, text-selectable PDF (for example, *Harry Potter and the Chamber of Secrets*). The book itself is **not included**.

## System design

```text
Book PDF → pypdf → LangChain text chunks with PDF page numbers
                 ├── Hugging Face all-MiniLM-L6-v2 (local CPU) → FAISS
                 └── tokenized text → BM25 (rank-bm25)
Question → FAISS top candidates + BM25 top candidates → RRF → Top 5 chunks

Streamlit question
    ↓
Agent 1: book knowledge + custom Hybrid RAG search tool (page references)
    ↓
Agent 2: independent Serper web search + an answer with URLs
    ↓
Agent 3: TXT writer tool combines question, Answer 1, Answer 2
    ↓
Streamlit shows both original answers + downloadable report
```

The first PDF indexing step **does not call an OpenAI embedding API or any generative LLM**. The Hugging Face embedding model is downloaded from Hugging Face on first use and then cached. Agent responses do use a generative LLM; web search uses Serper.

## Requirements

- Python **3.12** recommended, or Docker on Hostinger VPS.
- CPU and enough available RAM for PyTorch/sentence-transformers alongside other services. An 8 GB VPS with unused headroom is a more comfortable starting point; smaller instances may need external embeddings or local pre-indexing.
- `OPENAI_API_KEY` and `SERPER_API_KEY` for the full agent workflow, but **neither is needed to index the PDF**.
- A text-selectable, legally obtained PDF. Scanned pages need OCR before indexing.

## Setup — Windows

Extract the ZIP and open a terminal in the folder containing `app.py`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```ini
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
CREW_LLM_MODEL=gpt-4o-mini
```

Run:

```powershell
streamlit run app.py
```

Open http://localhost:8501. Upload your PDF, click **Build / load RAG index**, then ask a question. You can create the index **without API keys**, and add keys when you want to run the agents.

## Setup — Hostinger VPS with Docker Compose

From the project directory on the VPS:

```bash
cp .env.example .env
nano .env  # enter the API keys; do not commit this file
mkdir -p data outputs hf_cache
docker compose up -d --build
docker compose logs -f customer-support
```

The Streamlit container listens on `127.0.0.1:8510` on the **VPS only**; it is intentionally not publicly exposed. Use your existing authenticated HTTPS reverse proxy or an SSH tunnel for private access. Do not expose an unauthenticated file-upload interface to the internet.

For a temporary SSH tunnel from Windows:

```bash
ssh -L 8510:127.0.0.1:8510 VPS_USER@VPS_HOST
```

Then visit http://localhost:8510 on the Windows computer. The indexed data, report files and embedding model cache persist in `./data`, `./outputs` and `./hf_cache` respectively.

**If port 8510 is already used**, change the left-hand port in `compose.yaml`. If your VPS is short on RAM, run the indexing job on your Windows PC instead, then copy the complete `data/hybrid_indexes` folder to the VPS. The same `all-MiniLM-L6-v2` model must be used for question embeddings (including the same normalization setting).

## Main files

- `rag.py`: PDF text extraction, chunks, local Hugging Face embeddings, FAISS and BM25 indices, hybrid search.
- `retrieval_utils.py`: tokenization and reciprocal rank fusion (RRF).
- `crew_agents.py`: 3 CrewAI agents, sequential tasks, custom RAG and report-writing tools.
- `app.py`: Streamlit upload, index build/reuse, question input, answers and TXT download.
- `requirements.txt`: Python packages.
- `compose.yaml`, `Dockerfile`: VPS deployment.
- `tests/test_retrieval_utils.py`: quick unit tests for the ranking helpers.

## How indexing and searching work

1. The PDF is read page-by-page, preserving **physical PDF page numbers**.
2. LangChain splits the text into 900-character chunks with a 150-character overlap.
3. Hugging Face `sentence-transformers/all-MiniLM-L6-v2` creates 384-dimensional embeddings on your CPU.
4. FAISS saves the vector index to `data/hybrid_indexes/<hash>-<version>/index.faiss` and an internal metadata pickle; JSON stores matching chunks for BM25.
5. `rank-bm25` calculates keyword relevance in memory when the index is first loaded (cached thereafter).
6. For each customer question, FAISS retrieves semantically similar chunks and BM25 retrieves exact-term matches.
7. Reciprocal Rank Fusion combines **stable chunk IDs** and returns the top 5 chunks, including pages.
8. Agent 1 answers from grounded PDF passages; Agent 2 independently checks the web; Agent 3 receives their previous task outputs and saves the TXT file.

PDF text is private to the application, but **retrieved excerpts are transmitted to the configured generative LLM API** when Agent 1 answers a question. Do not upload documents that cannot be sent to that provider.

## Verification

Test the dependency-free ranking functions:

```bash
python -m unittest discover -s tests -v
```

Check Agent 1 retrieval on the uploaded PDF by opening Streamlit, building the index, and asking questions such as "Who is Dobby?", "Who controlled the Basilisk?" or "Why did Dobby warn Harry?".

## Demo limitations / operational notes

- Full model installation, model download, CrewAI execution and live Serper web searches need your machine, internet access and API keys; none is simulated by this starter project.
- FAISS `load_local` uses Python pickle for metadata. **Never load untrusted index files**, and keep `data/` protected from modification by other users.
- This example assumes one private user/session and makes no authentication, rate-limit, tenant-isolation, abuse-prevention or privacy-compliance guarantees. Add those before using it for real customers.
- A copyright-protected PDF is not distributed with this project. Keep your copy private and comply with its usage rights.
- BM25 is reconstructed from local JSON per process, so it uses RAM. This is fine for a book-sized corpus.
- LLM answers can be wrong. Page references should be checked against the passages actually retrieved.
- The report is finally written deterministically from the actual first and second task outputs, even if Agent 3 skips its tool or paraphrases.

## Windows organizational application-control note

If your device blocks the native PyMuPDF DLL (`_mupdf`), this revision uses
`pypdf` for PDF text extraction. `pypdf` is pure Python, but the Hugging Face,
PyTorch and FAISS components may still require native libraries. Follow your
organization's software security policies; if those dependencies are blocked,
run the app on an authorized personal machine or your Hostinger VPS.

When starting Streamlit in an environment where the launcher is blocked but
Python itself is allowed, try `python -m streamlit run app.py`.
