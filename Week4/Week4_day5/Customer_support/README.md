# CrewAI Weekly Buildathon — Customer Support (single-file implementation)

## Matches the assignment

- Exactly three CrewAI agents: **Assistant**, **Web Search Assistant**, **Entry Agent**.
- Exactly three tasks with **Process.sequential**.
- Default **Direct Knowledge (assignment)** mode makes Assistant answer from its own LLM knowledge.
- Web Search Assistant has **SerperDevTool** and must search for original question.
- Entry Agent **calls a custom save-to-TXT tool**, which reads the original Task 1 and Task 2 outputs rather than attempting to copy long answers through LLM arguments.
- Both answers appear side by side in Streamlit; TXT is saved under `outputs/` and downloadable.
- API keys are loaded from environment or a `.env` file and are not committed.
- **All application Python code is in `app.py`.**

## Optional feature beyond assignment

Select **Hybrid RAG (FAISS + BM25)** in the sidebar, upload your searchable GenAI & DevOps knowledge-base PDF, and click **Build / load RAG index**. This uses local Hugging Face CPU embeddings, FAISS vectors, BM25 keyword retrieval and reciprocal rank fusion. The original PDF is kept in Streamlit session memory; only extracted chunks and indexes are saved under `data/hybrid_indexes/`.

The RAG option extends Agent 1 beyond the strict assignment description, so **keep Direct Knowledge selected when demonstrating exact compliance**.

## Windows (Python 3.12)

Open Command Prompt in this folder. You can use your existing `venv312` if it is active, or make a new environment:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your real `OPENAI_API_KEY` and `SERPER_API_KEY`. Then run:

```cmd
python -m streamlit run app.py
```

On your organization-managed computer, use approved execution methods. If Device Guard blocks a dependency, run on your personal VPS or an authorized machine. Do not disable or evade the security controls.

## Linux / Hostinger VPS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m streamlit run app.py --server.address 127.0.0.1
```

For public deployment, use your existing approved reverse proxy / HTTPS setup. Do not expose your `.env` or sensitive knowledge-base content.

## Quick verification

1. In **Direct Knowledge** mode, ask: `How do I troubleshoot a Kubernetes pod in CrashLoopBackOff?`
2. Check that Answer 1 and Answer 2 are present, and Answer 2 includes URLs from a real search.
3. Expand the Entry Agent section, download the TXT, and verify it matches both UI answers exactly.
4. For the optional Hybrid RAG demo, upload the 155-page Asterbridge GenAI & DevOps PDF and build the index.
5. Restart the application, upload the same PDF, and check for `Index reused` rather than generating all embeddings again.

## Data and privacy

- `data/` contains indexed document chunks and may contain sensitive content even when the original PDF is not saved.
- `outputs/` contains customer questions and answers. Both folders are in `.gitignore`.
- AI answer generation sends relevant question/context to the configured LLM provider; do not use confidential documents without authorization.
- Web search sends the user question to Serper.
- CrewAI execution tracing is disabled with `tracing=False`.
- Only load locally generated FAISS indexes: FAISS uses pickle-based metadata and must not load untrusted index files.

## Troubleshooting

- `CrewOutput has no attribute tasks`: use `crew_result.tasks_output` (already fixed).
- MuPDF DLL blocked: this project uses `pypdf`, not PyMuPDF.
- `The Entry Agent finished without calling its TXT-saving tool`: inspect the agent/tool output; the app intentionally reports non-compliance rather than pretending Agent 3 wrote the TXT.
- If the PDF upload changed, rebuild/load the corresponding index before querying it.
