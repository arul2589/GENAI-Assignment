# Government Multi-PDF RAG App

A one-file Streamlit app that answers questions across several PDFs using LangChain, OpenAI embeddings, and an in-memory FAISS vector index. Five supplied government PDFs are bundled in the same folder as `rag_app.py`; you can also upload more than one PDF at a time.

## Included files

- `rag_app.py` — the complete application code
- `hfw_e_cc_2026_27.pdf`
- `hfw_e_pb_2026_27.pdf`
- `hfw_e_pn_2026_27.pdf`
- `lwsd_e_pn_2026_27.pdf`
- `energy_e_pn_2026_27.pdf`

Keep the Python file and the five PDFs together in one folder. The FAISS store and chunk documents are kept in `st.session_state` as required by the assignment. They are rebuilt after the app restarts or the Streamlit session is reset.

## Windows setup

Open Command Prompt in the folder containing `rag_app.py` and the PDFs:

```bat
py -m venv venv
venv\Scripts\activate
pip install -U streamlit langchain-core langchain-openai langchain-text-splitters pypdf faiss-cpu numpy python-dotenv
```

Create a `.env` file in the same folder:

```env
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=government-multipdf-rag
OPENAI_CHAT_MODEL=gpt-4o-mini
```

Keep `.env` private and add it to `.gitignore`. LangSmith traces can contain questions, retrieved PDF excerpts, and generated answers; disable tracing with `LANGSMITH_TRACING=false` if you should not send this information to LangSmith.

## Run

```bat
streamlit run rag_app.py
```

Do not use `py rag_app.py`; that does not start the Streamlit web app.

## Build the combined index

The five bundled government PDFs are included automatically. To add more PDFs, use **Add multiple PDF files** in the sidebar and select them together. Click **Build / refresh index**. The app will:

1. Extract text page-by-page and retain the PDF filename and page number.
2. Split all documents into overlapping chunks.
3. Use OpenAI embeddings to create vectors.
4. Add all vectors to one FAISS index held in `st.session_state`.
5. Show how many chunks were indexed.

Ask a question after indexing. The answer is generated from retrieved excerpts and source expanders show the exact filename, page, and excerpt. The prompt asks the model to state disagreements between sources when the retrieved excerpts conflict.

## Notes

- The app uses the `faiss-cpu` library directly, so it does not need the sunset `langchain-community` package.
- The FAISS database is in memory for the Streamlit session. Rebuild it after restarting the app.
- PDFs must contain selectable text. Scanned image-only PDFs need OCR before indexing.
- OpenAI API usage is separate from a ChatGPT subscription and may incur API charges.
