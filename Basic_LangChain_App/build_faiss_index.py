from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

pdf_folder = Path("documents")
pdf_files = sorted(pdf_folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found inside the documents folder.")

pages = []

# Read the text from each PDF page
for pdf_path in pdf_files:
    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

print("PDF pages with text:", len(pages))

# Split pages into smaller pieces for searching
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
chunks = splitter.split_documents(pages)

print("Total chunks:", len(chunks))
print("Sending chunks to OpenAI for embeddings...")

# Convert the chunks to vectors and store them in FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# Save the index on your computer for later use
vector_store.save_local("government_faiss_index")

print("FAISS index saved in the government_faiss_index folder.")