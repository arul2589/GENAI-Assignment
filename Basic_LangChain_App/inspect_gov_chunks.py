from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_folder = Path("documents")
pages = []

# Read each PDF and save its text page by page
for pdf_path in sorted(pdf_folder.glob("*.pdf")):
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

# Split each page into smaller, overlapping pieces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(pages)

print("PDF pages with text:", len(pages))
print("Total chunks:", len(chunks))

# Show one chunk and its source information
if chunks:
    print("\nFirst chunk source:", chunks[0].metadata)
    print("First chunk preview:\n")
    print(chunks[0].page_content[:500])