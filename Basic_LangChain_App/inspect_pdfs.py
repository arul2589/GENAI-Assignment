from pathlib import Path
from pypdf import PdfReader

pdf_folder = Path("documents")
pdf_files = sorted(pdf_folder.glob("*.pdf"))

for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)

    print(f"\nFile: {pdf_path.name}")
    print("Number of pages:", len(reader.pages))

    for page_number in range(min(5, len(reader.pages))):
        page_text = reader.pages[page_number].extract_text() or ""

        if page_text.strip():
            print("First page with text:", page_number + 1)
            print("Sample:", page_text[:300].replace("\n", " "))
            break
    else:
        print("No extractable text found in the first five pages.")