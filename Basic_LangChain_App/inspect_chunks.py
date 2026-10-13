from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("knowledge.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)

chunks = splitter.split_text(document_text)

print("Number of chunks:", len(chunks))

for number, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {number} ---")
    print(chunk)