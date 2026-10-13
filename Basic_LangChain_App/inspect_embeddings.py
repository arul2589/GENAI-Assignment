from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

with open("knowledge.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)
chunks = splitter.split_text(document_text)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectors = embedding_model.embed_documents(chunks)

print("Number of chunks:", len(chunks))
print("Number of vectors:", len(vectors))
print("Numbers in each vector:", len(vectors[0]))
print("First five numbers:", vectors[0][:5])