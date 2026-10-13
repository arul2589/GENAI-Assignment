from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

with open("knowledge.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)
chunks = splitter.split_text(document_text)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vector_store = InMemoryVectorStore.from_texts(
    texts=chunks,
    embedding=embedding_model
)

question = "What time does the shop open on Saturday?"
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke(question)

print("Most relevant text:")
for result in results:
    print("-", result.page_content)