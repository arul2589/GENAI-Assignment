from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the index created by your own build_faiss_index.py script
vector_store = FAISS.load_local(
    "government_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

question = input("Ask a question about the government PDFs: ")

# Find the 3 chunks most similar to your question
results = vector_store.similarity_search(question, k=3)

for number, document in enumerate(results, start=1):
    print(f"\n--- Result {number} ---")
    print("Source:", document.metadata["source"])
    print("Page:", document.metadata["page"])
    print(document.page_content[:500])