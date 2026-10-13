from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

sample_text = "Saturday opening hours are 10:00 AM to 2:00 PM."

vectors = embeddings.embed_documents([sample_text])
vector = vectors[0]

print("Number of vectors:", len(vectors))
print("Numbers in this vector:", len(vector))
print("First five numbers:", vector[:5])