# Create a sample file to load
text = """LangChain is a framework for building AI apps.
It helps connect language models with your own data.
RAG means Retrieval-Augmented Generation."""

with open("sample.txt", "w") as f:
    f.write(text)

print("sample.txt file created!")

#using the LangChain library to load the text file

'''
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()

print("Number of documents:", len(documents))
print("Content:")
print(documents[0].page_content)
'''

from pathlib import Path
from langchain_core.documents import Document

file_path = Path("sample.txt")
text = file_path.read_text(encoding="utf-8")

documents = [
    Document(
        page_content=text,
        metadata={"source": str(file_path)}
    )
]

print("Number of documents:", len(documents))
print("Content:")
print(documents[0].page_content)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
)

chunks = splitter.split_text(documents[0].page_content)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# A small, cheap, high-quality OpenAI embedding model
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key = api_key)

print("OpenAI embedding model is ready!")

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_texts(chunks, openai_embeddings)

print("All texts are stored in the vector store!")


retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # return top 2 results

print("Retriever is ready!")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt: a template with a blank {topic} to fill in
prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")

# 2. Model: the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")

# 3. Output parser: turn the model's reply into plain text
output_parser = StrOutputParser()

print("All three pieces are ready!")

chain = prompt | model | output_parser

print("Chain is ready!")

answer = chain.invoke({"topic": "Python"})

print(answer)