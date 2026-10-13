from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Read the source document
with open("knowledge.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

# Split the document into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)
chunks = splitter.split_text(document_text)

# Convert the chunks to vectors and store them
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore.from_texts(
    texts=chunks,
    embedding=embedding_model
)

# Search the vector store for relevant chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Tell the model to answer using the retrieved context
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using only the context below. "
        "If the answer is not in the context, say you don't know based on the document.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

model = ChatOllama(model="qwen2.5:3b")
output_parser = StrOutputParser()
answer_chain = prompt_template | model | output_parser

# Ask a question, retrieve context, and generate an answer
question = input("Ask a question about GreenLeaf Books: ")

retrieved_chunks = retriever.invoke(question)
context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)

answer = answer_chain.invoke({
    "context": context,
    "question": question
})

print("\nAI:", answer)