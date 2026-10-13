from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load the saved FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    "government_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# Retrieve the 3 passages most relevant to the question
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Tell the model how to use the retrieved passages
prompt = ChatPromptTemplate.from_messages([
    (
        "Answer using only facts stated in the context. "
"If the question is broad or only partly covered, share the relevant facts "
"the context does provide and say what the documents do not cover. "
"If there are no relevant facts, say you don't know based on these documents. "
"Include the source filename and PDF page for the facts you use.\n\n"
"Context:\n{context}"
    ),
    ("human", "{question}"),
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
answer_chain = prompt | model | StrOutputParser()

question = input("Ask a question about the government PDFs: ")

documents = retriever.invoke(question)

context = "\n\n".join(
    f"Source: {doc.metadata['source']}, PDF page: {doc.metadata['page']}\n"
    f"{doc.page_content}"
    for doc in documents
)

answer = answer_chain.invoke({
    "context": context,
    "question": question,
})

print("\nAI answer:")
print(answer)