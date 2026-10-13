from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_FOLDER = BASE_DIR / "government_faiss_index"


@st.cache_resource
def load_rag_components():
    # Load the saved FAISS index and connect it to the same embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.load_local(
        str(INDEX_FOLDER),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer using only facts stated in the context. "
            "If the question is broad or only partly covered, share the "
            "relevant facts the context does provide and say what the "
            "documents do not cover. If there are no relevant facts, say "
            "you don't know based on these documents. Include the source "
            "filename and PDF page for the facts you use.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer_chain = prompt | model | StrOutputParser()

    return retriever, answer_chain


st.title("Tamil Nadu Government PDF Assistant")
st.write("Ask a question about the government documents.")

question = st.text_input(
    "Your question",
    placeholder="For example: Tell me about solar energy",
)

if st.button("Get answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the documents and preparing an answer..."):
                retriever, answer_chain = load_rag_components()
                documents = retriever.invoke(question)

                context = "\n\n".join(
                    f"Source: {doc.metadata['source']}, "
                    f"PDF page: {doc.metadata['page']}\n"
                    f"{doc.page_content}"
                    for doc in documents
                )

                answer = answer_chain.invoke({
                    "context": context,
                    "question": question,
                })

            st.subheader("Answer")
            st.write(answer)

            with st.expander("View retrieved passages"):
                for doc in documents:
                    st.markdown(
                        f"**{doc.metadata['source']} — "
                        f"PDF page {doc.metadata['page']}**"
                    )
                    st.write(doc.page_content)

        except Exception as error:
            st.error(f"Something went wrong: {error}")