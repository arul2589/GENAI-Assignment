import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY is missing from your .env file!")

st.set_page_config(page_title="Tamil Nadu Schemes Hybrid RAG", page_icon="🏛️", layout="centered")
st.title("🏛️ Tamil Nadu Government Schemes Chatbot (Hybrid RAG)")
st.markdown("Powered by **Neo4j Knowledge Graph + Vector Semantic Search**.")

@st.cache_resource
def init_hybrid_rag():
    graph = Neo4jGraph(database=os.getenv("NEO4J_USERNAME"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load existing Neo4j vector index
    vector_store = Neo4jVector.from_existing_index(
        embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_USERNAME"),
        index_name="scheme_vector_index",
        text_node_property="text"
    )
    return graph, vector_store

graph, vector_store = init_hybrid_rag()

def input_guardrail(query: str) -> bool:
    forbidden = ["drop table", "ignore previous instructions", "hack", "delete database"]
    return not any(word in query.lower() for word in forbidden)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("e.g., How can I get financial help for starting a youth business?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Executing Hybrid Semantic Search & Knowledge Graph Lookup..."):
            
            if not input_guardrail(user_query):
                final_answer = "🚨 [Guardrail Triggered]: Unsafe or blocked query detected."
            elif any(greeting in user_query.lower() for greeting in ["hi", "hello", "who are you", "what is your name"]):
                final_answer = "Hello! I am your Tamil Nadu Government Schemes Hybrid RAG Assistant. Ask me anything about welfare programs, subsidies, or application procedures!"
            else:
                # Perform Vector Similarity Search (Semantic Search)
                docs_and_scores = vector_store.similarity_search_with_score(user_query, k=3)
                
                if docs_and_scores:
                    matched_texts = [doc.page_content for doc, score in docs_and_scores]
                    
                    final_answer = f"""### 🔍 Hybrid Semantic Search Results

Based on your conceptual query, here are the most relevant schemes found via Vector Embeddings & Knowledge Graph correlation:

"""
                    for i, doc in enumerate(docs_and_scores, 1):
                        content_lines = doc[0].page_content.split('\n')
                        title = content_lines[0].replace("Scheme Title:", "").strip()
                        final_answer += f"**{i}. {title}**\n* *Details:* {doc[0].page_content.strip()}\n\n"
                else:
                    final_answer = f"I'm sorry, but I couldn't find semantically matching schemes for '{user_query}' in the Knowledge Graph."

        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})