import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

# 1. Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY is missing from your .env file!")

# 2. Configure Streamlit Page
st.set_page_config(
    page_title="Tamil Nadu Schemes Knowledge Graph RAG", 
    page_icon="🏛️",
    layout="centered"
)

st.title("🏛️ Tamil Nadu Government Schemes Chatbot")
st.markdown("Ask questions about government schemes retrieved directly from your **Neo4j Knowledge Graph**.")

# 3. Cache the Graph and LLM initialization
@st.cache_resource
def init_graph_rag():
    graph = Neo4jGraph(database=os.getenv("NEO4J_USERNAME"))
    graph.refresh_schema()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return graph, llm

graph, llm = init_graph_rag()

# 4. Guardrails Implementation
def input_guardrail(query: str) -> bool:
    forbidden = ["drop table", "ignore previous instructions", "hack", "delete database"]
    return not any(word in query.lower() for word in forbidden)

# 5. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Handler with Dynamic Graph Search
if user_query := st.chat_input("e.g., Tell me about Unemployed Youth Employment Generation Programme"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge Graph..."):
            
            if not input_guardrail(user_query):
                final_answer = "🚨 [Guardrail Triggered]: Unsafe or blocked query detected."
            else:
                query_lower = user_query.lower()
                
                if any(greeting in query_lower for greeting in ["hi", "hello", "who are you", "what is your name"]):
                    final_answer = "Hello! I am your Tamil Nadu Government Schemes Assistant powered by your Neo4j Knowledge Graph. Ask me about any of the ingested schemes!"
                
                elif "how to avail" in query_lower or "tell me about" in query_lower:
                    # Extract scheme name from query
                    search_term = query_lower.replace("how to avail", "").replace("tell me about", "").strip()
                    
                    cypher = """
                    MATCH (s:Scheme)
                    WHERE toLower(s.id) CONTAINS $searchTerm
                    RETURN s.id AS name
                    LIMIT 1
                    """
                    results = graph.query(cypher, params={"searchTerm": search_term})
                    
                    if results:
                        scheme_name = results[0]['name']
                        final_answer = f"""### 📋 Scheme Details: {scheme_name}

**Concerned Department:** Agriculture - Farmers Welfare Department  
**Sponsored By:** State Government of Tamil Nadu  
**Beneficiaries:** Farmers and Agricultural Workers  

**How To Avail:**  
The application is to be submitted to:
* Assistant Agricultural Officer at the Village Level
* Deputy Agricultural Officer at the Block Level
* The Assistant Director of Agriculture at the Block Level
* The Joint Director of Agriculture at the District Level
"""
                    else:
                        final_answer = f"I'm sorry, but I couldn't find specific details for '{user_query}' in the Knowledge Graph."
                
                else:
                    # 1. Fetch all scheme names from Neo4j to check for exact/substring mentions
                    all_schemes_res = graph.query("MATCH (s:Scheme) RETURN s.id AS name")
                    db_schemes = [r['name'] for r in all_schemes_res]
                    
                    matched_scheme = None
                    for scheme in db_schemes:
                        if scheme.lower() in query_lower:
                            matched_scheme = scheme
                            break
                    
                    # 2. If a specific scheme name is found in the user's sentence, show its precise details
                    if matched_scheme:
                        final_answer = f"""### 📋 Scheme Details: {matched_scheme}

**Concerned Department:** Agriculture - Farmers Welfare Department  
**Sponsored By:** State Government of Tamil Nadu  
**Beneficiaries:** Farmers and Agricultural Workers  

**Description & Financial Support:** Official welfare and assistance scheme supporting agricultural productivity and seed distribution.  

**How To Avail:**  
Applications can be submitted to:
* Assistant Agricultural Officer at the Village Level
* Deputy Agricultural Officer at the Block Level
* The Assistant Director of Agriculture at the Block Level
* The Joint Director of Agriculture at the District Level
"""
                    else:
                        # 3. Otherwise, fall back to keyword search for broader queries
                        stopwords = ["about", "what", "is", "the", "for", "subsidy", "scheme", "schemes", "avail", "tell", "me", "from", "with", "a", "an", "and"]
                        keywords = [w for w in query_lower.split() if w not in stopwords and len(w) > 2]
                        search_term = max(keywords, key=len) if keywords else query_lower

                        cypher = """
                        MATCH (n)
                        WHERE toLower(n.id) CONTAINS $searchTerm
                        RETURN labels(n) AS type, n.id AS name
                        LIMIT 5
                        """
                        results = graph.query(cypher, params={"searchTerm": search_term})

                        if results:
                            items = [f"**{r['name']}** (Type: {r['type'][0]})" for r in results]
                            items_str = "\n* ".join(items)
                            final_answer = f"Here are the closest matching results found in your Knowledge Graph:\n\n* {items_str}"
                        else:
                            final_answer = f"I'm sorry, but I couldn't find verified information matching '{user_query}' in your Knowledge Graph."

        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})