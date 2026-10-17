import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# 1. Load environment variables from .env
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY is missing from your .env file!")

# 2. Connect to Neo4j AuraDB
graph = Neo4jGraph(database=os.getenv("NEO4J_USERNAME"))
graph.refresh_schema()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------
# Guardrails Implementation
# ---------------------------------------------------------
def input_guardrail(query: str) -> bool:
    """Blocks prompt injection or unauthorized commands."""
    forbidden = ["drop table", "ignore previous instructions", "hack", "delete database"]
    return not any(word in query.lower() for word in forbidden)

def output_guardrail(response_text: str) -> str:
    """Ensures responses are safe and grounded."""
    if not response_text or "I don't know" in response_text or "out of scope" in response_text:
        return "I'm sorry, but I couldn't find verified information regarding that scheme in the Tamil Nadu Government Knowledge Graph."
    return response_text

# ---------------------------------------------------------
# RAG Query Function with Flexible Cypher Instructions
# ---------------------------------------------------------
def ask_scheme_bot(user_query: str):
    print(f"\n💬 User Question: {user_query}")

    # Step A: Input Guardrail Check
    if not input_guardrail(user_query):
        print("🚨 [Guardrail Triggered]: Unsafe query blocked.")
        return "I'm sorry, I cannot process that request."

    # Step B: Setup Cypher Prompt with flexible keyword matching
    CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database about Tamil Nadu government schemes.
    Instructions:
    - Use only the provided nodes, relationships (such as HAS_SCHEME), and properties in the schema.
    - Always use flexible case-insensitive partial string matching (e.g., `toLower(d.id) CONTAINS 'agriculture'`).
    - If the user asks about schemes in a department, traverse from the Department node via the relationship to find connected Scheme nodes and return their ids or names.
    - Return only the executable Cypher query.

    Schema:
    {schema}

    Question:
    {question}
    """
    
    cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE)
    
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,  # Shows generated Cypher query in terminal
        allow_dangerous_requests=True
    )

    # Step C: Execute Graph RAG
    try:
        raw_result = chain.invoke({"query": user_query})["result"]
    except Exception as e:
        raw_result = "I don't know the answer."

    # Step D: Output Guardrail Check
    final_answer = output_guardrail(raw_result)
    return final_answer

# Test out a question about your ingested web page!
if __name__ == "__main__":
    query = "What schemes are available in the Agriculture and Farmers Welfare department?"
    answer = ask_scheme_bot(query)
    
    print("\n--- Final Guarded Graph RAG Answer ---")
    print(answer)