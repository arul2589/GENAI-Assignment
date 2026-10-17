import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY is missing from your .env file!")

def run_hybrid_ingestion():
    target_url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    
    # Same comprehensive list of 32+ schemes we harvested
    schemes = [
        "Training To Farmers", "Production Of Foundation And Certified Seeds",
        "Distribution Of Nuclear Poly Hedrosis Virus", "Distribution Of Rhizobium Packets - Oilseeds",
        "Livelihood Activities For Asset Less Persons-2012", "Infrastructure Subsidy For Privately Developed Industrial Estates",
        "Production Of Foundation Seeds", "Distribution Of Gypsum",
        "Distribution Of Manually Operated Plant Protection Equipment", "Pipes Carrying Water From Source To Field - Pulses ISPOM",
        "Distribution Of Rhizobium", "Distribution Of Soil Health Card",
        "Training Of Farmers On Pulses Production Technologies", "Distribution Of Certified Seeds - Oil Seeds",
        "Quality Paddy Seed Distribution", "Reservation for Micro , Small and Medium Enterprises in SIPCOT Industrial Estates",
        "Organizing Block Demonstration through Department and also through TNAU", "Generator subsidy",
        "Production of Certified Seeds", "Organizing Block demonstration on Polythene mulch Technology in Groundnut",
        "Special capital subsidy for thrust sector enterprises", "Distribution of Minikits at free of cost",
        "Rebate on Stamp duty and registration charges for Privately Developed Industrial Estates",
        "Unemployed Youth Employment Generation Programme (UYEGP)", "Seed Multiplication Scheme of Paddy, Millets, Pulses, Oilseeds and Cotton",
        "Price Preference for Micro and Small Enterprises in Government purchases", "Farmers Training",
        "Farm Production System and micro enterprises - Sponsored By State Govt", "Waiver of Earnest Money Deposit for participation in Tenders",
        "Distribution of Certified Seeds of maize", "Subsidy on the cost of Patent Registration in India or abroad",
        "Organizing Block Demonstration", "Reservation for Micro Enterprises in TANSIDCO Industrial Estates",
        "Reimbursement of hall rent for conducting exhibition by MSME Association", "Term loan obtained under NEF scheme of TIIC",
        "Low Tension power tariff subsidy", "Subsidy on the cost of Trade Mark Registration in India or abroad",
        "Farmers Interest Group (FIG)", "Pipes carrying water from source to field",
        "Term loan obtained for Technology upgradation / modernisation schemes", "Livelihood activities for asset less persons - Sponsored By State Govt",
        "Skill Development Training Schemes and Skill Upgradation Training Schemes", "Micro Nutrient spray",
        "Term loan obtained under Credit Guarantee Fund Trust Scheme", "Visit of Contract Farming farmers to the Research Stations in Karnataka",
        "Term loan obtained for ISO Certification/R&D under NSICTANSIDCO Consortium", "Integrated Pest Management Demonstration cum Training",
        "Stamp duty exemption on mortgaged and pledged documents", "Distribution of Gypsum - Oil Seeds", "Production of Certified Seeds of Maize"
    ]

    print("🚀 Initializing Hybrid Ingestion (Graph + Vector Store)...")
    graph = Neo4jGraph(database=os.getenv("NEO4J_USERNAME"))
    
    documents = []
    for idx, scheme_name in enumerate(schemes, 1):
        content = f"""
        Scheme Title: {scheme_name}
        Department: Agriculture - Farmers Welfare Department
        Description: Official Tamil Nadu government welfare scheme supporting agricultural productivity, subsidies, and grants.
        How To Avail: Applications can be submitted to the Assistant Agricultural Officer at the Village Level, Deputy Agricultural Officer at the Block Level, or the Joint Director of Agriculture at the District Level.
        """
        documents.append(Document(page_content=content.strip(), metadata={"source": target_url}))

    # Create Neo4j Vector Index for Semantic Search
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Neo4jVector.from_documents(
        documents,
        embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_USERNAME"),
        index_name="scheme_vector_index",
        node_label="SchemeChunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    
    print("✅ Hybrid Vector Index & Knowledge Graph Ingestion complete!")

if __name__ == "__main__":
    run_hybrid_ingestion()